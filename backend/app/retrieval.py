"""混合检索：BM25 + 向量 → RRF 融合 → 可选重排。

设计要点
--------
- **RRF（Reciprocal Rank Fusion）**：不依赖两路分数尺度，天然可比，
  这是稀疏/稠密融合最稳的做法
- **查询改写**：把口语化问题（"防风草能卖多少钱"）扩展为
  wiki 术语（"防风草 卖价 sellprice"），提升 BM25 命中
- **结构化优先**：命中 infobox/table 块时加权，因为数值题的答案都在那里
- **重排（可选）**：用小模型或规则打分，控制延迟
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.app.embedding import get_embedder
from backend.app.store import Store


@dataclass
class Retrieved:
    chunk_id: int
    title: str
    section: str | None
    text: str
    score: float
    kind: str = 'text'
    sources: list[str] = field(default_factory=list)  # bm25 / vec

    def as_dict(self) -> dict:
        return {
            'chunk_id': self.chunk_id,
            'title': self.title,
            'section': self.section,
            'text': self.text,
            'score': round(self.score, 4),
            'kind': self.kind,
            'sources': self.sources,
        }


# ---------------------------------------------------------------- 查询处理

# 口语 → wiki 术语映射（游戏问答高频词）
SYNONYMS = {
    '卖多少钱': ['卖价', '售价'],
    '能卖多少': ['卖价', '售价'],
    '值多少钱': ['卖价', '售价'],
    '价格': ['卖价', '售价'],
    '怎么种': ['生长', '种植', '生长时间'],
    '多久成熟': ['生长时间', 'growth'],
    '几天': ['生长时间', 'growth'],
    '什么时候种': ['季节'],
    '在哪钓': ['钓鱼', '地点', '位置'],
    '哪里钓': ['钓鱼', '地点', '位置'],
    '吃什么': ['喜好', '最爱'],
    '喜欢什么': ['喜好', '最爱'],
    '送礼': ['喜好', '礼物'],
    '怎么获得': ['获得', '来源'],
    '怎样得到': ['获得', '来源'],
    '有什么用': ['用途', '用途说明'],
    '能做啥': ['用途', '配方'],
    '几星': ['品质', '星级'],
    '好卖': ['卖价'],
}

# 数值题型识别（用于决定是否触发计算工具）
RE_NUMERIC_Q = re.compile(
    r'多少|几[个天星级]|价格|卖价|售价|收益|利润|值|赚|倍|×|\d+\s*(?:金|倍|个|天|星)')
RE_COMPARE_Q = re.compile(r'哪个|对比|比较|还是|更(?:好|高|值|赚)|vs|和.*比')
RE_CALC_Q = re.compile(r'金星|银星|铱星|品质|果酒|腌菜|果汁|加工|酿|做成|变卖')


def expand_query(query: str) -> str:
    """把口语查询扩展为 wiki 友好查询串。"""
    extra: list[str] = []
    for k, vs in SYNONYMS.items():
        if k in query:
            extra.extend(vs)
    if not extra:
        return query
    return query + ' ' + ' '.join(dict.fromkeys(extra))


def classify(query: str) -> dict[str, bool]:
    return {
        'numeric': bool(RE_NUMERIC_Q.search(query)),
        'compare': bool(RE_COMPARE_Q.search(query)),
        'calc': bool(RE_CALC_Q.search(query)),
    }


# ---------------------------------------------------------------- 检索器


class HybridRetriever:
    def __init__(self, store: Store, top_k_dense: int = 20, top_k_sparse: int = 20,
                 rrf_k: int = 60) -> None:
        self.store = store
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 6,
                 kinds_boost: set[str] | None = None) -> list[Retrieved]:
        q = expand_query(query)

        sparse = self.store.search_bm25(q, self.top_k_sparse)
        if not sparse:
            sparse = self.store.search_bm25(query, self.top_k_sparse)

        dense: list[dict] = []
        try:
            emb = get_embedder().encode_one(q)
            dense = self.store.search_vec(emb, self.top_k_dense)
        except Exception:
            pass

        fused = self._rrf(sparse, dense)
        if not fused:
            return []

        # 结构化块加权：数值/属性块优先
        boost = kinds_boost or {'infobox': 1.25, 'table': 1.15}
        for r in fused:
            if r.kind in boost:
                r.score *= boost[r.kind]
            # 标题精确度加权：解决「防风草」被「防风草种子/防风草汤」压过的问题
            r.score *= _title_affinity(query, r.title)

        fused.sort(key=lambda r: -r.score)
        return fused[:top_k]

    def _rrf(self, sparse: list[dict], dense: list[dict]) -> list[Retrieved]:
        acc: dict[int, Retrieved] = {}
        k = self.rrf_k

        for rank, item in enumerate(sparse, start=1):
            cid = item['chunk_id']
            r = acc.get(cid) or Retrieved(
                chunk_id=cid, title=item['title'], section=item.get('section'),
                text=item['text'], score=0.0,
            )
            r.score += 1.0 / (k + rank)
            r.sources.append('bm25')
            acc[cid] = r

        for rank, item in enumerate(dense, start=1):
            cid = item['chunk_id']
            r = acc.get(cid) or Retrieved(
                chunk_id=cid, title=item['title'], section=item.get('section'),
                text=item['text'], score=0.0,
            )
            r.score += 1.0 / (k + rank)
            r.sources.append('vec')
            acc[cid] = r

        for r in acc.values():
            kind = self._kind_of(r.chunk_id)
            r.kind = kind
        return list(acc.values())

    def _kind_of(self, chunk_id: int) -> str:
        try:
            row = self.store.conn.execute(
                'SELECT kind FROM chunks WHERE id = ?', (chunk_id,)).fetchone()
            return row['kind'] if row else 'text'
        except Exception:
            return 'text'


def _title_affinity(query: str, title: str) -> float:
    """标题与查询的贴合度。

    解决的核心问题：问「防风草卖多少钱」时，
    《防风草种子》《防风草汤》因含"卖价"字段而与《防风草》同分甚至更高。
    规则：
    - 标题本身就是查询中的实体 → 强加权
    - 查询实体只是标题的前缀（防风草 ⊂ 防风草种子）→ 轻微降权
    """
    if not title or not query:
        return 1.0
    t = title.strip()
    q = query.strip()

    # 标题完整出现在查询里：最高优先
    if t and t in q:
        # 标题越长，说明是更具体的实体，加权越高
        return 1.35 if len(t) >= 3 else 1.20

    # 查询里的实体是标题前缀（说明是衍生物品：种子/汤等）
    core = _core_entity(q)
    if core and t.startswith(core) and len(t) > len(core):
        return 0.82

    # 标题包含查询核心实体
    if core and core in t:
        return 1.10
    return 1.0


ENTITY_SPLIT = re.compile(r'(?:和|与|跟|还有|以及|、|,|，|\bvs\.?\b|\bVS\b)')

QUERY_STOP_WORDS = (
    '卖多少钱', '能卖多少', '值多少钱', '值多少', '卖多少', '多少钱',
    '要几天才能成熟', '要几天成熟', '几天才能', '几天成熟', '要几天',
    '怎么获得', '怎样得到', '有什么用', '有什么用吗', '有什么用途',
    '哪个卖得更贵', '哪个更贵', '哪个更好', '哪个更赚', '哪个更值',
    '哪个卖得贵', '哪个', '更贵', '更赚', '更值', '更好', '比较',
    '是什么', '怎么样', '能卖', '能', '要', '的',
)


def extract_entities(query: str) -> list[str]:
    """从问题中提取实体列表（支持对比题的多实体）。

    「钻石和防风草哪个卖得更贵」→ ['钻石', '防风草']
    「防风草卖多少钱」→ ['防风草']
    """
    if not query:
        return []

    # 先按连接词切分
    raw_parts = [p for p in ENTITY_SPLIT.split(query) if p and p.strip()]

    out: list[str] = []
    for part in raw_parts:
        s = part.strip()
        # 去掉疑问/动词后缀
        for stop in QUERY_STOP_WORDS:
            s = s.replace(stop, '')
        # 去掉开头残留的疑问词
        s = re.sub(r'^[的是在里]', '', s).strip()
        s = s.strip('？?。，,！!　 ')
        if s and len(s) >= 2:
            out.append(s)

    # 去重保序
    return list(dict.fromkeys(out))


def _core_entity(query: str) -> str:
    """取第一个实体（兼容旧调用）。"""
    ents = extract_entities(query)
    if ents:
        return ents[0]
    # 兜底：原逻辑
    q = expand_query(query)
    first = q.split(' ')[0] if q else ''
    for stop in QUERY_STOP_WORDS:
        first = first.replace(stop, '')
    return first.strip()


def build_context(chunks: list[Retrieved], max_chars: int = 3600) -> str:
    """把检索结果拼成给 LLM 的上下文，带可引用的编号。"""
    parts, used = [], 0
    for i, c in enumerate(chunks, start=1):
        body = c.text.strip()
        if used + len(body) > max_chars and parts:
            break
        parts.append(f'[{i}] 来源：《{c.title}》{(" · " + c.section) if c.section else ""}\n{body}')
        used += len(body)
    return '\n\n'.join(parts)
