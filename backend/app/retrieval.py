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

        # 稀疏检索跑两路查询，再融合。为什么必须两路：
        #  _fts_query 把整句切成双字前缀并用 OR 连接，于是有两种相反的失效——
        #
        #  ① 要「原句精确匹配」：「你一共走了多少步」整句进 FTS，
        #     命中的双字片段最多，《沙漠节》稳居第 1（覆盖率 1.00）。
        #     一旦压成实体（只剩「你一/共走/多少」）就掉到第 3，
        #     被《金色核桃》这种只擦到两个片段的短页反超。
        #
        #  ② 要「实体精查」：「沙漠节什么时候办」整句进 FTS，
        #     "什么"* OR "么时"* OR "时候"* 这些虚词片段在语料里到处都是，
        #     第 1 名变成《版本历史/1.6》，《沙漠节》连前 10 都进不去。
        #     压成「沙漠节」后立刻回到第 1。
        #
        # 两路各自都能赢一半，所以都跑、按 RRF 融合，谁也不单独决定排序。
        sparse_full = self.store.search_bm25(q, self.top_k_sparse)
        sparse_ent = self.store.search_bm25(_sparse_query(query), self.top_k_sparse)
        sparse = _merge_ranked([(sparse_full, 1.0), (sparse_ent, 0.95)])

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
            # 查询覆盖率：压制「只命中一个通用词」的干扰页
            r.score *= _coverage_factor(query, r.text, r.title)

        fused.sort(key=lambda r: -r.score)
        return fused[:top_k]

    # ---------------------------------------------- 命中块 → 上下文窗口
    def expand(self, hits: list[Retrieved], radius: int = 1,
               max_chars: int = 1900, lead_first: bool = True,
               query: str = '') -> list[Retrieved]:
        """把命中的窄块还原成更大的上下文窗口（small-to-big 检索）。

        为什么必须做
        ------------
        切块按 480 字窗口，答案和提问经常被切开。实测《沙漠节》：
          ord 0  「沙漠节是每年春季 15日、16日 和 17 日…」   ← 日期在这
          ord 18 「…三个数字的中间值将会是正确答案」         ← 答题规则这
          ord 19 「你一共走了多少步？」                     ← BM25 命中的是这
        只把 ord19 交给模型，它拿到一道没有答案的题，于是编出「每年秋季15日」。

        做法：以命中块为锚点拼接同文档相邻块，再补上该文档首段
        （wiki 首段就是条目的定义与概要，问日期/归属/是什么都靠它）。
        同一文档只展开一次（保留分数最高的那个锚点），
        否则《信件》这类长页会占掉两三个引用位。
        """
        if not hits:
            return []

        out: list[Retrieved] = []
        seen_docs: set[int] = set()

        for h in hits:
            row = self.store.conn.execute(
                'SELECT doc_id, ord FROM chunks WHERE id = ?', (h.chunk_id,)).fetchone()
            if not row:
                out.append(h)
                continue
            doc_id, ord_ = row['doc_id'], row['ord']
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)

            # 只在**同一章节内**扩展，不跨章节拼接。
            #
            # 为什么必须限制：切块时一个 480 字窗口会跨章节边界，
            # 《沙漠节》里「村民商铺」的商品表格和下一个章节
            # 「艾米丽的服装服务」被拼进了同一块。
            # 模型于是把商铺的商品（树液/山洞萝卜/手部骨骼）
            # 说成服装服务的产出，又把「三花蛋节日后消失」
            # 套到服装上（说服装也不保留）。
            # 章节内扩展既保住「问题+答案」的完整性，又切断了这种错误跨章引用。
            rows = self.store.conn.execute(
                """SELECT ord, text FROM chunks
                   WHERE doc_id = ? AND ord BETWEEN ? AND ?
                     AND COALESCE(section, '') = COALESCE(?, '')
                   ORDER BY ord""",
                (doc_id, ord_ - radius, ord_ + radius, h.section)).fetchall()
            if not rows:      # 章节信息缺失时的兜底
                rows = self.store.conn.execute(
                    """SELECT ord, text FROM chunks
                       WHERE doc_id = ? AND ord BETWEEN ? AND ? ORDER BY ord""",
                    (doc_id, ord_ - radius, ord_ + radius)).fetchall()
            bodies = [r['text'] for r in rows]

            # 首段（wiki 条目的定义/概要）：按相关性决定要不要带。
            #
            # 为什么要带：问「沙漠节什么时候办」时，日期只在首段里
            # （「每年春季 15日、16日 和 17 日」），不带就答不出。
            #
            # 为什么不能无条件带：问「艾米丽的服装服务」时，
            # 首段讲的是「卡利科三花蛋节日后消失」，
            # 挨着服装段落一起给模型，它就把这句话套到服装上，
            # 答「服装节日后不保留」（wiki 里没这句，服装是普通的衣服道具）。
            # 所以只有首段确实提到查询里的实词时才带。
            if lead_first and rows and rows[0]['ord'] > 0:
                lead = self.store.conn.execute(
                    'SELECT text FROM chunks WHERE doc_id = ? AND ord = 0',
                    (doc_id,)).fetchone()
                if (lead and lead['text'] not in bodies
                        and _lead_is_relevant(query, lead['text'], h.title)):
                    bodies.insert(0, lead['text'])

            text = '\n'.join(bodies)
            if len(text) > max_chars:
                text = _fit_window(bodies, h.text, max_chars)

            out.append(Retrieved(
                chunk_id=h.chunk_id, title=h.title, section=h.section,
                text=text, score=h.score, kind=h.kind,
                sources=[*h.sources, 'expanded'],
            ))
        return out

    def _rrf(self, sparse: list[dict], dense: list[dict]) -> list[Retrieved]:
        acc: dict[int, Retrieved] = {}
        k = self.rrf_k

        # 权重设计（踩过的坑）：
        # 纯 RRF 里「双路命中」的普通结果会盖过「单路第一」的精确结果。
        # 实测：查「你一共走了多少步」时，
        #   - 《沙漠节》BM25 排第 1（含原句），向量没命中 → RRF 只有 1/(60+1)
        #   - 《信件》双路都命中（但内容无关）→ RRF 约它的 2 倍
        # 结果正确答案被挤到第 5 名，模型只能答错。
        # 因此给稀疏检索（BM25，字面命中更可信）更高权重。
        W_SPARSE = 1.0
        W_DENSE = 0.6

        for rank, item in enumerate(sparse, start=1):
            cid = item['chunk_id']
            r = acc.get(cid) or Retrieved(
                chunk_id=cid, title=item['title'], section=item.get('section'),
                text=item['text'], score=0.0,
            )
            r.score += W_SPARSE / (k + rank)
            r.sources.append('bm25')
            acc[cid] = r

        for rank, item in enumerate(dense, start=1):
            cid = item['chunk_id']
            r = acc.get(cid) or Retrieved(
                chunk_id=cid, title=item['title'], section=item.get('section'),
                text=item['text'], score=0.0,
            )
            r.score += W_DENSE / (k + rank)
            r.sources.append('vec')
            acc[cid] = r

        # BM25 第 1 名额外加权。
        # 理由：BM25 是字面匹配，第 1 名通常就是含原句的那一页。
        # 而向量检索容易被泛词带偏（实测查「你一共走了多少步」时，
        # 向量把《信件》《运动鞋》排前面，《沙漠节》根本没进前 20）。
        # 只靠 RRF 相加，会让"双路都命中但无关"的结果盖过"单路第一的精确结果"。
        if sparse:
            top_sparse_id = sparse[0]['chunk_id']
            if top_sparse_id in acc:
                acc[top_sparse_id].score *= 1.8
                acc[top_sparse_id].sources.append('bm25_top1')

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


def _coverage_factor(query: str, text: str, title: str) -> float:
    """查询覆盖率：问题里的**实词**有多少真的出现在这一块里。

    为什么需要这个
    --------------
    `_fts_query` 用 OR 串起所有双字切片，只要命中**一个**词就能进候选。
    实测「石鱼在哪钓」的前 5 名全是《冬季》《大树桩（地点）》这类页面，
    它们只擦到「在哪」两个通用字；「沙漠节是几月几号」的第 1 名
    更是《版本历史/1.6》，真正的《沙漠节》压根没进前 8。

    关键：只统计**实词**（物品名/地名/人名），不统计疑问词。
    否则「多少」「在哪」「几月」会把所有页面拉成同一档，覆盖率失去意义。

    注意是**降权**不是**过滤**——覆盖率低仍可能是对的
    （问「金星蓝莓果酒」时答案在《果酒》页，只含「果酒」「金星」两词），
    所以只压分不删除。
    """
    grams = _content_grams(query)
    if not grams:
        return 1.0
    hay = f'{title}\n{text}'
    hit = sum(1 for g in grams if g in hay)
    cov = hit / len(grams)
    if cov >= 0.6:
        return 1.0
    if cov <= 0.2:
        return 0.30
    return 0.30 + (cov - 0.2) / 0.4 * 0.70


def _merge_ranked(sources: list[tuple[list[dict], float]]) -> list[dict]:
    """把多路 BM25 结果按 RRF 合并（去重，保留最高分的那份原文）。

    为什么不用「分数相加」：两路 BM25 的分数量纲不同（一个查长句、
    一个查实体串），相加会让分数高的那一路直接垄断。RRF 只看名次，天然可比。
    """
    acc: dict[int, float] = {}
    best: dict[int, dict] = {}
    for rows, weight in sources:
        for rank, item in enumerate(rows, start=1):
            cid = item['chunk_id']
            acc[cid] = acc.get(cid, 0.0) + weight / (60 + rank)
            best.setdefault(cid, item)
    out = []
    for cid, score in acc.items():
        item = dict(best[cid])
        item['score'] = score
        out.append(item)
    out.sort(key=lambda x: -x['score'])
    return out


def _sparse_query(query: str) -> str:
    """把自然语言问题压成「实词串」，专供 BM25 使用。

    为什么不直接拿整句去查 FTS
    --------------------------
    `_fts_query` 会把每个中文 token 切成双字前缀，再用 OR 连接。
    整句进入后：
        「沙漠节什么时候办」
        → "沙漠节什么时候办" OR "沙漠"* OR "漠节"* OR "节什"* OR "什么"*
          OR "么时"* OR "时候"* OR "候办"*
    「什么」「么时」「时候」这类词在语料里到处都是，
    OR 一放，BM25 就按这些词打分，第 1 名成了《版本历史/1.6》，
    真正的《沙漠节》连前 10 都进不去。

    压成实体串后：
        「沙漠节」
        → "沙漠节" OR "沙漠"* OR "漠节"*
    命中《沙漠节》第 2 名（第 1 是《家具》，因为它列了所有沙漠节家具）。
    """
    ents = extract_entities(query)
    if not ents:
        return query
    # 保留长度 >= 2 的实体；单字实体（「蒜」「蛋」）只在没有其他实体时才用
    return ' '.join(ents)


def _content_grams(query: str) -> set[str]:
    entities = extract_entities(query)
    if not entities:
        entities = [query]
    grams: set[str] = set()
    for ent in entities:
        for tok in re.findall(r'[\u4e00-\u9fa5]{2,}', ent):
            if len(tok) == 2:
                grams.add(tok)
            for i in range(len(tok) - 1):
                grams.add(tok[i:i + 2])
    for tok in re.findall(r'[A-Za-z]{3,}', query or ''):
        grams.add(tok.lower())
    return grams


def _lead_is_relevant(query: str, lead_text: str, title: str) -> bool:
    """判断「条目首段」是否值得随命中块一起给模型。

    判据：首段里出现了查询实词（或查询就是问这个条目本身）。
    宁可漏带（模型说"资料里没有"），也不带错（模型会把别的事实套过来）。
    """
    if not query:
        return True
    if title and title in query:
        return True
    grams = _content_grams(query)
    if not grams:
        return True
    # 首段命中一半以上实词才算相关
    hits = sum(1 for g in grams if g in lead_text)
    return hits / len(grams) >= 0.5


def _fit_window(bodies: list[str], anchor: str, max_chars: int) -> str:
    """超预算时以锚点块为中心保留上下文。

    锚点块是真正命中检索词的那一块，优先级最高；
    剩余预算前后各分一半，保证「问题」与「答案」都还在窗口里。
    """

    budget = max(max_chars - len(anchor), 200)
    before: list[str] = []
    after: list[str] = []
    seen = False
    for b in bodies:
        if not seen and b == anchor:
            seen = True
            continue
        (after if seen else before).append(b)

    head = ''
    for b in reversed(before):
        if len(head) + len(b) > budget // 2:
            break
        head = b + '\n' + head
    tail = ''
    for b in after:
        if len(tail) + len(b) > budget // 2:
            break
        tail += b + '\n'
    return (head + anchor + '\n' + tail).strip()


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


ENTITY_SPLIT = re.compile(
    r'(?:和|与|跟|还有|以及|、|,|，|\bvs\.?\b|\bVS\b'
    r'|的|是|谁|在|有|能|会|要|把|被|对|给|让|吗|呢|吧'
    r'|什么时候|什么时|何时|哪天|几号|几月|多久|多长)'
)

# 疑问/动作短语（直接删除，不参与实体）
QUERY_STOP_PHRASES = (
    '卖多少钱', '能卖多少', '值多少钱', '值多少', '卖多少', '多少钱',
    '要几天才能成熟', '要几天成熟', '几天才能', '几天成熟', '要几天', '几天',
    '怎么获得', '怎样得到', '在哪获得', '从哪里获得',
    '有什么用', '有什么用途', '用来做什么', '干什么用',
    '哪个卖得更贵', '哪个更贵', '哪个更好', '哪个更赚', '哪个更值',
    '哪个卖得贵', '哪个', '更贵', '更赚', '更值', '更好', '比较',
    '是什么', '怎么样', '怎么选', '怎么答', '怎么挑', '如何选', '如何答',
    '是谁在经营', '谁经营', '谁在经营', '经营者',
    '需要几天', '多少天', '多少金', '成熟', '生长', '收获', '获得', '得到', '经营', '举办', '刷新', '出现',
    # 位置/时间疑问：「石鱼在哪钓」里的「哪钓」会被当成实词计入覆盖率，
    # 导致正确答案扣分而干扰页不扣分。这些都不是实体。
    '在哪里', '在哪', '哪钓', '哪里钓', '哪买', '哪里买', '哪挖', '哪里挖',
    '什么时候', '什么时', '何时', '哪天', '几号', '几月', '多久', '多长',
    '怎么回事', '是怎样', '怎么办', '能不能', '可不可以',
)

# 单个字也要滤掉的虚词
QUERY_STOP_CHARS = set('的是在里了吗呢吧啊呀和与跟这那个些出题型问才都就还又也很更最')

# 单字实体里属于动词/副词/量词的碎片，不当作实体
SINGLE_CHAR_NOISE = set('成熟可长多少种怎样如何几大小高低新旧前后上下左右中'
                        '成做用买去来给让被把从到对为以于而并或但如若因所')


def extract_entities(query: str) -> list[str]:
    """从问题中提取实体列表（支持对比题的多实体）。

    设计要点（踩过的坑）
    --------------------
    早期实现只按连接词（和/与/跟）切分，再整体删除停止词。
    结果「沙漠节学者出的题怎么选」被当成**一个实体**，
    与任何页面标题都匹配不上，导致相关性门控把所有正确结果判为不相关
    —— 检索明明命中了《沙漠节》，最终却拒答。

    现在改为：**按结构助词也切分**（的/是/谁/在/有/能…），
    再逐个滤短片段，确保「沙漠节」「学者」能被单独取出。
    """
    if not query:
        return []

    # 先按连接词 + 结构助词切分（顺序很关键：
    # 若先删短语再切分，「沙漠节学者出的题怎么选」删掉"怎么选"后
    # 剩下"沙漠节学者出的题"，再切分就只能得到"沙漠节学者出"。
    raw_parts = [p.strip() for p in ENTITY_SPLIT.split(query) if p and p.strip()]

    out: list[str] = []
    for part in raw_parts:
        s = part.strip('？?。，,！!　 ')
        # 去掉整段疑问短语
        for ph in sorted(QUERY_STOP_PHRASES, key=len, reverse=True):
            s = s.replace(ph, '')
        # 再去虚词与标点
        s = ''.join(ch for ch in s if ch not in QUERY_STOP_CHARS).strip()
        # 单字是否保留，取决于它是不是「像物品名」的词：
        # 保留「蒜」「蛋」「鱼」，但滤掉「才」「成」「熟」这类动词/副词碎片。
        if len(s) >= 2:
            out.append(s)
        elif len(s) == 1 and s not in SINGLE_CHAR_NOISE:
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
    for stop in QUERY_STOP_PHRASES:
        first = first.replace(stop, '')
    return first.strip()


def build_context(chunks: list[Retrieved], max_chars: int = 3600) -> str:
    """把检索结果拼成给 LLM 的上下文，带可引用的编号。

    注意：不能因为某一段太长就 `break` 丢掉后面所有片段。
    实测失败案例：查「沙漠节学者问的最后一天是几号」，
    检索命中了《沙漠节》（含答案），但排在前面的《季节》片段很长，
    把预算吃光后直接 break，《沙漠节》整段被丢弃，
    模型只能说「资料中未提及」。

    正确做法：超长的片段做截断并保留，后续片段继续尝试。
    """
    parts: list[str] = []
    used = 0
    for i, c in enumerate(chunks, start=1):
        body = c.text.strip()
        remaining = max_chars - used
        if remaining <= 120:      # 剩余预算太小，后面都放不下
            break
        if len(body) > remaining:
            # 截断保留，而不是整段丢弃
            body = body[: max(remaining - 20, 100)].rstrip() + ' …'
        parts.append(
            f'[{i}] 来源：《{c.title}》{(" · " + c.section) if c.section else ""}\n{body}'
        )
        used += len(body)
    return '\n\n'.join(parts)
