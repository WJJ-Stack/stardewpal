"""Agent 决策层：判断是否检索、结果是否充分、是否需要计算。

为什么需要 Agent，而不是"永远检索 + 直接回答"
--------------------------------------------
固定链路有三个失效场景（都能在评测集上量化）：
1. **不该检索时乱检索**：闲聊、纯常识问题，检索反而引入噪声
2. **该深挖时浅尝辄止**：问题跨多页（"金星果酒 vs 银星果酒差多少"），
   单轮 Top-K 拿不全
3. **该算不算**：把算术交给模型 → 数值幻觉

因此本层做三件事：
- **路由**：need_retrieval 判断（省 token，减少噪声）
- **自省**：sufficient 判断 + 改写重查（多跳问题）
- **工具**：数值题强制走 compute，不让模型心算

每一步都有最大轮数 + 超时，避免死循环。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from backend.app.config import settings
from backend.app.llm import LLMError, get_llm
from backend.app.retrieval import (
    HybridRetriever,
    Retrieved,
    build_context,
    classify,
    extract_entities,
)
from backend.app.store import Store
from backend.app.tools.calc import call_tool

ROUTER_SYSTEM = """你是路由分类器。判断用户问题是否需要查询星露谷物语 wiki 知识库。
必须只输出 JSON，不要任何解释：
{"need_retrieval": true/false, "reason": "简短原因", "query": "改写后的检索词"}

判断规则：
- 需要检索：涉及具体数值（卖价/生长天数/利润）、具体物品/角色/地点、游戏机制细节
- 不需要检索：打招呼、闲聊、让重述上文、纯概念性提问
- query 改写要点：把口语转换为 wiki 术语，补上关键实体名"""

SUFFICIENCY_SYSTEM = """你是检索充分性判断器。给定问题和已检索到的资料，判断资料是否足以回答。
必须只输出 JSON：
{"sufficient": true/false, "reason": "原因", "missing": "缺什么信息", "better_query": "如果不足，给出更好的检索词"}

判断规则：
- sufficient=true：资料中已包含回答问题所需的全部关键事实
- sufficient=false：缺少关键数值、缺少对比对象、或资料完全不相关
- 如果问题需要多个数字做比较，只找到一个也算不足"""

ANSWER_SYSTEM = """你是星露谷物语的攻略助手。基于提供的资料回答问题。

硬性规则：
1. **只用资料中的事实回答**，资料没有的信息明确说"资料中未提及"
2. **所有数值必须标注来源编号**，格式如 [1]
3. **不做心算**。需要计算时，先说明计算依据，由工具给出结果
4. 回答简洁，先说结论，再说细节
5. 如果资料相互矛盾，指出矛盾点

资料格式：[编号] 来源：《页面标题》· 章节
"""


@dataclass
class Trace:
    """记录每一步决策，用于可观测性与评测。"""
    steps: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def add(self, name: str, **kw: Any) -> None:
        self.steps.append({'step': name, 'elapsed_ms': int((time.time() - self.started_at) * 1000),
                           **kw})

    def as_list(self) -> list[dict]:
        return self.steps


@dataclass
class AgentResult:
    answer: str
    citations: list[dict[str, Any]]
    queries: list[str]
    trace: list[dict[str, Any]]
    calc: list[dict[str, Any]] = field(default_factory=list)
    mocked: bool = False
    vision: dict[str, Any] | None = None


class Agent:
    def __init__(self, store: Store) -> None:
        self.store = store
        self.retriever = HybridRetriever(
            store,
            top_k_dense=settings.top_k_dense,
            top_k_sparse=settings.top_k_sparse,
            rrf_k=settings.rrf_k,
        )
        self.llm = get_llm()

    # ---------------------------------------------------------------- 路由
    async def _route(self, question: str, trace: Trace) -> dict[str, Any]:
        # 硬拦截：问题明确指向其他作品时，无论模型怎么判断都强制检索，
        # 由后续相关性门控触发拒答。
        # 原因：模型对《原神》《我的世界》等热门内容有训练记忆，
        # 会绕过知识库直接凭记忆作答，而这类回答无法核实、也超出本项目范围。
        if _mentions_other_domain(question):
            trace.add('route_forced_retrieval', reason='问题涉及本知识库范围外的作品')
            return {'need_retrieval': True, 'reason': '跨域问题强制走检索+门控',
                    'query': question}

        try:
            res = await self.llm.chat(
                [{'role': 'system', 'content': ROUTER_SYSTEM},
                 {'role': 'user', 'content': question}],
                temperature=0.0, max_tokens=200,
            )
            data = _extract_json(res.content)
            if data and 'need_retrieval' in data:
                trace.add('route', need_retrieval=data['need_retrieval'],
                          reason=data.get('reason'), query=data.get('query'))
                return data
        except LLMError as e:
            trace.add('route_error', error=str(e)[:200])

        # 降级：规则路由
        flags = classify(question)
        need = flags['numeric'] or flags['compare'] or len(question) > 6
        trace.add('route_fallback', need_retrieval=need)
        return {'need_retrieval': need, 'reason': '规则降级', 'query': question}

    # ---------------------------------------------------------------- 检索
    def _search(self, query: str, top_k: int) -> list[Retrieved]:
        return self.retriever.retrieve(query, top_k=top_k)

    async def _check_sufficient(self, question: str, chunks: list[Retrieved],
                                trace: Trace) -> dict[str, Any]:
        if not chunks:
            return {'sufficient': False, 'reason': '无检索结果',
                    'better_query': question}
        ctx = build_context(chunks, max_chars=2400)
        try:
            res = await self.llm.chat(
                [{'role': 'system', 'content': SUFFICIENCY_SYSTEM},
                 {'role': 'user', 'content': f'问题：{question}\n\n资料：\n{ctx}'}],
                temperature=0.0, max_tokens=250,
            )
            data = _extract_json(res.content) or {}
            trace.add('sufficiency', sufficient=data.get('sufficient'),
                      reason=data.get('reason'), better_query=data.get('better_query'))
            return data
        except LLMError as e:
            trace.add('sufficiency_error', error=str(e)[:200])
            return {'sufficient': len(chunks) >= 2, 'reason': '降级判断'}

    # ---------------------------------------------------------------- 计算
    def _maybe_calculate(self, question: str, chunks: list[Retrieved],
                         trace: Trace) -> list[dict[str, Any]]:
        """识别数值题并调用计算工具。

        策略：从检索到的结构化字段里取 base，按问题中的品质/加工词计算。
        这样 base 来自 wiki（可信），运算来自代码（确定），模型只做解读。
        """
        if not settings.enable_calculator:
            return []
        flags = classify(question)
        if not (flags['numeric'] or flags['calc']):
            return []

        # 生长周期类问题不需要「卖价计算」，它的答案就在属性字段里。
        # 之前漏了这个判断，导致 attr 题型被错误地算成卖价（评测发现的回归）。
        if _is_growth_question(question):
            trace.add('calc_skip', reason='生长周期问题，直接取字段')
            return []

        results: list[dict[str, Any]] = []

        # 定位问题涉及的物品
        item = self._find_item(question, chunks)
        if not item:
            trace.add('calc_skip', reason='未定位到物品')
            return []

        fields = item.get('fields') or {}
        base = _pick_number(fields, ['sellprice', '卖价', '售价', 'price'])
        if base is None:
            trace.add('calc_skip', reason='物品缺少卖价字段', item=item.get('name_zh'))
            return []

        quality = _detect_quality(question)
        artisan = _detect_artisan(question)

        try:
            out = call_tool('sell_price', {'base': base, 'quality': quality,
                                           'artisan': artisan} if artisan
                            else {'base': base, 'quality': quality})
            out['item'] = item.get('name_zh') or item.get('key')
            out['input'] = {'base': base, 'quality': quality, 'artisan': artisan}
            results.append(out)
            trace.add('calc', tool='sell_price', item=out['item'],
                      value=out.get('value'))
        except Exception as e:  # noqa: BLE001
            trace.add('calc_error', error=str(e)[:150])

        return results

    def _find_item(self, question: str, chunks: list[Retrieved]) -> dict[str, Any] | None:
        """在问题中定位物品。

        关键：必须按名称长度降序匹配，否则「防风草」会抢走「防风草种子」的匹配，
        导致用错 base 价格（评测中发现的真实 bug）。
        """
        items = self.store.list_items()
        # 按名称长度降序，长名优先
        candidates: list[tuple[str, dict[str, Any]]] = []
        for it in items:
            for cand in (it.get('name_zh'), it.get('key')):
                if cand and len(str(cand)) >= 2:
                    candidates.append((str(cand), it))
        candidates.sort(key=lambda x: -len(x[0]))

        for name, it in candidates:
            if name in question:
                return it

        # 兜底：从命中的 chunk 标题反查
        for c in chunks:
            it = self.store.get_item(c.title)
            if it:
                return it
        return None

    # ---------------------------------------------------------------- 主流程
    async def run(self, question: str, *, top_k: int | None = None,
                  vision_ctx: dict[str, Any] | None = None) -> AgentResult:
        trace = Trace()
        top_k = top_k or settings.top_k_final
        queries: list[str] = []
        all_chunks: list[Retrieved] = []
        calc_results: list[dict[str, Any]] = []
        mocked = False

        q = question
        if vision_ctx:
            q = _merge_vision_query(question, vision_ctx)
            trace.add('vision_merge', merged_query=q[:120])

        route = await self._route(q, trace)
        if not route.get('need_retrieval', True):
            answer = await self._answer_without_retrieval(q, trace)
            return AgentResult(answer=answer, citations=[], queries=[], trace=trace.as_list(),
                               mocked=mocked, vision=vision_ctx)

        search_q = route.get('query') or q
        queries.append(search_q)

        step = 0
        while step < settings.max_agent_steps:
            step += 1
            chunks = self._search(search_q, top_k)
            trace.add('retrieve', round=step, query=search_q, hits=len(chunks),
                      top=[c.title for c in chunks[:3]])
            all_chunks = _merge_chunks(all_chunks, chunks)

            if step == 1:
                calc_results = self._maybe_calculate(q, all_chunks, trace)

            check = await self._check_sufficient(q, all_chunks, trace)
            if check.get('sufficient'):
                break
            better = (check.get('better_query') or '').strip()
            if not better or better in queries:
                break
            queries.append(better)
            search_q = better

        answer, mocked = await self._answer(q, all_chunks, calc_results, trace)
        citations = [
            {'index': i, 'title': c.title, 'section': c.section,
             'score': round(c.score, 4), 'kind': c.kind,
             'snippet': c.text[:220]}
            for i, c in enumerate(all_chunks[:top_k], start=1)
        ]
        return AgentResult(answer=answer, citations=citations, queries=queries,
                           trace=trace.as_list(), calc=calc_results, mocked=mocked,
                           vision=vision_ctx)

    async def _answer_without_retrieval(self, q: str, trace: Trace) -> str:
        try:
            res = await self.llm.chat(
                [{'role': 'system', 'content': '你是星露谷物语助手，简洁回答。'},
                 {'role': 'user', 'content': q}],
                max_tokens=500,
            )
            trace.add('answer_no_rag', mocked=res.mocked)
            return res.content
        except LLMError as e:
            trace.add('answer_error', error=str(e)[:200])
            return '模型服务暂时不可用，请稍后重试。'

    async def _answer(self, q: str, chunks: list[Retrieved],
                      calc_results: list[dict[str, Any]],
                      trace: Trace) -> tuple[str, bool]:
        # 相关性门控：检索到的内容与问题无关时，直接拒答而不是硬答。
        # 这是抑制幻觉的关键一环——资料存在≠资料相关。
        relevant = [c for c in chunks if _is_relevant(q, c)]
        if not relevant and not calc_results:
            trace.add('answer_refused', reason='检索结果与问题不相关',
                      considered=len(chunks))
            return ('资料库中没有找到与该问题相关的信息，无法回答。\n\n'
                    '本知识库只覆盖《星露谷物语》的内容，'
                    '如果你问的是其他游戏或领域，我无法提供准确答案。'), False
        if relevant:
            chunks = relevant

        if not chunks and not calc_results:
            trace.add('answer_empty')
            return ('资料库中没有找到与这个问题相关的内容。'
                    '可以换个问法，或告诉我具体的物品/角色名称。'), False

        ctx = build_context(chunks)
        calc_block = ''
        if calc_results:
            lines = ['【工具计算结果（由工具生成，可直接引用）】']
            for r in calc_results:
                lines.append(f"- {r.get('item', '')}: {r.get('formula', '')} = {r.get('value')}")
                for s in r.get('steps', []):
                    lines.append(f'  · {s}')
            calc_block = '\n'.join(lines) + '\n\n'

        user = f'问题：{q}\n\n{calc_block}资料：\n{ctx}'
        try:
            res = await self.llm.chat(
                [{'role': 'system', 'content': ANSWER_SYSTEM},
                 {'role': 'user', 'content': user}],
                temperature=0.2, max_tokens=900,
            )
            trace.add('answer', mocked=res.mocked,
                      usage=res.usage, chars=len(res.content))
            return res.content, res.mocked
        except LLMError as e:
            trace.add('answer_error', error=str(e)[:200])
            # 降级：直接返回检索片段，保证可用性
            top = chunks[0]
            return (f'模型服务不可用，以下是检索到的最相关内容：\n\n'
                    f'《{top.title}》{top.section or ""}\n{top.text[:400]}'), False


# ---------------------------------------------------------------- 辅助函数

def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    m = re.search(r'\{[\s\S]*\}', text)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _merge_chunks(existing: list[Retrieved], new: list[Retrieved]) -> list[Retrieved]:
    seen = {c.chunk_id for c in existing}
    out = list(existing)
    for c in new:
        if c.chunk_id not in seen:
            out.append(c)
            seen.add(c.chunk_id)
    out.sort(key=lambda c: -c.score)
    return out


def _pick_number(fields: dict, keys: list[str]) -> float | None:
    """按优先级取数值字段。

    注意：wiki 用 -1 表示「无固定售价」，必须跳过，
    否则会把 -1 当成卖价参与计算（这是评测里发现的真实 bug）。
    """
    for k in keys:
        for fk, fv in fields.items():
            if fk.lower() != k.lower():
                continue
            try:
                val = float(fv)
            except (TypeError, ValueError):
                continue
            if val < 0:
                continue
            return val
    return None


def _detect_quality(q: str) -> str:
    for k in ('铱星', '金星', '银星'):
        if k in q:
            return k
    return '普通'


GROWTH_PATTERNS = ('几天', '多久', '成熟', '生长', '长成', '天数')


def _is_growth_question(q: str) -> bool:
    """区分「生长周期」与「卖价」类问题。

    两者都含数字疑问词，但需要完全不同的数据。
    """
    return any(p in q for p in GROWTH_PATTERNS)


def _detect_artisan(q: str) -> str | None:
    for k in ('果酒', '果汁', '腌菜', '果酱', '蜂蜜酒'):
        if k in q:
            return k
    return None


def _is_relevant(question: str, chunk: Retrieved) -> bool:
    """判断检索片段是否与问题相关。

    难点：问题常含「跨作品实体」——问「原神胡桃」，库里却有「胡桃木茶几」。
    实体字面命中≠相关，必须区分「被问的主体」与「构词前缀」。

    同时支持对比题：命中问题中任意实体即算相关。
    """
    # ---- 防线 1：明确提及其他作品 → 直接判不相关
    if _mentions_other_domain(question):
        return False

    entities = extract_entities(question)
    t = chunk.title

    # ---- 防线 2：任一实体在标题中占主导地位
    for ent in entities:
        if not ent or ent not in t:
            continue
        # 单字实体（「蒜」「蛋」「鱼」）只在标题完全等于实体时放行，
        # 否则会被「蒜油」「鱼饵」大量误命中；但也必须能命中真正的单字条目
        if len(ent) == 1:
            if t == ent:
                return True
            continue
        # 「胡桃木茶几」里「胡桃」只是构词前缀
        if len(t) > len(ent) + 1 and not _is_entity_suffix_ok(t, ent):
            continue
        return True

    # ---- 防线 3：实体出现在正文中（对比题的另一方常只出现在正文）
    text = chunk.text
    for ent in entities:
        if len(ent) >= 2 and ent in text:
            return True

    # ---- 防线 4：单字实体问题放宽（如「蒜几天成熟」）
    if any(len(e) == 1 for e in entities):
        for ent in entities:
            if len(ent) == 1 and ent in t:
                return True

    # ---- 防线 5：无实体命中 → 只有非常高的分数才放行
    return chunk.score > 0.055


OTHER_DOMAIN = {
    '原神': '原神', '崩坏': '崩坏', '星穹铁道': '星穹铁道', '明日方舟': '明日方舟',
    '我的世界': '我的世界', 'minecraft': 'minecraft', '塞尔达': '塞尔达',
    '泰拉瑞亚': '泰拉瑞亚', '宝可梦': '宝可梦', '英雄联盟': '英雄联盟',
    '王者荣耀': '王者荣耀', '魔兽': '魔兽', '艾尔登': '艾尔登',
    '赛博朋克': '赛博朋克', '加农': '加农', '林克': '林克',
    '胡桃': '原神角色', '能天使': '明日方舟角色', '雷电将军': '原神角色',
    '钟离': '原神角色', '流萤': '星穹铁道角色', '钻石镐': '我的世界物品',
}


def _mentions_other_domain(question: str) -> bool:
    q = question.lower()
    for kw in OTHER_DOMAIN:
        if kw.lower() in q:
            return True
    return False


def _is_entity_suffix_ok(title: str, core: str) -> bool:
    """核心实体在标题中作为独立主体出现。

    「钻石矿」→ 钻石相关，放行
    「胡桃木茶几」→ 木是构词成分，拒绝
    「防风草种子」→ 种子是衍生物，仍与主体强相关，放行
    """
    idx = title.find(core)
    if idx < 0:
        return False
    rest = title[idx + len(core):]
    if rest and rest[0] in '木色形纹制':
        return False
    return True




def _merge_vision_query(question: str, vision: dict[str, Any]) -> str:
    parts = [question]
    if vision.get('entities'):
        parts.append(' '.join(str(e) for e in vision['entities'][:5]))
    if vision.get('items'):
        parts.append(' '.join(str(i.get('name', '')) for i in vision['items'][:3]))
    return ' '.join(p for p in parts if p).strip()
