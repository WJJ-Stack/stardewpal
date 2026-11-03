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
)
from backend.app.store import Store
from backend.app.tools.calc import call_tool

# ---------------------------------------------------------------- 提示词

SYSTEM_PROMPT = """你是《星露谷物语》的攻略助手。下面会附上从 wiki 检索到的原文。

**怎么用这些资料，由你自己判断**：
- 相关的就用，不相关的就忽略，不必强行引用
- 资料里有的，用资料里的事实说话，关键数值注明来源编号如 [1]
- 资料里没有的，说"资料里没有"，不要编
- 结合对话历史理解用户的指代（"这个""它""那题"）
- 用户提出猜测时（"这不就是XX吗"），按资料核实后再确认或否定，不要盲目附和
- 资料和你的常识冲突时，以资料为准

**不要把现实世界的历法套到游戏上**：
游戏里只有四季（春/夏/秋/冬），每季 28 天，没有"月份"。
wiki 写"春季15日"就是游戏内的第 15 天，**不存在"3月15日"这种说法**。
用户问"几月几号"时，按资料答"春季15日"，不要换算成现实月份。
同理，不要给游戏内时间补上"周一到周日""上午下午"之外的现实概念。

**不要把一个条目的结论套到另一个条目上**：
资料里"卡利科三花蛋节日后消失"说的是三花蛋，不是服装、不是家具、
也不是任何别的东西。每条事实都有它自己的主语，
换主语之前先确认资料里真的这么写了，没写就不说。
同理，看到图里有高亮框、光标、"已选中"之类的视觉标记时，
不要据此推断"答案就是这个"——高亮可能只是光标停在那里。

**答完就停**：
问题问「选哪个」就给出选项；问「几号」就给出日期。
不要额外解释游戏的设计意图、不要评价这个机制合不合理、
不要补一句「顺便一提…」——这类收尾几乎全是编的。
资料里没写为什么要这样设计，那就不要谈设计。

回答要自然，像在跟玩家聊天，不必照抄资料原文，
也不必套固定格式。资料够就直接答，不够就说不知道。"""


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


    def _search(self, query: str, top_k: int) -> list[Retrieved]:
        return self.retriever.retrieve(query, top_k=top_k)

    def _maybe_calculate(self, question: str, chunks: list[Retrieved],
                         trace: Trace) -> list[dict[str, Any]]:
        """识别数值题并调用计算工具。

        策略：从检索到的结构化字段里取 base，按问题中的品质/加工词计算。
        这样 base 来自 wiki（可信），运算来自代码（确定），模型只做解读。
        """
        if not settings.enable_calculator:
            return []

        # 先看多选题：问题里有 2 个以上大数字，且资料里说要取「中间值」。
        # 这类题不能走卖价计算，必须走中位数工具。
        # 实测模型有两种错法：把「中间值」当平均值（答出选项里没有的 558953.5），
        # 或直接选屏幕上高亮的那个而不是数值排序后的中间那个。
        median = self._maybe_median(question, chunks, trace)
        if median:
            return median

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

    def _maybe_median(self, question: str, chunks: list[Retrieved],
                      trace: Trace) -> list[dict[str, Any]]:
        """多选题：从资料里确认规则，再算数值中位数。

        触发条件必须**同时**满足，缺一不可：
          ① 问题里有 >= 2 个大数字（选项）
          ② 检索到的资料里出现「中间值/中值/中位数」这类措辞

        只看条件 ① 会误伤「防风草 35 金 和 蓝莓 50 金」这种正常对比；
        只看条件 ② 又会在资料没命中时硬算。
        """
        options = [float(x) for x in re.findall(r'\d{3,}', question)]
        if len(options) < 2:
            return []

        rule = ''
        for c in chunks:
            if _MEDIAN_RULE.search(c.text):
                rule = _MEDIAN_RULE.search(c.text).group(0)
                break
        if not rule:
            trace.add('median_skip', reason='资料里没有「取中间值」的规则')
            return []

        try:
            out = call_tool('median_of_options', {'values': options})
            if out.get('error'):
                trace.add('median_error', error=out['error'])
                return []
            out['item'] = '选项'
            out['input'] = {'values': options}
            out['rule'] = rule
            trace.add('median', values=options, value=out.get('value'), rule=rule)
            return [out]
        except Exception as e:  # noqa: BLE001
            trace.add('median_error', error=str(e)[:150])
            return []

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
                  vision_ctx: dict[str, Any] | None = None,
                  history: list[dict[str, str]] | None = None,
                  image_b64: str | None = None) -> AgentResult:
        """主流程：**先查 wiki，再让模型自己回答**。

        只有两步，没有中间判断层：
          1. 检索：用用户原话（+ 截图）去 wiki 查，拿回原文
          2. 回答：wiki 原文 + 对话历史 + 用户问题一起交给模型，自由作答

        为什么不再让模型自己决定「要不要检索」：
        实测同一个问题问两次，一次查一次不查。查的那次答对了，
        没查的那次说"我需要更多上下文"。既然这是星露谷知识库，
        用户问的东西基本都该查，那就一律查。
        """
        trace = Trace()
        top_k = top_k or settings.top_k_final

        q = question
        if vision_ctx and not image_b64:
            q = _merge_vision_query(question, vision_ctx)
            trace.add('vision_merge', merged_query=q[:120])

        # ---- 第一步：检索（无条件）
        #
        # 截图问题必须**先看图提取检索线索**再检索。
        # 否则检索词只有「这是什么场景」这种泛问，图片信息完全用不上，
        # 实测会命中《山姆》《星露谷展览会》这类无关页面。
        search_q = q
        hints = ''
        if image_b64:
            hints = await self._vision_search_hints(image_b64, q, trace)
            if hints:
                search_q = f'{q} {hints}'
                trace.add('vision_hints', hints=hints[:160])

        # 清洗检索词：去掉数字/题号等噪声，否则 BM25 会被带偏
        search_q = clean_search_query(search_q) or q
        raw_hits = self._search(search_q, top_k)
        # 把窄块还原成上下文窗口：切块会把「问题」和「答案」分到不同块里，
        # 只给 rank1 等于给模型一道没有答案的题（它会自己编，比如「每年秋季15日」）
        chunks = self.retriever.expand(raw_hits, query=search_q)
        trace.add('retrieve', query=search_q[:100], hits=len(chunks),
                  top=[c.title for c in chunks[:3]])

        # 数值题走确定性计算（算术正确性，与语言理解无关）。
        # 截图路径要注意：题目选项在**图里**，不在 q 里，
        # 所以这里要把 VLM 抄出来的 hints 一起交给计算层，
        # 否则「这题该选哪个」这种问法永远触发不了中位数工具
        # （实测截图问答时 calc 为空，全靠模型自己心算，两次一次对一次错）。
        calc_q = f'{q} {hints}' if (image_b64 and hints) else q
        calc_results = self._maybe_calculate(calc_q, chunks, trace)

        # ---- 第二步：组装 messages，交给模型自由回答
        messages: list[dict[str, Any]] = [
            {'role': 'system', 'content': self._build_system(chunks)},
        ]
        for h in (history or [])[-6:]:
            hq = str(h.get('question') or '').strip()
            ha = str(h.get('answer') or '').strip()
            if hq:
                messages.append({'role': 'user', 'content': hq[:600]})
            if ha:
                messages.append({'role': 'assistant', 'content': ha[:1500]})

        user_content: Any = q
        if calc_results:
            lines = ['【计算结果（由工具生成）】']
            for r in calc_results:
                lines.append(f"- {r.get('item', '')}: {r.get('formula', '')} = {r.get('value')}")
            user_content = q + '\n\n' + '\n'.join(lines)

        if image_b64:
            messages.append({'role': 'user', 'content': [
                {'type': 'text', 'text': user_content},
                {'type': 'image_url',
                 'image_url': {'url': f'data:image/png;base64,{image_b64}'}},
            ]})
            trace.add('image_attached', chars=len(image_b64))
        else:
            messages.append({'role': 'user', 'content': user_content})

        answer, mocked = '', False
        try:
            res = await self.llm.chat(messages, temperature=0.2, max_tokens=1200)
            answer = res.content or ''
            mocked = res.mocked
            trace.add('answer', mocked=res.mocked, chars=len(answer),
                      usage=res.usage, with_context=bool(chunks))
        except LLMError as e:
            trace.add('answer_error', error=str(e)[:200])
            answer = ('模型服务暂时不可用。'
                      + (f'以下是检索到的最相关内容：\n\n《{chunks[0].title}》'
                         f'{chunks[0].text[:400]}' if chunks else ''))

        citations = [
            {'index': i, 'title': c.title, 'section': c.section,
             'score': round(c.score, 4), 'kind': c.kind,
             'snippet': c.text[:220]}
            for i, c in enumerate(chunks[:top_k], start=1)
        ]
        return AgentResult(answer=answer, citations=citations, queries=[q],
                           trace=trace.as_list(), calc=calc_results, mocked=mocked,
                           vision=vision_ctx)

    async def _vision_search_hints(self, image_b64: str, question: str,
                                   trace: Trace) -> str:
        """让 VLM 只看图、说出检索线索，不做判断、不给答案。

        返回用于检索的关键词串（画面文字 + 物品/场景名词）。
        设计要点：**只提取，不推断**。如果让它顺手判断"这是什么场景"，
        它会先下一个结论（比如错认成夜市），然后拿这个错误结论去检索，
        形成自我强化，越查越偏。
        """
        # 提示词要点（实测总结）：
        # 用户常把「浏览器窗口 + 聊天记录 + 游戏画面」整屏截图上传。
        # VLM 会优先读网页 UI 的文字（工具栏、网址、按钮），
        # 完全忽略中间那一小块游戏画面。
        # 所以必须明确指示：忽略外围界面，只读游戏画面。
        prompt = (
            '这张图可能是整屏截图：外面是浏览器/软件界面，中间嵌着一张游戏画面。\n'
            '**请忽略所有外围界面文字**（网址、按钮、菜单、"在线""已连接"这类），'
            '只看中间那块像素风游戏内容。\n\n'
            '只做一件事：把**游戏画面里的文字**原样抄下来，'
            '优先抄这几类：\n'
            '- 对话/题目/选项（连同数字一起）\n'
            '- 物品名称、价格、数量\n'
            '- 界面标题、NPC 名字\n\n'
            '如果游戏画面里确实没有文字，就列出你看到的主要物体和界面布局特征。\n'
            '不要判断这是什么场景，不要回答问题，不要推测游戏机制。'
        )
        try:
            res = await self.llm.vision(prompt, image_b64, max_tokens=300)
            text = (res.content or '').strip()
            # 去掉常见的前缀说明，只保留内容
            text = text.replace('\n', ' ')
            return text[:300]
        except LLMError as e:
            trace.add('vision_hints_error', error=str(e)[:150])
            return ''

    def _build_system(self, chunks: list[Retrieved]) -> str:
        """把 wiki 原文拼进 system prompt。"""
        if not chunks:
            return (SYSTEM_PROMPT
                    + '\n\n（本次没有检索到相关资料，请根据你已有的理解回答；'
                      '不确定就直说。）')
        ctx = build_context(chunks, max_chars=3600)
        return f'{SYSTEM_PROMPT}\n\n【星露谷物语 wiki 原文】\n{ctx}'


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


# 多选题「取中间值」的规则措辞（wiki《沙漠节》学者题原文：
# 「三个数字的中间值将会是正确答案（此处指的是三个数字的中值而非中间的选项）」）
_MEDIAN_RULE = re.compile(r'中间值|中值|中位数|取中间')

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


# 明确属于其他作品的实体。命中即判定为跨域问题。
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


def _merge_vision_query(question: str, vision: dict[str, Any]) -> str:
    parts = [question]
    if vision.get('entities'):
        parts.append(' '.join(str(e) for e in vision['entities'][:5]))
    if vision.get('items'):
        parts.append(' '.join(str(i.get('name', '')) for i in vision['items'][:3]))
    return ' '.join(p for p in parts if p).strip()


# 检索词清洗时要去掉的标点（统一换成空格）
_PUNCT_TO_SPACE = str.maketrans({
    c: ' ' for c in
    '，。！？、；：“”‘’（）()[]{}<>《》!?.,;:\'"/\\|~`@#$%^&*+=_-'
})


def clean_search_query(text: str) -> str:
    """清洗检索词：去掉对 BM25 有害的噪声。

    保留：中文短语、物品名、英文名
    去掉：长数字（金额/步数/选项）、题号、标点、空白

    为什么必须去掉数字：
    实测「问题3：你一共走了多少步？224175 896702 448351」检索到《蟹笼》
    （因为那页有很多数字），而「你一共走了多少步」能正确命中《沙漠节》。
    数字在 BM25 里是高权重的"精确匹配"，会把真正含语义的页面挤掉。
    """
    import re as _re

    if not text:
        return ''
    t = text
    # 题号：「问题3」「第3题」「Question 3」
    t = _re.sub(r'(?:问题|第)\s*\d+\s*(?:题)?', ' ', t)
    t = _re.sub(r'\bquestion\s*\d+\b', ' ', t, flags=_re.I)
    # 长数字（3 位以上，含千分位）—— 选项数字、金额、步数
    t = _re.sub(r'\d{1,3}(?:,\d{3})+', ' ', t)
    t = _re.sub(r'\b\d{3,}\b', ' ', t)
    # 标点：用 translate 而不是正则字符类。
    # 这串标点里有 \ [ ' " / 等正则元字符，塞进 [...] 里需要层层转义，
    # 之前漏转义 \[ 触发 SyntaxWarning，也很难读。
    t = t.translate(_PUNCT_TO_SPACE)
    # 压缩空白
    t = _re.sub(r'\s+', ' ', t).strip()
    return t
