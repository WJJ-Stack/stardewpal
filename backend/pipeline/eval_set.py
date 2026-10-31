"""评测集自动生成：从 wiki 结构化字段反推问题。

为什么能自动生成
----------------
物品页有结构化字段（sellprice/growth/regrow），页面本身给出事实。
于是可以**反向构造问题 + 标准答案**，不需要人工标注。

题型设计（对应不同能力）
----------------------
1. 单事实型     "防风草卖多少钱"              → 测基础检索
2. 品质计算型   "金星防风草卖多少"            → 测工具调用
3. 加工计算型   "防风草做成腌菜卖多少"        → 测链式计算
4. 属性型       "防风草要几天成熟"            → 测数值抽取
5. 跨文档型     "两个物品谁更值钱"            → 测多跳检索
6. 拒答型       "赛博朋克2077的义体价格"       → 测幻觉抑制

评分
----
- 数值题：答案中的数字与标准值精确匹配（或相对误差 <1%）
- 拒答题：回答应包含"未提及/没有/不知道"等拒答标志
- 其余：关键词命中率
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.app.store import Store
from backend.app.tools.calc import QUALITY_MULTIPLIER, sell_price

QUALITY_CASES = ['银星', '金星', '铱星']
ARTISAN_CASES = ['果酒', '果汁', '腌菜']

REFUSAL_QUESTIONS = [
    '赛博朋克2077 里最强义体多少钱',
    '原神里胡桃的元素爆发倍率是多少',
    '星露谷里怎么打败最终 BOSS 加农',
    '明日方舟能天使的攻击力是多少',
    '我的世界钻石镐耐久度是多少',
]


@dataclass
class EvalCase:
    id: str
    question: str
    expected: str
    kind: str            # fact | quality | artisan | attr | compare | refusal
    numeric: float | None = None
    source: str = ''

    def as_dict(self) -> dict[str, Any]:
        return {'id': self.id, 'question': self.question, 'expected': self.expected,
                'kind': self.kind, 'numeric': self.numeric, 'source': self.source}


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def generate(store: Store, per_type: int = 40, seed: int = 42) -> list[EvalCase]:
    rng = random.Random(seed)
    items = [it for it in store.list_items() if it.get('fields')]
    if not items:
        raise SystemExit('items 表为空，先运行 build_index')

    def price_of(it: dict) -> float | None:
        """取有效卖价，跳过 wiki 用于表示"无售价"的 -1。"""
        for k in ('sellprice', '卖价', 'price', '售价'):
            v = it['fields'].get(k)
            if v is None:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if f > 0:
                return f
        return None

    # 只有作物/农产品/采集物才吃品质加成与加工倍率；
    # 武器、工具、戒指不适用（否则会造出错误的标准答案）
    CROP_CATS = {'春季作物', '夏季作物', '秋季作物', '冬季作物', '蔬菜', '水果', '花卉',
                 '可采集矿物', '古物', '动物制品'}
    NOT_QUALITY = {'武器', '匕首', '剑', '工具', '戒指', '鞋子', '鱼', '传奇鱼',
                   '种子', '原料', '可制作物品'}

    # 只有「可直接加工/可带品质」的作物才适用倍率计算。
    # 树苗、种子、非食材必须排除，否则会生成错误标准答案
    # （真实模型评测中模型的反驳是对的，说明词表判断不可靠）。
    NON_ARTISAN = ('树苗', '种子', '苗', '斧', '锤', '剑', '刀', '弓', '杖',
                   '戒指', '鞋', '工具', '镐', '锄', '帽', '衣', '裤')
    # wiki 明确排除在果汁/果酒原料外的作物
    JUICE_EXCLUDED = ('小麦', '啤酒花', '茶叶')
    # wiki 明确说明「既非水果也非蔬菜，不能放入工匠设备」的例外
    ARTISAN_FORBIDDEN = ('宝石甜莓', '齐瓜', '茶树叶', '咖啡豆', '仙人掌果子')

    def is_crop_like(it: dict) -> bool:
        """可作为品质/加工计算对象：有生长周期、非工具/种子/武器。"""
        name = str(it.get('name_zh') or '')
        title = str(it.get('doc_title') or '')
        if any(x in name or x in title for x in NON_ARTISAN):
            return False
        t = str(it['fields'].get('type') or it['fields'].get('类型') or '')
        if t in NOT_QUALITY:
            return False
        return (it['fields'].get('growth') or it['fields'].get('生长时间')) is not None

    def artisan_kind_ok(it: dict, artisan: str) -> bool:
        """加工方式与原料类型是否匹配（严格模式）。

        只认 wiki 分类，分类缺失或不明确就跳过该题。
        理由：靠名称猜会生成错误标准答案——实测中模型的反驳
        比我的词表更准确（如「苋菜不能做果汁」「茶叶不能腌」）。
        宁可少几道题，也不要错的标准答案。
        """
        name = str(it.get('name_zh') or '')
        if name in ARTISAN_FORBIDDEN:
            return False
        if artisan in ('果酒', '果汁') and any(w in name for w in JUICE_EXCLUDED):
            return False

        cats = set(it.get('categories') or [])
        if artisan in ('果酒', '果汁'):
            return '水果' in cats
        if artisan in ('腌菜', '果酱'):
            return '蔬菜' in cats
        return False

    priced = [it for it in items if price_of(it) is not None]
    crop_like = [it for it in priced if is_crop_like(it)]
    growable = [it for it in items
                if (it['fields'].get('growth') or it['fields'].get('生长时间')) is not None]
    rng.shuffle(priced)
    rng.shuffle(crop_like)
    rng.shuffle(growable)

    cases: list[EvalCase] = []
    n = 0

    # ---- 1. 单事实：卖价
    for it in priced[:per_type]:
        base = price_of(it)
        name = it.get('name_zh') or it.get('key')
        n += 1
        cases.append(EvalCase(
            id=f'fact-{n:03d}', question=f'{name}卖多少钱？',
            expected=f'{int(base)} 金', kind='fact', numeric=base,
            source=it.get('doc_title') or name))

    # ---- 2. 品质计算（仅限作物类）
    for it in crop_like[:per_type]:
        base = price_of(it)
        name = it.get('name_zh') or it.get('key')
        q = rng.choice(QUALITY_CASES)
        val = sell_price(base, q).value
        n += 1
        cases.append(EvalCase(
            id=f'qual-{n:03d}', question=f'{q}{name}卖多少钱？',
            expected=f'{int(val)} 金', kind='quality', numeric=float(val),
            source=it.get('doc_title') or name))

    # ---- 3. 加工计算（仅限原料类型匹配的作物）
    artisan_cases = 0
    for it in crop_like[per_type:]:
        if artisan_cases >= per_type:
            break
        name = it.get('name_zh') or it.get('key')
        base = price_of(it)
        if base is None:
            continue
        # 逐个尝试匹配的加工方式
        matched = [a for a in ARTISAN_CASES if artisan_kind_ok(it, a)]
        if not matched:
            continue
        a = rng.choice(matched)
        try:
            val = sell_price(base, '普通', artisan=a).value
        except Exception:
            continue
        n += 1
        artisan_cases += 1
        cases.append(EvalCase(
            id=f'art-{n:03d}', question=f'{name}做成{a}能卖多少钱？',
            expected=f'{int(val)} 金', kind='artisan', numeric=float(val),
            source=it.get('doc_title') or name))

    # ---- 4. 属性：生长周期
    for it in growable[:per_type]:
        g = it['fields'].get('growth') or it['fields'].get('生长时间')
        try:
            gv = float(str(g).replace('天', '').strip())
        except (TypeError, ValueError):
            continue
        name = it.get('name_zh') or it.get('key')
        n += 1
        cases.append(EvalCase(
            id=f'attr-{n:03d}', question=f'{name}要几天才能成熟？',
            expected=f'{int(gv)} 天', kind='attr', numeric=gv,
            source=it.get('doc_title') or name))

    # ---- 5. 跨文档对比（卖价接近的两件，答案需比较）
    pairs = []
    for _ in range(per_type * 3):
        if len(priced) < 2:
            break
        a, b = rng.sample(priced, 2)
        pa, pb = price_of(a), price_of(b)
        if pa is None or pb is None or abs(pa - pb) < 1:
            continue
        hi = a if pa > pb else b
        pairs.append((a, b, hi))
        if len(pairs) >= per_type:
            break
    for a, b, hi in pairs:
        na = a.get('name_zh') or a.get('key')
        nb = b.get('name_zh') or b.get('key')
        nh = hi.get('name_zh') or hi.get('key')
        n += 1
        cases.append(EvalCase(
            id=f'cmp-{n:03d}', question=f'{na}和{nb}哪个卖得更贵？',
            expected=nh, kind='compare', numeric=None,
            source=f'{a.get("doc_title")}|{b.get("doc_title")}'))

    # ---- 6. 拒答（幻觉抑制）
    for q in REFUSAL_QUESTIONS:
        n += 1
        cases.append(EvalCase(
            id=f'ref-{n:03d}', question=q,
            expected='资料中未提及 / 无法回答（不应编造具体数值）',
            kind='refusal', numeric=None))

    return cases


def save(cases: list[EvalCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for c in cases:
            f.write(json.dumps(c.as_dict(), ensure_ascii=False) + '\n')


def load(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------- 评分

RE_NUM_IN_TEXT = re.compile(r'-?\d[\d,]*\.?\d*')
REFUSAL_MARKERS = ['未提及', '没有记录', '无法回答', '找不到', '不在资料', '无法确定',
                   '没有相关', '不清楚', '不在本知识库', '不属于星露谷', '资料中未']


def score_answer(case: dict[str, Any], answer: str) -> dict[str, Any]:
    kind = case['kind']
    ans = answer or ''

    if kind == 'refusal':
        refused = any(m in ans for m in REFUSAL_MARKERS)
        invented = _has_invented_number(ans)
        ok = refused and not invented
        return {'correct': ok, 'score': 1.0 if ok else 0.0,
                'reason': ('已拒答' if refused else '未拒答') +
                          ('' if not invented else '（且编造了数值）')}

    if case.get('numeric') is not None:
        expected = float(case['numeric'])
        nums = []
        for raw in RE_NUM_IN_TEXT.findall(ans):
            try:
                nums.append(float(raw.replace(',', '')))
            except ValueError:
                continue
        hit = any(_close(n, expected) for n in nums)
        return {'correct': hit, 'score': 1.0 if hit else 0.0,
                'reason': f'期望 {expected:g}，回答中数值 {nums[:6]}'}

    # 文本型：关键词命中
    exp = str(case.get('expected') or '')
    ok = exp in ans or any(tok in ans for tok in _tokens(exp))
    return {'correct': ok, 'score': 1.0 if ok else 0.0,
            'reason': f'期望关键词「{exp}」'}


def _close(got: float, exp: float) -> bool:
    if exp == 0:
        return abs(got) < 1e-6
    return abs(got - exp) / max(abs(exp), 1e-9) < 0.01


def _has_invented_number(text: str) -> bool:
    """粗判：拒答时是否编造了具体数值。

    要点：引用编号（[1]、[1–17]、[1][2]）不是编造，
    必须先剔除再检查。这是评测中发现的评分 bug——
    模型正确拒答并标注引用，却被误判为"编造数值"。
    """
    if not any(m in text for m in REFUSAL_MARKERS):
        return False
    # 剔除引用角标：[1] [1,2] [1–17] [1][2]
    cleaned = re.sub(r'\[\s*\d+(?:\s*[,，\-–~]\s*\d+)*\s*\]', ' ', text)
    # 剔除常见倍率/百分比表达（不是"编造的价格"）
    cleaned = re.sub(r'\d+(?:\.\d+)?\s*[%％倍]', ' ', cleaned)
    # 剩下的才是真正的事实性数字
    nums = [n for n in RE_NUM_IN_TEXT.findall(cleaned)
            if len(n.replace(',', '').replace('.', '')) >= 2]
    return len(nums) > 0


def _tokens(s: str) -> list[str]:
    return [t for t in re.split(r'[\s/、，,（）()]+', s) if len(t) >= 2]
