"""计算工具：品质倍率、加工倍率、利润核算。

为什么必须做成工具，而不是让模型算
----------------------------------
"金星果酒卖多少" = 基础卖价 × 加工倍率 × 品质倍率。
三个数字里任何一个记错，结果就全错，而模型**算错时语气一样自信**。
把算术交给确定性的代码，模型只负责选工具和解读结果。

数据来源
--------
星露谷 wiki 中「工匠物品」「品质」页面的固定规则。
这些规则是游戏机制常量，写死并标注来源比让模型背更可靠。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------- 机制常量

# 品质倍率（wiki: 品质）
QUALITY_MULTIPLIER: dict[str, float] = {
    '普通': 1.0,
    '银星': 1.25,
    '金星': 1.5,
    '铱星': 2.0,
}

# 工匠物品倍率（wiki: 工匠物品）
# 注意：果酒/果汁为纯倍率，腌菜/果酱为「倍率 + 固定加成」
ARTISAN_MULTIPLIER: dict[str, float] = {
    '果酒': 3.0,        # 水果基础价 × 3
    '果汁': 2.25,       # 蔬菜基础价 × 2.25
    '腌菜': 2.0,        # 蔬菜基础价 × 2 + 50
    '果酱': 2.0,        # 水果基础价 × 2 + 50
    '蜂蜜酒': 3.0,
    '啤酒': 3.0,
    '奶酪': 2.0,
    '蛋黄酱': 2.0,
    '布料': 1.0,
}

# 加工后的固定加成（wiki 明确写在该物品的卖价栏里）
ARTISAN_FLAT_BONUS: dict[str, float] = {
    '腌菜': 50,
    '果酱': 50,
}

# 各加工方式的原料类别（用于校验问题是否合理）
ARTISAN_INPUT_KIND: dict[str, str] = {
    '果酒': '水果',
    '果酱': '水果',
    '果汁': '蔬菜',
    '腌菜': '蔬菜',
}

# 不受品质加成的工匠物品（wiki 明确：果酒、果汁、腌菜等固定不吃品质）
QUALITY_EXEMPT = {'果酒', '果汁', '腌菜', '果酱', '蜂蜜酒'}

# 工匠技能加成
PROFESSION_BONUS = {
    '无': 1.0,
    '工匠': 1.4,       # 工匠物品 +40%
}


@dataclass
class CalcResult:
    value: float
    formula: str
    steps: list[str]
    # 结果单位（'金' / '金/天' / ''）。不是所有计算都产出货币：
    # 中位数题的输出是选项数值，硬标「金」会写成「448351 金」。
    unit: str = ''

    def as_dict(self) -> dict[str, Any]:
        return {'value': round(self.value, 2), 'formula': self.formula,
                'steps': self.steps, 'unit': self.unit}


def sell_price(base: float, quality: str = '普通', profession: str = '无',
               artisan: str | None = None) -> CalcResult:
    """计算物品最终卖价。

    base       基础卖价（wiki 上的卖价）
    quality    普通 / 银星 / 金星 / 铱星
    profession 无 / 工匠
    artisan    工匠加工方式（果酒/果汁/…），None 表示不加工
    """
    steps: list[str] = []
    value = float(base)
    steps.append(f'基础卖价：{_fmt(value)}')

    q_mult = QUALITY_MULTIPLIER.get(quality, 1.0)
    p_mult = PROFESSION_BONUS.get(profession, 1.0)

    if artisan:
        a_mult = ARTISAN_MULTIPLIER.get(artisan)
        if a_mult is None:
            raise ValueError(f'未知加工方式：{artisan}，可选：{list(ARTISAN_MULTIPLIER)}')
        value *= a_mult
        steps.append(f'加工为{artisan}：×{a_mult}　→ {_fmt(value)}')

        # 腌菜/果酱有固定加成（wiki: 蔬菜基础价 × 2 + 50）
        bonus = ARTISAN_FLAT_BONUS.get(artisan, 0)
        if bonus:
            value += bonus
            steps.append(f'{artisan}固定加成：+{_fmt(bonus)}　→ {_fmt(value)}')

        if profession != '无' and artisan in {'果酒', '果汁', '腌菜', '果酱', '蜂蜜酒'}:
            value *= p_mult
            steps.append(f'{profession}技能：×{p_mult}　→ {_fmt(value)}')
        if quality != '普通' and artisan in QUALITY_EXEMPT:
            steps.append(f'注意：{artisan}不受{quality}品质加成影响')
        elif quality != '普通':
            value *= q_mult
            steps.append(f'{quality}品质：×{q_mult}　→ {_fmt(value)}')
    else:
        if quality != '普通':
            value *= q_mult
            steps.append(f'{quality}品质：×{q_mult}　→ {_fmt(value)}')
        if profession == '农耕' or profession == '工匠':
            pass  # 生鲜作物不吃工匠加成

    formula = f'{_fmt(base)}'
    if artisan:
        formula += f' × {ARTISAN_MULTIPLIER[artisan]}'
        bonus = ARTISAN_FLAT_BONUS.get(artisan, 0)
        if bonus:
            formula += f' + {_fmt(bonus)}'
    if quality != '普通' and (not artisan or artisan not in QUALITY_EXEMPT):
        formula += f' × {QUALITY_MULTIPLIER[quality]}'

    return CalcResult(value=math.floor(value), formula=formula, steps=steps, unit='金')


def compare_growth(days_a: float, yield_a: float, price_a: float,
                   days_b: float, yield_b: float, price_b: float) -> CalcResult:
    """比较两种作物的日均收益（不计算种子成本，仅毛利）。"""
    gpd_a = (yield_a * price_a) / max(days_a, 1)
    gpd_b = (yield_b * price_b) / max(days_b, 1)
    steps = [
        f'A：{_fmt(yield_a)} 个 × {_fmt(price_a)} 金 ÷ {_fmt(days_a)} 天 = {gpd_a:.2f} 金/天',
        f'B：{_fmt(yield_b)} 个 × {_fmt(price_b)} 金 ÷ {_fmt(days_b)} 天 = {gpd_b:.2f} 金/天',
        f'差值：{abs(gpd_a - gpd_b):.2f} 金/天，'
        f'{"A" if gpd_a > gpd_b else "B"} 更优',
    ]
    return CalcResult(value=max(gpd_a, gpd_b), formula='日均收益 = 产量 × 单价 ÷ 生长天数',
                      steps=steps, unit='金/天')


def regrow_total(base_yield: float, price: float, first_growth: float,
                 regrow: float, season_days: float = 28) -> CalcResult:
    """计算可再生长作物在一个季节内的总收益。"""
    if regrow <= 0:
        total = base_yield * price
        steps = [f'不可再生长：{_fmt(base_yield)} × {_fmt(price)} = {_fmt(total)}']
        return CalcResult(value=math.floor(total), formula='产量 × 单价', steps=steps)

    harvests = 1 + max(0, math.floor((season_days - first_growth) / regrow))
    total = harvests * base_yield * price
    steps = [
        f'首次成熟 {_fmt(first_growth)} 天，之后每 {_fmt(regrow)} 天可再收',
        f'一季 {_fmt(season_days)} 天内可收获 {harvests} 次',
        f'总收益 = {harvests} × {_fmt(base_yield)} × {_fmt(price)} = {_fmt(total)} 金',
    ]
    return CalcResult(value=math.floor(total),
                      formula='收获次数 × 单次产量 × 单价', steps=steps, unit='金')


def median_of_options(values: list[float]) -> CalcResult:
    """三个选项取中间值（沙漠节「你一共走了多少步」那题的固定解法）。

    为什么必须做成工具
    ------------------
    wiki 原文：「三个数字的中间值将会是正确答案（此处指的是三个数字的中值
    而非中间的选项）」。实测模型有两种错法：
      ① 把「中间值」当成平均值：(224175+896702)/2 = 558953.5，答出选项里没有的数
      ② 直接选屏幕上高亮的那个，而不是按数值排序
    同一题问两次，一次对一次错 —— 语气都同样自信。

    正确解法：排序后取下标 len//2 的元素（中位数），不是平均、不是取中间选项。
    """
    nums = sorted(float(v) for v in values)
    n = len(nums)
    if n == 0:
        raise ValueError('数值列表为空')
    mid = nums[n // 2] if n % 2 else (nums[n // 2 - 1] + nums[n // 2]) / 2
    steps = [
        f'原始选项：{", ".join(_fmt(v) for v in values)}',
        f'按大小排序：{", ".join(_fmt(v) for v in nums)}',
        f'取中间那个（第 {n // 2 + 1} 位）：{_fmt(mid)}',
        '注意：是「中位数」，不是平均值，也不是屏幕上排在中间的那个选项',
    ]
    return CalcResult(value=mid, formula=f'{n} 个数的中位数', steps=steps)


def _fmt(v: float) -> str:
    if isinstance(v, float) and v.is_integer():
        return f'{int(v):,}'
    return f'{v:,}' if isinstance(v, int) else f'{v:,.2f}'.rstrip('0').rstrip('.')


# ---------------------------------------------------------------- 工具描述（给 Agent）

TOOL_SCHEMA = [
    {
        'name': 'sell_price',
        'description': '计算物品卖价，支持品质（银星/金星/铱星）、工匠技能、加工方式（果酒/果汁/腌菜）',
        'parameters': {
            'type': 'object',
            'properties': {
                'base': {'type': 'number', 'description': '物品基础卖价（wiki 数值）'},
                'quality': {'type': 'string', 'enum': list(QUALITY_MULTIPLIER),
                            'description': '品质，默认普通'},
                'profession': {'type': 'string', 'enum': list(PROFESSION_BONUS),
                               'description': '技能，默认无'},
                'artisan': {'type': 'string', 'enum': list(ARTISAN_MULTIPLIER),
                            'description': '加工方式，不加工则不填'},
            },
            'required': ['base'],
        },
    },
    {
        'name': 'compare_growth',
        'description': '比较两种作物的日均收益',
        'parameters': {
            'type': 'object',
            'properties': {
                'days_a': {'type': 'number'}, 'yield_a': {'type': 'number'},
                'price_a': {'type': 'number'}, 'days_b': {'type': 'number'},
                'yield_b': {'type': 'number'}, 'price_b': {'type': 'number'},
            },
            'required': ['days_a', 'yield_a', 'price_a',
                         'days_b', 'yield_b', 'price_b'],
        },
    },
    {
        'name': 'regrow_total',
        'description': '计算可再生长作物在一季内的总收益',
        'parameters': {
            'type': 'object',
            'properties': {
                'base_yield': {'type': 'number'}, 'price': {'type': 'number'},
                'first_growth': {'type': 'number'}, 'regrow': {'type': 'number'},
                'season_days': {'type': 'number', 'description': '一季天数，默认 28'},
            },
            'required': ['base_yield', 'price', 'first_growth', 'regrow'],
        },
    },
    {
        'name': 'median_of_options',
        'description': '从多个数值选项里取中位数（按大小排序后的中间值，不是平均值）',
        'parameters': {
            'type': 'object',
            'properties': {
                'values': {'type': 'array', 'items': {'type': 'number'},
                           'description': '题目给出的全部数值选项'},
            },
            'required': ['values'],
        },
    },
]

TOOL_FUNCS = {
    'sell_price': sell_price,
    'compare_growth': compare_growth,
    'regrow_total': regrow_total,
    'median_of_options': median_of_options,
}


def call_tool(name: str, args: dict) -> dict:
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return {'error': f'未知工具 {name}', 'available': list(TOOL_FUNCS)}
    if name == 'regrow_total':
        args.setdefault('season_days', 28)
    try:
        result = fn(**args)
    except TypeError as e:
        return {'error': f'参数不匹配：{e}'}
    except ValueError as e:
        return {'error': str(e)}
    return result.as_dict()
