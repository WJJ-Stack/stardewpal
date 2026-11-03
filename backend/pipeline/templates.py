"""模板处理：按「是否携带信息」分类，而不是一刀切删除。

实测统计（5259 个页面）
-----------------------
  {{name|...}}        13,822 次  → 物品名，必须保留
  {{price|...}}        6,090 次  → 价格，必须保留
  {{description}}      2,882 次  → 物品描述，必须保留
  {{history}}          3,694 次  → 版本历史，可压缩
  {{npc|...}}          1,535 次  → NPC 名，必须保留
  {{squote}}           1,295 次  → 台词，保留
  {{tprice}}             540 次  → 旅行货车价，保留
  {{qualityprice}}       385 次  → 品质价，保留
  {{xp}}                 322 次  → 经验值，保留
  {{season|spring}}      573 次  → 季节，转中文
  {{edibilitygrid}}      107 次  → 食用价值表，保留

之前一律删除，导致：
  "{{price|43000}}{{name|Wood|1350}}" → "43000 1350"
  → 看不出哪个是金、哪个是木头
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------- 模板分类

# 这些模板的第 N 个参数是「有意义的文本」，按格式转写
TEMPLATE_REWRITERS: dict[str, callable] = {}

SEASON_MAP = {
    'spring': '春季', 'summer': '夏季', 'fall': '秋季', 'autumn': '秋季',
    'winter': '冬季', 'all': '全季节', 'any': '任意季节',
}

# 技能英文键 → 中文。注意 wiki 里 `fish` 和 `fishing` 都在用
# （{{Skill level|fish|10}} / {{Xp|3|fish}}），只映射 fishing 会漏掉前者，
# 清洗出「在 fish 10 级」这种半中半英的句子。
SKILL_MAP = {
    'farming': '耕种', 'mining': '采矿', 'foraging': '采集',
    'fishing': '钓鱼', 'fish': '钓鱼', 'combat': '战斗',
    'luck': '运气', 'combat_level': '战斗',
}

QUALITY_MAP = {
    'normal': '普通', 'silver': '银星', 'gold': '金星', 'iridium': '铱星',
    '0': '普通', '1': '银星', '2': '金星', '4': '铱星',
}


def _split_args(body: str) -> list[str]:
    """按顶层 | 分割参数（忽略嵌套模板和链接里的 |）。"""
    out, buf = [], []
    depth_t = 0   # {{ }} 深度
    depth_l = 0   # [[ ]] 深度
    i = 0
    while i < len(body):
        two = body[i:i + 2]
        if two == '{{':
            depth_t += 1
            buf.append(two)
            i += 2
            continue
        if two == '}}':
            depth_t -= 1
            buf.append(two)
            i += 2
            continue
        if two == '[[':
            depth_l += 1
            buf.append(two)
            i += 2
            continue
        if two == ']]':
            depth_l -= 1
            buf.append(two)
            i += 2
            continue
        if body[i] == '|' and depth_t == 0 and depth_l == 0:
            out.append(''.join(buf))
            buf = []
            i += 1
            continue
        buf.append(body[i])
        i += 1
    out.append(''.join(buf))
    return out


def _clean_inline(s: str) -> str:
    """清理参数里的内联标记（链接、粗体、<br> 等）。"""
    s = re.sub(r'\[\[([^\]|]+)\|([^\]]*)\]\]', r'\2', s)     # [[A|B]] → B
    s = re.sub(r'\[\[([^\]]+)\]\]', r'\1', s)                # [[A]] → A
    s = re.sub(r'<br\s*/?>', ' ', s, flags=re.I)
    s = re.sub(r'<[^>]+>', '', s)
    s = s.replace("'''", '').replace("''", '')
    return s.strip()


def _fmt_name(args: list[str]) -> str:
    """{{name|Item|N}} → "N 个中文名"；{{name|Item}} → "中文名"。

    wiki 里 name 用于材料清单、产物清单，是最高频模板（13,822 次）。
    参数用的是英文键（Wood/Stone），需要翻成中文，否则模型看不懂。
    """
    if not args:
        return ''
    from backend.pipeline.name_map import translate_item

    item = translate_item(_clean_inline(args[0]))
    qty = _clean_inline(args[1]) if len(args) > 1 else ''
    if qty and '=' not in qty:
        return f'{qty} 个{item}' if not qty.endswith(('个', '张', '块')) else f'{qty}{item}'
    return item


def _fmt_price(args: list[str]) -> str:
    """{{price|1200}} → 1200 金；{{price|1350|Wood}} → 1350 木材。

    注意：第二个参数是**单位**，可能是货币（calico/qi）也可能是材料（Wood/Stone）。
    wiki 原文示例：{{price|43000}}、{{price|1350|Wood}} 和{{price|650|Stone}}。
    早期只映射了货币单位，导致材料类显示成英文。
    """
    if not args:
        return ''
    from backend.pipeline.name_map import translate_item

    amount = _clean_inline(args[0])
    if len(args) > 1:
        raw_unit = _clean_inline(args[1])
        unit_lower = raw_unit.lower()
        currency = {'calico': '卡利科三花蛋', 'qi': '齐钻', 'gold': '金', 'g': '金'}
        if unit_lower in currency:
            return f'{amount} {currency[unit_lower]}'
        # 材料单位：翻译成中文
        return f'{amount} {translate_item(raw_unit)}'
    return f'{amount} 金'


def _fmt_tprice(args: list[str]) -> str:
    """{{tprice|200}} → 旅行货车价 200 金。"""
    if not args:
        return ''
    return f'旅行货车价 {_clean_inline(args[0])} 金'


def _fmt_qualityprice(args: list[str]) -> str:
    """{{qualityprice|Item|750}} → Item 各品质价格。"""
    if len(args) < 2:
        return _fmt_price(args)
    item = _clean_inline(args[0])
    base = _clean_inline(args[1])
    try:
        b = float(base.replace(',', ''))
        return (f'{item} 价格：普通 {int(b)} 金 / 银星 {int(b * 1.25)} 金 / '
                f'金星 {int(b * 1.5)} 金 / 铱星 {int(b * 2)} 金')
    except ValueError:
        return f'{item} 价格 {base} 金'


def _fmt_season(args: list[str]) -> str:
    if not args:
        return ''
    return SEASON_MAP.get(_clean_inline(args[0]).lower(), _clean_inline(args[0]))


def _fmt_xp(args: list[str]) -> str:
    """{{xp|3|fish}} → 钓鱼经验 3。

    注意参数顺序是 **等级在前、技能在后**（实测 wiki 原文
    `[[钓鱼竿]]：{{Xp|3|fish}}`），不是 skill 在前。
    早期写反了，输出「3经验 fish」这种半中半英的句子。
    """
    if len(args) < 2:
        return ''
    amount = _clean_inline(args[0])
    raw_skill = _clean_inline(args[1]).lower()
    skill = SKILL_MAP.get(raw_skill, raw_skill)
    return f'{skill}经验 {amount}'


def _fmt_npc(args: list[str]) -> str:
    if not args:
        return ''
    return _clean_inline(args[0])


def _fmt_description(args: list[str]) -> str:
    return _clean_inline(args[0]) if args else ''


def _fmt_squote(args: list[str]) -> str:
    """台词引用，保留原文（在中文引号内）。"""
    if not args:
        return ''
    who = _clean_inline(args[0])
    text = _clean_inline(args[1]) if len(args) > 1 else ''
    if text:
        return f'{who}：「{text}」'
    return f'「{who}」'


def _fmt_duration(args: list[str]) -> str:
    return f'{_clean_inline(args[0])} 分钟' if args else ''


def _fmt_edibility(args: list[str]) -> str:
    return f'食用价值 {_clean_inline(args[0])}' if args else ''


def _fmt_history(args: list[str]) -> str:
    """{{history|1.6|加入游戏}} → 版本 1.6：加入游戏（压缩表达）。"""
    if len(args) < 2:
        return ''
    return f'版本 {_clean_inline(args[0])}：{_clean_inline(args[1])}'


def _fmt_choice(args: list[str]) -> str:
    """{{choice|A|B}} → A / B（选项列表）。"""
    return ' / '.join(_clean_inline(a) for a in args if a.strip())


def _fmt_probability(args: list[str]) -> str:
    if len(args) >= 2:
        return f'{_clean_inline(args[0])}（{_clean_inline(args[1])}）'
    return _clean_inline(args[0]) if args else ''


def _fmt_key(args: list[str]) -> str:
    return f'按键 {_clean_inline(args[0])}' if args else ''


def _fmt_skill_level(args: list[str]) -> str:
    """{{Skill level|fish|10}} → 钓鱼 10 级。

    早期只取第一个参数，结果变成「fish 级」——技能名没翻译，等级数字也丢了。
    这句话在《石鱼》页是「在{{Skill level|fish|10}}且浮漂位于 5 级水深…」，
    清洗后成了「在 fish 级 且浮漂位于 5 级水深…」，
    模型只能照抄这个残缺句子，答出「钓鱼等级为 fish 级」。
    """
    if not args:
        return ''
    skill = SKILL_MAP.get(_clean_inline(args[0]).lower(), _clean_inline(args[0]))
    level = _clean_inline(args[1]) if len(args) > 1 else ''
    return f'{skill} {level} 级' if level else skill


def _fmt_version(args: list[str]) -> str:
    return f'版本 {_clean_inline(args[0])}' if args else ''


# 注册：模板名（小写） → 转写函数
REWRITERS = {
    'name': _fmt_name,
    'price': _fmt_price,
    'tprice': _fmt_tprice,
    'tcprice': _fmt_tprice,
    'qualityprice': _fmt_qualityprice,
    'season': _fmt_season,
    'xp': _fmt_xp,
    'npc': _fmt_npc,
    'description': _fmt_description,
    'squote': _fmt_squote,
    'quote': _fmt_squote,
    'duration': _fmt_duration,
    'edibility': _fmt_edibility,
    'edibilitygrid': _fmt_edibility,
    'history': _fmt_history,
    'choice': _fmt_choice,
    'probability': _fmt_probability,
    'key': _fmt_key,
    'skill level': _fmt_skill_level,
    'version': _fmt_version,
}

# 纯装饰/导航性质，直接删除
DROP_TEMPLATES = {
    'navbox', 'navboxfurniture', 'navboxclothing', 'navboxfestivals',
    'clear', 'clr', 'main article', 'see also', 'about', 'for',
    'infobox', 'infobox item', 'infobox crop', 'infobox building',
    'collapse', 'colspan', 'snippets', 'giftsbyitem', 'crafts', 'recipes',
    'bundlereq', 'bundle', 'quest', 'dialogue', 'eventbyid', 'en',
    '!', '~', '1', 'anchor', 'color', 'date', 'time',
}


def template_to_text(name: str, args: list[str]) -> str:
    """把模板转成可读文本。未注册的模板返回空串（删除）。"""
    key = name.strip().lower()
    fn = REWRITERS.get(key)
    if fn:
        try:
            return fn(args)
        except Exception:  # noqa: BLE001
            return ''
    return ''
