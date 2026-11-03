"""把 infobox 子页面的参数转成可读正文。

问题
----
`畜棚/infobox` 这类子页面是「多建筑共用模板」：多个 {{Infobox building}}
以参数形式嵌在外层模板里，还有 {{{2|...}}} 三层花括号占位。
strip_templates 按 {{ }} 配对处理时，这些占位符会被整体删除，
导致 capacity=8 / size=4x7 这类结构化数据完全丢失。

修复
----
在合并 infobox 之前，先把其中形如 `|key = value` 的参数行
转成「key：value」的普通文本，这样清洗后能保留下来。
"""
from __future__ import annotations

import re

# infobox 常见参数名 → 中文标签
PARAM_LABELS = {
    'name': '名称', 'eng': '英文名', 'base': '前置建筑',
    'build': '建造时间', 'cost': '造价', 'materials': '材料',
    'animals': '可养动物', 'capacity': '容量', 'size': '占地',
    'sizehv': '占地（横向）', 'ability': '功能', 'width': '图标宽度',
    'image': '图片', 'description': '描述', 'source': '来源',
    'season': '季节', 'growth': '生长时间', 'sellprice': '卖价',
    'edibility': '食用价值', 'regrow': '再生长', 'time': '时间',
    'location': '地点', 'weather': '天气', 'difficulty': '难度',
    'behavior': '行为', 'size_min': '最小尺寸', 'size_max': '最大尺寸',
    'xp': '经验值', 'recipe': '配方', 'ingredients': '原料',
    'craftingstation': '制作设备', 'crafttime': '制作时间',
    'type': '类型', 'quality': '品质', 'buff': '增益',
}

# 这些参数是纯展示属性，没有信息量
SKIP = {'width', 'image', 'img', 'icon', 'color', 'class', 'colspan',
        'rowspan', 'align', 'style', 'size_x', 'size_y', 'sizehv'}


def infobox_params_to_text(wikitext: str) -> str:
    """从 infobox wikitext 中抽出有意义的参数，转成正文行。"""
    lines: list[str] = []
    seen: set[str] = set()

    # 逐行匹配 `|key = value`，用 MULTILINE 锚定行首。
    # 早期没用行首锚定，导致匹配到 {{name|White Cow|link=牛}} 里的内层
    # |link= 参数，把 materials/animals 两行的内容截断了。
    pattern = (r'^[ \t]*\|[ \t]*([a-z_][a-z0-9_ ]{1,24})[ \t]*=[ \t]*'
               r'([^\n]*?)[ \t]*$')
    for m in re.finditer(pattern, wikitext, re.I | re.M):
        key = m.group(1).strip().lower()
        val = m.group(2).strip()
        if not val or key in SKIP or key in seen:
            continue
        # 去掉残留的模板占位
        if val.startswith('{{') and val.count('{') != val.count('}'):
            continue
        label = PARAM_LABELS.get(key, key)
        seen.add(key)
        lines.append(f'{label}：{val}')

    if not lines:
        return ''
    return '\n'.join(lines)
