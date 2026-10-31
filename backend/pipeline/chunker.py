"""文档切分：按标题层级 + 长度切块，保留语义边界。

切分策略（针对 MediaWiki 内容特点）
------------------------------------
1. 优先按 `== 标题 ==` 切分 —— wiki 页面天然有良好的层级结构
2. 超长章节按段落贪心合并，接近 max_len 时断开
3. 表格内容单独成块并标记 kind=table —— 数值查询优先命中
4. infobox 字段单独成块并标记 kind=infobox
5. 块之间保留 1 句重叠，减少跨块信息断裂
6. **关键数值注入**：infobox 的数值同时写进正文块，
   因为玩家问"卖多少"时，向量检索需要正文里也出现数字

为什么第 6 条重要
-----------------
实测发现：物品页正文（"防风草是一种蔬菜…"）完全不含卖价数字，
数字只在 infobox 模板里。只把 infobox 单独成块会导致
「防风草卖多少钱」这类问题的语义检索命中不到该页。
"""
from __future__ import annotations

import re

MAX_LEN = 480        # 目标块长度（字符）
MIN_LEN = 80         # 低于此长度尝试合并
OVERLAP = 60         # 重叠字符数

RE_SECTION = re.compile(r'^#{1,6}\s+(.+)$', re.M)

# 需要在正文中显式出现的字段（中文友好名）
FIELD_ALIASES = {
    'sellprice': '卖价', '售价': '卖价', 'price': '卖价',
    'growth': '生长时间', '生长时间': '生长时间',
    'regrow': '再生长', '再生长': '再生长',
    'edibility': '食用价值', 'energy': '能量', 'health': '生命',
    'crafttime': '制作时间', 'season': '季节', '季节': '季节',
    'source': '来源', 'eng': '英文名',
}

# 不进入正文摘要的字段（噪声或非语义）
SKIP_FIELDS = {'color', 'class', 'link', 'period', 'type', 'total', 'image', 'icon',
               'shirt', 'id', 'uid', 'index'}


def numeric_facts(fields: dict | None) -> str:
    """把结构化字段转成一句自然语言事实，便于检索命中。"""
    if not fields:
        return ''
    parts: list[str] = []
    for k, v in fields.items():
        if not v or len(str(v)) > 60:
            continue
        if k.lower() in SKIP_FIELDS or k in SKIP_FIELDS:
            continue
        s = str(v).strip()
        if s.startswith('{{') or '{{' in s:
            continue
        alias = FIELD_ALIASES.get(k.lower()) or FIELD_ALIASES.get(k)
        if alias:
            # 数值补单位
            if alias in {'卖价'} and re.fullmatch(r'\d+(?:\.\d+)?', s):
                s = f'{s} 金'
            elif alias == '生长时间' and re.fullmatch(r'\d+(?:\.\d+)?', s):
                s = f'{s} 天'
            parts.append(f'{alias}：{s}')
    if not parts:
        return ''
    return '　'.join(parts[:12])


def split_sections(text: str) -> list[tuple[str, str]]:
    """按 markdown 标题切分为 [(section_title, body)]。"""
    lines = text.split('\n')
    sections: list[tuple[str, list[str]]] = [('概述', [])]
    for line in lines:
        m = RE_SECTION.match(line.strip())
        if m:
            sections.append((m.group(1).strip(), []))
        else:
            sections[-1][1].append(line)
    return [(t, '\n'.join(body).strip()) for t, body in sections if '\n'.join(body).strip()]


def _split_long(paragraph: str, max_len: int) -> list[str]:
    """单段过长时按句号切。"""
    if len(paragraph) <= max_len:
        return [paragraph]
    sents = re.split(r'(?<=[。！？!?；;\n])', paragraph)
    out, buf = [], ''
    for s in sents:
        if not s.strip():
            continue
        if len(buf) + len(s) > max_len and buf:
            out.append(buf.strip())
            buf = s
        else:
            buf += s
    if buf.strip():
        out.append(buf.strip())
    return out


def _is_noise(line: str) -> bool:
    """判断是否为清洗残留的孤立标点/空壳行。"""
    stripped = re.sub(r'[；。，、\s…·\-—|]', '', line)
    return len(stripped) <= 1


def chunk_text(title: str, text: str, fields: dict | None = None,
               max_len: int = MAX_LEN) -> list[dict]:
    """把一篇文档切成若干块。"""
    chunks: list[dict] = []
    ord_ = 0

    # ---- infobox 结构化字段单独成块（数值查询的关键入口）
    facts = numeric_facts(fields)
    if facts:
        body = f'【{title} · 属性数据】\n{facts}'
        chunks.append({
            'title': title, 'ord': ord_, 'section': '属性数据',
            'text': body, 'kind': 'infobox', 'token_len': len(body),
        })
        ord_ += 1

    # ---- 正文按章节切
    for section, body in split_sections(text):
        lines = [ln for ln in body.split('\n') if ln.strip() and not _is_noise(ln)]
        if not lines:
            continue
        body = '\n'.join(lines)
        is_table_heavy = body.count(' / ') > 6 and len(body) < max_len * 2

        buf = ''
        pieces: list[str] = []
        for p in lines:
            for piece in _split_long(p, max_len):
                if len(buf) + len(piece) + 1 > max_len and buf:
                    pieces.append(buf.strip())
                    buf = (buf[-OVERLAP:] + '\n' + piece) if OVERLAP else piece
                else:
                    buf = (buf + '\n' + piece).strip() if buf else piece
        if buf.strip():
            pieces.append(buf.strip())

        for idx, piece in enumerate(pieces):
            if len(piece) < MIN_LEN and chunks and chunks[-1]['title'] == title:
                # 太短的合并进上一块（但仍保留 infobox 块独立）
                if chunks[-1]['kind'] != 'infobox':
                    chunks[-1]['text'] += '\n' + piece
                    chunks[-1]['token_len'] = len(chunks[-1]['text'])
                    continue

            prefix = f'【{title}】{section}\n' if section != '概述' else f'【{title}】\n'
            # 首个正文块附带关键数值，提升数值类检索命中
            inject = f'{facts}\n' if (facts and idx == 0 and section == '概述') else ''
            body_text = prefix + inject + piece
            chunks.append({
                'title': title, 'ord': ord_, 'section': section,
                'text': body_text,
                'kind': 'table' if is_table_heavy else 'text',
                'token_len': len(body_text),
            })
            ord_ += 1

    return chunks
