"""星露谷 Wiki 数据管道。

从 BWiki 抓取：
1. 全部 ns0 页面正文（wikitext → 清洗后的纯文本 + 结构化字段）
2. 物品图标（用于图像匹配）
3. 页面元信息（分类、更新时间）

输出统一写到 backend/data/raw/ 下，供索引层消费。

注意事项：
- BWiki 内容遵循 CC BY-NC-SA，本项目仅用于学习/求职作品展示，非商用
- 严格限速 1 req/s，尊重站点资源
"""
from __future__ import annotations

import json
import os
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request

UA = {
    'User-Agent': 'StardewPal/0.1 (educational, non-commercial; contact: local)',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Referer': 'https://wiki.biligame.com/stardewvalley/',
}

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

API = 'https://wiki.biligame.com/stardewvalley/api.php'
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'backend', 'data')
RAW = os.path.join(DATA, 'raw')
ICONS = os.path.join(DATA, 'icons')

RATE_LIMIT = 1.0  # 秒/请求
_last_call = [0.0]


def _throttle():
    gap = time.time() - _last_call[0]
    if gap < RATE_LIMIT:
        time.sleep(RATE_LIMIT - gap)
    _last_call[0] = time.time()


def api(**params) -> dict:
    params.setdefault('format', 'json')
    params.setdefault('formatversion', '2')
    last = None
    for attempt in range(5):
        _throttle()
        try:
            url = API + '?' + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=40, context=CTX) as r:
                return json.loads(r.read().decode('utf-8', 'ignore'))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2.0 * (attempt + 1))
    raise last  # type: ignore[misc]


def fetch_bytes(url: str, timeout: int = 40) -> bytes:
    _throttle()
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return r.read()


# ---------------------------------------------------------------- wikitext 清洗

# 注意：不再用正则剥离模板。BWiki 的模板嵌套很深（如
# {{Price|20|calico}}、{{Name|Calico Egg|class=inline}} 常嵌在表格与正文中间），
# 正则的非贪婪匹配会错位，把整段正文一起吃掉。
# 实测后果：2002 篇文档里 1054 篇（53%）清洗后不足 300 字，
# 《沙漠节》原文几千字被压成 132 字，赛跑/商人/货币等正文全部丢失。
# 改用花括号配对计数解析。
# 注意：ref 的匹配必须分开处理，否则会灾难性误删。
# 错误写法：r'<ref[^>]*>.*?</ref>|<ref[^>]*/>' 配合 re.S
#   → 第一个 `<ref name="x" />` 是自闭合的，但 `.*?` 会一路吃到最后一个 `</ref>`，
#     把中间整篇正文全删掉。实测《沙漠节》因此丢掉 30883 字。
# 正确做法：自闭合标签单独匹配，配对的 <ref>…</ref> 限制不跨行吞掉过多内容。
RE_REF_SELF = re.compile(r'<ref[^>]*/\s*>', re.I)
RE_REF_PAIR = re.compile(r'<ref(?![^>]*/\s*>)[^>]*>.*?</ref>', re.S | re.I)
RE_TAG = re.compile(r'<[^>]+>')
RE_COMMENT = re.compile(r'<!--.*?-->', re.S)
# 文件/图片链接：说明文字里可能再嵌一层内链
#   [[File:Alex's Mum.png|thumb|克莱拉抱着[[亚历克斯]]。]]
# 用 [^\]]* 匹配时会在内层 ]] 处停下，留下 `]]。]]` 这种残缺碎片
# （实测 30 篇文档中招，如《古物》《克莱拉》）。
# 改为匹配到下一个 `]]` 后继续吃到配对，最后统一交给 RE_LINK 清内层。
RE_FILE = re.compile(r'\[\[(?:File|文件|Image):(?:[^\[\]]|\[\[[^\]]*\]\])*\]\]', re.I)
RE_LINK = re.compile(r'\[\[([^\]|]+)(?:\|([^\]]*))?\]\]')
RE_EXT_LINK = re.compile(r'\[(https?://[^\s\]]+)\s*([^\]]*)\]')
RE_HEADING = re.compile(r'^(={2,6})\s*(.+?)\s*\1\s*$', re.M)
RE_TABLE_ATTR = re.compile(r'(?:style|rowspan|colspan|align|class|width|height)\s*=\s*"[^"]*"', re.I)

# 这些模板的「参数值」是有意义的正文，剥离模板时要保留
KEEP_TEMPLATE_ARGS = {
    'Price', 'price', 'Name', 'name', 'NPC', 'npc', 'Item', 'item',
    'Color', 'color', 'Quote', 'quote', 'Emote', 'emote',
    'Key', 'key', 'Mouse', 'mouse',
}


def strip_templates(text: str) -> str:
    """用花括号配对计数剥离模板，保留有意义的参数值。

    与正则剥离的区别
    ----------------
    正则 `\\{\\{[^{}]*\\}\\}` 遇到嵌套模板会错位。
    例如 `{{A|x{{B|y}}z}}`：先剥内层得 `{{A|xz}}`，看似没问题，
    但 BWiki 的表单模板常跨越多行且含大量 `|`，
    正则会把这些内容当成一个模板整体删除，导致正文消失。

    配对计数则严格按 `{{` / `}}` 的嵌套层级处理，不会误吞。
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith('{{', i):
            depth, j = 0, i
            while j < n:
                if text.startswith('{{', j):
                    depth += 1
                    j += 2
                elif text.startswith('}}', j):
                    depth -= 1
                    j += 2
                    if depth == 0:
                        break
                else:
                    j += 1
            body = text[i + 2:j - 2] if depth == 0 else text[i + 2:]
            out.append(_template_to_text(body))
            i = j
        else:
            out.append(text[i])
            i += 1
    return ''.join(out)


def _template_to_text(body: str) -> str:
    """把模板转成可读文本，不再一刀切删除。

    之前所有模板都被剥掉，导致：
      "{{price|43000}}{{name|Wood|1350}}"  →  "43000 1350"
      → 看不出 43000 是金、1350 是木头，数值题必错。

    现在按模板用途转写：
      {{price|43000}}        → 43000 金
      {{name|Wood|1350}}     → 1350 个Wood
      {{season|spring}}      → 春季
      {{qualityprice|X|750}} → X 价格：普通 750 / 银星 937 / 金星 1125 / 铱星 1500
    其余导航/装饰类模板仍然删除。
    """
    from backend.pipeline.templates import _split_args, template_to_text

    parts = _split_args(body)
    if not parts:
        return ' '
    name = parts[0].strip()
    args = [strip_templates(a) for a in parts[1:]]
    text = template_to_text(name, args)
    return f' {text} ' if text else ' '


def parse_template_params(wikitext: str) -> dict[str, str]:
    """抽取顶层模板的简单键值参数（用于结构化字段）。

    值的正则**不能**排除竖线：链接 [[火山地牢#火山口|火山口]] 里就有竖线。
    早期写成 `[^\n|}]{0,120}` 会在竖线处截断，得到残缺的 `[[火山地牢#火山口`，
    于是 infobox 属性块里出现「来源：[[火山地牢#火山口」，链接清洗不掉。

    正确做法：按行取到行尾，再把值里的链接/标签清理掉。
    """
    out: dict[str, str] = {}
    pattern = (r'^[ \t]*\|[ \t]*([\u4e00-\u9fa5A-Za-z][\u4e00-\u9fa5A-Za-z0-9_ ]{0,24})'
               r'[ \t]*=[ \t]*([^\n]{0,200})')
    for m in re.finditer(pattern, wikitext, re.M):
        k, v = m.group(1).strip(), m.group(2).strip()
        if not k or k in out:
            continue
        v = RE_LINK.sub(lambda mm: mm.group(2) or mm.group(1), v)
        v = RE_TAG.sub('', v).strip()
        if v:
            out[k] = v
    return out


# 表格列名会被展平成 `问题 / 答案`、`图片 / 名称 / 价格` 这种样子，
# 若 wiki 里恰好用 `===问题 / 答案===` 作小节标题，就会变成假章节。
# 这类章节名不该成为 chunker 的切分边界 —— 它会把
# 「题目列表」和「答题规则」硬切成两个 section，导致检索到题目却看不到答案。
_TABLE_COL_WORDS = {
    '图片', '名称', '价格', '问题', '答案', '描述', '材料', '奖励', '日期',
    '任务', '地点', '时间', '数量', '物品', '来源', '等级', '效果', '条件',
    '序号', '状态', '备注', '类型', '配方', '售价', '品质', 'n/a',
}


def _is_table_header_heading(text: str) -> bool:
    """判断标题是不是表格列名残留。"""
    t = (text or '').strip()
    if '/' not in t:
        return False
    parts = [p.strip().lower() for p in t.split('/') if p.strip()]
    if not parts:
        return True
    return all(p in _TABLE_COL_WORDS for p in parts)


def wikitext_to_text(wikitext: str) -> str:
    """把 wikitext 转成适合检索的纯文本，保留标题层级与表格内容。"""
    t = RE_COMMENT.sub(' ', wikitext)
    # 先删自闭合引用（这类不需要配对），再删成对的部分
    t = RE_REF_SELF.sub(' ', t)
    t = RE_REF_PAIR.sub(' ', t)
    t = RE_FILE.sub(' ', t)

    # 表格：先按 `{| ... |}` 配对抽出，展平成「a / b / c」行，再回填
    t = _flatten_tables(t)

    # 模板：花括号配对剥离（保留 Price/Name 等有意义的参数值）
    t = strip_templates(t)

    # 残留的花括号与参数名（未被模板包裹的零散行）
    t = re.sub(r'\{\{+|\}\}+', ' ', t)
    t = re.sub(r'^\s*\|\s*[\w\u4e00-\u9fa5 ]+\s*=', ' ', t, flags=re.M)

    # 标题转 markdown。
    # 过滤掉表格展平后残留的假标题：`# / 问题 / 答案`、`# a / b`。
    # 这类行是 `{| ... |}` 表头被展平成 `a / b` 后又恰好以 `=` 开头，
    # 识别成标题会在 chunker 里切出一堆无意义章节。
    t = RE_HEADING.sub(
        lambda m: ('' if _is_table_header_heading(m.group(2)) else
                   f'\n{"#" * (len(m.group(1)) - 1)} {m.group(2)}\n'), t)
    # 链接
    t = RE_LINK.sub(lambda m: m.group(2) or m.group(1), t)
    t = RE_EXT_LINK.sub(lambda m: m.group(2) or '', t)
    t = RE_TAG.sub(' ', t)
    # 兜底：清掉落单的方括号。两类来源都不该进语料：
    #   ① wiki 源文件笔误：`被[闪电]]击中`、`[[旅行货车]处]`
    #   ② 图片说明里嵌外链，剥掉 URL 后剩半边括号
    # 正常链接已在上面两行转成纯文本，剩下的都是残渣。
    t = re.sub(r'\[\[|\]\]|\[|\]', ' ', t)
    t = t.replace("'''", '').replace("''", '')
    t = t.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    t = t.replace('&quot;', '"').replace('&#8776;', '≈')
    # 规整空白
    lines = []
    for line in t.split('\n'):
        line = re.sub(r'[ \t]+', ' ', line).strip()
        # 只剥 wiki 列表符号（* # : ;），**不能**动 markdown 标题的 #。
        # 早期写成 [*#:;]+ ，把 `## 赛跑` 剥成 `赛跑`，
        # 于是 chunker 的 RE_SECTION(^#{1,6}\s+) 一个都匹配不到 ——
        # 《沙漠节》整页 22 个块 section 全是「概述」，
        # 商品表格和「艾米丽的服装服务」被硬拼进同一块，
        # 模型就把别的商铺的商品当成服装服务的产出。
        if not line.startswith('#'):
            line = re.sub(r'^[*#:;]+\s*', '', line)
        if line:
            lines.append(line)
    return '\n'.join(lines)


def _flatten_tables(text: str) -> str:
    """把 `{| ... |}` 表格配对抽出并展平，其余文字原样保留。"""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith('{|', i):
            depth, j = 0, i
            while j < n:
                if text.startswith('{|', j):
                    depth += 1
                    j += 2
                elif text.startswith('|}', j):
                    depth -= 1
                    j += 2
                    if depth == 0:
                        break
                else:
                    j += 1
            out.append(_flatten_table(text[i:j]))
            i = j
        else:
            out.append(text[i])
            i += 1
    return ''.join(out)


def _flatten_table(table: str) -> str:
    rows = []
    for raw in table.split('|-'):
        raw = RE_TABLE_ATTR.sub('', raw)
        cells = [c for c in re.split(r'\|\||\n\||!', raw) if c.strip()]
        vals = []
        for c in cells:
            c = RE_LINK.sub(lambda m: m.group(2) or m.group(1), c)
            c = RE_TAG.sub(' ', c)
            c = re.sub(r'^\s*[|!]\s*', '', c)
            # 清掉行首的 #：`!colspan="3" ...| 内容` 这种表头行，
            # 剥掉 `!` 和属性后可能残留 `# `，展平后就成了
            # 「# / 问题 / 答案」—— 被误当成 markdown 一级标题，
            # chunker 于是切出一个叫「/ 问题 / 答案」的假章节，
            # 把「题目列表」和「答题规则」分到两个 section，
            # 检索命中题目却看不到答案。
            c = re.sub(r'^#+\s*', '', c)
            c = re.sub(r'\s+', ' ', c).strip()
            if c and not c.startswith('{'):
                vals.append(c)
        if vals:
            rows.append(' / '.join(vals))
    # 注意：不要以换行开头。
    # 表格常常紧跟在章节标题后面（===学者=== 的下一行就是 {|），
    # 展平后若以换行开头，标题行就变成了「==学者== <换行> / 问题 / 答案」，
    # RE_HEADING 的 $ 会连后半截一起吞进标题，
    # 于是多出一个叫「/ 问题 / 答案」的假章节。
    # chunker 随即把「题目列表」和「答题规则」切成两个 section，
    # 检索命中了题目却看不到答案（这题因此一度答成「资料里没给答案」）。
    return '\n'.join(rows) + ('\n' if rows else '')


# ---------------------------------------------------------------- 抓取主流程


def list_all_titles() -> list[str]:
    titles, cont = [], {}
    while True:
        d = api(action='query', list='allpages', apnamespace='0',
                aplimit='500', **cont)
        titles.extend(p['title'] for p in d.get('query', {}).get('allpages', []))
        if 'continue' not in d:
            break
        cont = d['continue']
        print(f'    已列出 {len(titles)} 个页面...', flush=True)
    return titles


def fetch_pages(titles: list[str], batch: int = 20) -> list[dict]:
    """批量取 wikitext + 元信息。"""
    out = []
    for i in range(0, len(titles), batch):
        chunk = titles[i:i + batch]
        try:
            d = api(action='query', prop='revisions|categories|info',
                    titles='|'.join(chunk), rvprop='content|timestamp',
                    rvslots='main', cllimit='max')
            for p in d.get('query', {}).get('pages', []):
                if 'missing' in p:
                    continue
                revs = p.get('revisions') or []
                if not revs:
                    continue
                slot = revs[0].get('slots', {}).get('main', {})
                wt = slot.get('content') or revs[0].get('*') or ''
                out.append({
                    'title': p.get('title'),
                    'pageid': p.get('pageid'),
                    'length': p.get('length', 0),
                    'touched': revs[0].get('timestamp'),
                    'categories': [c['title'].replace('分类:', '') for c in p.get('categories', [])],
                    'wikitext': wt,
                })
        except Exception as e:  # noqa: BLE001
            print(f'  batch {i} 失败: {str(e)[:100]}', flush=True)
        if i % 200 == 0:
            print(f'    已抓取 {min(i + batch, len(titles))}/{len(titles)}', flush=True)
    return out


def download_icons(limit: int = 1200) -> list[dict]:
    """下载物品图标（方形小 PNG）。"""
    os.makedirs(ICONS, exist_ok=True)
    cands, cont = [], {}
    while len(cands) < limit * 3:
        d = api(action='query', list='allimages', ailimit='500',
                aiprop='url|size|mime', **cont)
        for im in d.get('query', {}).get('allimages', []):
            w, h = im.get('width') or 0, im.get('height') or 0
            name = im.get('name', '')
            if (im.get('mime') == 'image/png' and 24 <= w <= 128 and w == h
                    and ' ' not in name):
                cands.append(im)
        if 'continue' not in d:
            break
        cont = d['continue']

    meta = []
    for im in cands[:limit]:
        local = os.path.join(ICONS, im['name'])
        if not os.path.exists(local):
            try:
                data = fetch_bytes(im['url'])
                with open(local, 'wb') as f:
                    f.write(data)
            except Exception:  # noqa: BLE001
                continue
        meta.append({
            'name': im['name'],
            'file': im['name'],
            'width': im.get('width'),
            'height': im.get('height'),
            'source_url': im['url'],
        })
        if len(meta) % 100 == 0:
            print(f'    图标 {len(meta)}/{limit}', flush=True)
    return meta


def main():
    os.makedirs(RAW, exist_ok=True)
    print('[1/4] 列出全部页面', flush=True)
    titles = list_all_titles()
    print(f'  共 {len(titles)} 个页面', flush=True)

    print('[2/4] 抓取正文', flush=True)
    pages = fetch_pages(titles)
    print(f'  成功 {len(pages)} 个', flush=True)

    # 保存原始 wikitext：以后调整清洗逻辑时不必重新抓取（省 1.5 小时）
    raw_path = os.path.join(RAW, 'wikitext.jsonl')
    with open(raw_path, 'w', encoding='utf-8') as f:
        for p in pages:
            f.write(json.dumps({
                'title': p['title'], 'pageid': p['pageid'],
                'touched': p['touched'], 'categories': p['categories'],
                'wikitext': p['wikitext'],
            }, ensure_ascii=False) + '\n')
    print(f'  原始 wikitext 已保存：{raw_path}', flush=True)

    print('[3/4] 清洗 + 抽取结构化字段', flush=True)

    # 合并 xxx/infobox 子页面到父页面。
    # 这类子页面是纯模板（容量/占地/造价等结构化字段），
    # 单独清洗会变成空串被丢弃，父页面正文里也缺少这些数据。
    from backend.pipeline.infobox import infobox_params_to_text

    sub_map: dict[str, list[str]] = {}
    for p in pages:
        t = p['title']
        if '/' in t and t.rsplit('/', 1)[-1].lower() == 'infobox':
            parent = t.rsplit('/', 1)[0]
            # 先把参数转成可读正文（否则 {{{2|...}}} 占位符会导致整段被删）
            readable = infobox_params_to_text(p['wikitext'])
            sub_map.setdefault(parent, []).append(
                (readable or '') + '\n' + p['wikitext'])
    if sub_map:
        print(f'  发现 {sum(len(v) for v in sub_map.values())} 个 infobox 子页面，'
              f'涉及 {len(sub_map)} 个父页面', flush=True)

    docs = []
    for p in pages:
        wt = p['wikitext']
        # 若是父页面且存在 infobox 子页面，拼接进来
        extra = sub_map.get(p['title'])
        if extra:
            wt = wt + '\n' + '\n'.join(extra)
        # 纯 infobox 子页面本身跳过（内容已并入父页面）
        if '/' in p['title'] and p['title'].rsplit('/', 1)[-1].lower() == 'infobox':
            continue

        text = wikitext_to_text(wt)
        if len(text) < 40:  # 跳过空壳页
            continue
        docs.append({
            'title': p['title'],
            'pageid': p['pageid'],
            'touched': p['touched'],
            'categories': p['categories'],
            'text': text,
            'fields': parse_template_params(wt),
            'wikitext_len': len(wt),
            'text_len': len(text),
        })
    docs.sort(key=lambda d: d['title'])
    print(f'  有效文档 {len(docs)} 个，平均 {sum(d["text_len"] for d in docs)//max(len(docs),1)} 字',
          flush=True)

    with open(os.path.join(RAW, 'pages.jsonl'), 'w', encoding='utf-8') as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    print('[4/4] 下载图标', flush=True)
    icons = download_icons()
    with open(os.path.join(RAW, 'icons.json'), 'w', encoding='utf-8') as f:
        json.dump(icons, f, ensure_ascii=False, indent=1)
    print(f'  图标 {len(icons)} 个', flush=True)

    stats = {
        'pages_total': len(titles),
        'docs_valid': len(docs),
        'icons': len(icons),
        'avg_text_len': sum(d['text_len'] for d in docs) // max(len(docs), 1),
        'crawled_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RAW, 'crawl_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print('完成:', json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        RATE_LIMIT = 0.2
    main()
