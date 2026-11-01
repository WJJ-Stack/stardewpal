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

RE_TEMPLATE = re.compile(r'\{\{[^{}]*\}\}')
RE_TRIPLE_BRACE = re.compile(r'\{\{\{[^{}]*\}\}\}')   # {{{1|...}}} 参数占位
RE_REF = re.compile(r'<ref[^>]*>.*?</ref>|<ref[^>]*/>', re.S)
RE_TAG = re.compile(r'<[^>]+>')
RE_COMMENT = re.compile(r'<!--.*?-->', re.S)
RE_FILE = re.compile(r'\[\[(?:File|文件|Image):[^\]]*\]\]', re.I)
RE_LINK = re.compile(r'\[\[([^\]|]+)(?:\|([^\]]*))?\]\]')
RE_EXT_LINK = re.compile(r'\[(https?://[^\s\]]+)\s*([^\]]*)\]')
RE_HEADING = re.compile(r'^(={2,6})\s*(.+?)\s*\1\s*$', re.M)
RE_TABLE = re.compile(r'\{\|.*?\|\}', re.S)
# 表格内联样式与 rowspan/colspan 等属性
RE_TABLE_ATTR = re.compile(r'(?:style|rowspan|colspan|align|class|width|height)\s*=\s*"[^"]*"', re.I)


def parse_template_params(wikitext: str) -> dict[str, str]:
    """抽取顶层模板的简单键值参数（用于结构化字段）。"""
    out: dict[str, str] = {}
    for m in re.finditer(r'\|\s*([\u4e00-\u9fa5A-Za-z][\u4e00-\u9fa5A-Za-z0-9_ ]{0,24})\s*=\s*([^\n|}]{0,120})',
                         wikitext):
        k, v = m.group(1).strip(), m.group(2).strip()
        if not k or k in out:
            continue
        v = RE_LINK.sub(lambda mm: mm.group(2) or mm.group(1), v)
        v = RE_TAG.sub('', v).strip()
        if v:
            out[k] = v
    return out


def wikitext_to_text(wikitext: str) -> str:
    """把 wikitext 转成适合检索的纯文本，保留标题层级。"""
    t = RE_COMMENT.sub(' ', wikitext)
    t = RE_REF.sub(' ', t)
    t = RE_FILE.sub(' ', t)
    # 三花括号参数占位必须先剥（否则留下 {{{ } 残骸）
    for _ in range(3):
        new = RE_TRIPLE_BRACE.sub(' ', t)
        if new == t:
            break
        t = new
    # 表格保留内容，去掉标记
    t = RE_TABLE.sub(lambda m: _flatten_table(m.group(0)), t)
    # 模板迭代剥离（嵌套模板需要多轮）
    for _ in range(8):
        new = RE_TEMPLATE.sub(' ', t)
        if new == t:
            break
        t = new
    # 残留的花括号与参数名（如 |name = foo 这类未被模板包裹的行）
    t = re.sub(r'\{\{+|\}\}+', ' ', t)
    t = re.sub(r'^\s*\|\s*[\w\u4e00-\u9fa5 ]+\s*=', ' ', t, flags=re.M)
    # 标题转 markdown
    t = RE_HEADING.sub(lambda m: f'\n{"#" * (len(m.group(1)) - 1)} {m.group(2)}\n', t)
    # 内部链接
    t = RE_LINK.sub(lambda m: m.group(2) or m.group(1), t)
    t = RE_EXT_LINK.sub(lambda m: m.group(2) or '', t)
    t = RE_TAG.sub(' ', t)
    t = t.replace("'''", '').replace("''", '')
    t = t.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    t = t.replace('&quot;', '"').replace('&#8776;', '≈')
    # 规整空白
    lines = []
    for line in t.split('\n'):
        line = re.sub(r'[ \t]+', ' ', line).strip()
        line = re.sub(r'^[*#:;]+\s*', '', line)
        if line:
            lines.append(line)
    return '\n'.join(lines)


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
            c = re.sub(r'\s+', ' ', c).strip()
            if c and not c.startswith('{'):
                vals.append(c)
        if vals:
            rows.append(' / '.join(vals))
    return '\n' + '\n'.join(rows) + '\n'


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

    print('[3/4] 清洗 + 抽取结构化字段', flush=True)
    docs = []
    for p in pages:
        text = wikitext_to_text(p['wikitext'])
        if len(text) < 40:  # 跳过空壳页
            continue
        docs.append({
            'title': p['title'],
            'pageid': p['pageid'],
            'touched': p['touched'],
            'categories': p['categories'],
            'text': text,
            'fields': parse_template_params(p['wikitext']),
            'wikitext_len': len(p['wikitext']),
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
