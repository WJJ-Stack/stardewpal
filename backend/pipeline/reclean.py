"""离线重跑清洗：从 wikitext.jsonl 生成 pages.jsonl（不联网、不重抓）。

用途：调整清洗逻辑（模板转写、链接处理、infobox 合并）后，
不必再等 1.5 小时抓取，直接重建 pages.jsonl。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.pipeline.crawl import RAW, parse_template_params, wikitext_to_text
from backend.pipeline.infobox import infobox_params_to_text


def main() -> None:
    raw_path = os.path.join(RAW, 'wikitext.jsonl')
    if not os.path.exists(raw_path):
        raise SystemExit(f'缺少 {raw_path}')

    pages = []
    with open(raw_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                pages.append(json.loads(line))
    print(f'[1/3] 读入 {len(pages)} 个页面的原始 wikitext', flush=True)

    # infobox 子页面合并（与 crawl.main 保持一致）
    sub_map: dict[str, list[str]] = {}
    for p in pages:
        t = p['title']
        if '/' in t and t.rsplit('/', 1)[-1].lower() == 'infobox':
            parent = t.rsplit('/', 1)[0]
            readable = infobox_params_to_text(p['wikitext'])
            sub_map.setdefault(parent, []).append((readable or '') + '\n' + p['wikitext'])
    print(f'[2/3] infobox 子页面 {sum(len(v) for v in sub_map.values())} 个 '
          f'→ {len(sub_map)} 个父页面', flush=True)

    docs = []
    for p in pages:
        wt = p['wikitext']
        extra = sub_map.get(p['title'])
        if extra:
            wt = wt + '\n' + '\n'.join(extra)
        if '/' in p['title'] and p['title'].rsplit('/', 1)[-1].lower() == 'infobox':
            continue
        text = wikitext_to_text(wt)
        if len(text) < 40:
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

    out = os.path.join(RAW, 'pages.jsonl')
    with open(out, 'w', encoding='utf-8') as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    total = sum(d['text_len'] for d in docs)
    print(f'[3/3] 写出 {len(docs)} 个有效文档 → {out}', flush=True)
    print(f'  总字数 {total:,}', flush=True)


if __name__ == '__main__':
    main()
