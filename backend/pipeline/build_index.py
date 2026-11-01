"""构建检索索引：读取抓取结果 → 切块 → 嵌入 → 写入 SQLite。

用法
----
    python -m backend.pipeline.build_index
    python -m backend.pipeline.build_index --limit 500   # 快速试跑
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.app.config import settings  # noqa: E402
from backend.app.embedding import get_embedder  # noqa: E402
from backend.app.store import Store  # noqa: E402
from backend.pipeline.chunker import chunk_text  # noqa: E402

# 数值型字段白名单：用于把 wiki 模板参数转成可计算的结构化数据
NUMERIC_FIELDS = {
    'growth', 'regrow', 'sellprice', 'edibility', 'crafttime', 'energy', 'health',
    'price', 'stack', 'max_stack', 'size', 'damage', 'defense', 'speed',
    '生长时间', '再生长', '卖价', '售价', '能量', '体力', '价格', '伤害', '防御',
}

RE_NUM = re.compile(r'^-?\d+(?:\.\d+)?$')
# 带单位/附加说明的数值，如 "4 天"、"7200 分钟（约 4 天 13 小时）"
RE_NUM_PREFIX = re.compile(r'^(-?\d+(?:\.\d+)?)')


def parse_num(v: Any) -> float | None:
    """从 wiki 字段值中解析数值，容忍单位后缀。

    wiki 的 growth 常写成 "4 天"，不解析会导致作物识别全部失败
    （这是评测暴露的真实 bug）。
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    m = RE_NUM_PREFIX.match(s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def load_pages(raw_dir: Path, limit: int | None = None) -> list[dict]:
    src = raw_dir / 'pages.jsonl'
    if not src.exists():
        raise SystemExit(f'缺少抓取结果: {src}，请先运行 crawl.py')
    docs = []
    with src.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    # 按标题排序（稳定），不再按长度 —— 按长度排序会让物品页被截断
    docs.sort(key=lambda d: d['title'])
    if limit:
        docs = docs[:limit]
    return docs


def normalize_fields(fields: dict) -> dict:
    """保留数值型与短文本字段，转成可直接计算的类型。"""
    out = {}
    for k, v in (fields or {}).items():
        if not v or len(str(v)) > 80:
            continue
        s = str(v).strip()
        if k.lower() in NUMERIC_FIELDS or k in NUMERIC_FIELDS:
            num = parse_num(s)
            if num is not None:
                out[k] = int(num) if float(num).is_integer() else num
                continue
        # 保留少量关键文本字段（中文名、季节等）
        if k.lower() in {'name', 'eng', 'engname', 'source', 'season', 'season1',
                         'type', 'category', 'quality', '名称', '季节', '类型'}:
            out[k] = s[:60]
    return out


def build_item_records(docs: list[dict], icons_meta: list[dict]) -> list[dict]:
    """把物品类文档转成结构化 items 表。"""
    icon_by_key = {}
    for im in icons_meta:
        stem = Path(im['name']).stem
        icon_by_key[stem.lower()] = im['name']

    ITEM_CATS = {'春季作物', '夏季作物', '秋季作物', '冬季作物', '蔬菜', '水果', '花卉',
                 '鱼', '传奇鱼', '可采集矿物', '古物', '动物制品', '可制作物品', '烹饪',
                 '矿物', '矿石', '种子', '原料', '食物', '工具', '武器', '戒指', '鞋子'}

    items = []
    for d in docs:
        cats = set(d.get('categories') or [])
        if not (cats & ITEM_CATS):
            continue
        fields = normalize_fields(d.get('fields') or {})
        eng = str(fields.get('eng') or fields.get('engname') or fields.get('name') or '').strip()
        # 图标文件名匹配：英文名 -> 下划线形式
        icon = None
        for cand in {eng, d['title'], d['title'].replace(' ', '_')}:
            if not cand:
                continue
            key = cand.replace(' ', '_').lower()
            if key in icon_by_key:
                icon = icon_by_key[key]
                break
        items.append({
            'key': eng or d['title'],
            'name_zh': d['title'],
            'icon_file': icon,
            'fields': fields,
            'doc_title': d['title'],
        })
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None, help='只索引前 N 篇（调试用）')
    ap.add_argument('--rebuild', action='store_true', help='清空后重建')
    args = ap.parse_args()

    docs = load_pages(settings.raw_dir, args.limit)
    print(f'[1/4] 读取文档 {len(docs)} 篇')

    icons_file = settings.raw_dir / 'icons.json'
    icons_meta = json.loads(icons_file.read_text(encoding='utf-8')) if icons_file.exists() else []
    print(f'     图标元数据 {len(icons_meta)} 条')

    print('[2/4] 切块')
    all_chunks: list[dict] = []
    for d in docs:
        all_chunks.extend(chunk_text(d['title'], d['text'], d.get('fields')))
    print(f'     共 {len(all_chunks)} 块，平均 {sum(c["token_len"] for c in all_chunks)//max(len(all_chunks),1)} 字')

    print('[3/4] 嵌入（ONNX 本地模型）')
    emb = get_embedder(settings.model_dir)
    t0 = time.time()
    vectors = emb.encode([c['text'] for c in all_chunks], batch_size=32)
    print(f'     {vectors.shape} 用时 {time.time()-t0:.1f}s '
          f'({len(all_chunks)/(time.time()-t0):.0f} 条/秒)')

    print('[4/4] 写入 SQLite')
    store = Store(settings.db_path)
    if args.rebuild:
        store.clear()
    store.insert_docs(docs, all_chunks, vectors)

    items = build_item_records(docs, icons_meta)
    store.insert_items(items)
    with_icon = sum(1 for i in items if i.get('icon_file'))
    print(f'     物品记录 {len(items)} 条（有图标 {with_icon} 条）')

    stats = store.stats()
    (settings.raw_dir / 'index_stats.json').write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding='utf-8')
    print('完成:', json.dumps(stats, ensure_ascii=False))


if __name__ == '__main__':
    main()
