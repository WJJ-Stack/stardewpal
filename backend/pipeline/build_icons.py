"""图标库构建：下载图标 → 建立特征索引 → 关联物品中文名。

用法
----
    python -m backend.pipeline.build_icons
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.app.config import settings  # noqa: E402
from backend.app.store import Store  # noqa: E402
from backend.app.vision.icons import get_icon_index  # noqa: E402


def build_key_map(store: Store) -> dict[str, str]:
    """图标文件名(stem, 小写) → 中文名。"""
    mapping: dict[str, str] = {}
    for it in store.list_items():
        icon = it.get('icon_file')
        if not icon:
            continue
        stem = Path(icon).stem.lower()
        if it.get('name_zh'):
            mapping[stem] = it['name_zh']
    # 再用英文名/键值兜底
    for it in store.list_items():
        for cand in (it.get('key'), it.get('name_zh')):
            if cand:
                mapping.setdefault(str(cand).replace(' ', '_').lower(),
                                   it.get('name_zh') or it.get('key'))
    return mapping


def main() -> None:
    store = Store(settings.db_path)
    icons_dir = settings.icons_dir
    files = sorted(icons_dir.glob('*.png'))
    print(f'图标文件 {len(files)} 个 -> {icons_dir}')

    key_map = build_key_map(store)
    print(f'中文名映射 {len(key_map)} 条')

    idx = get_icon_index()
    n = idx.load(icons_dir, key_map=key_map)
    print(f'索引构建完成，入索引 {n} 个')

    # 覆盖率诊断
    covered = sum(1 for k in idx.keys if k.lower() in key_map)
    print(f'能映射到中文名: {covered}/{n} ({covered/max(n,1)*100:.0f}%)')

    out = settings.raw_dir / 'icon_index.json'
    out.write_text(json.dumps({
        'count': n,
        'mapped': covered,
        'keys': idx.keys[:50],
    }, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'摘要写入 {out}')

    examples = [(k, key_map.get(k.lower())) for k in idx.keys[:12]]
    print('示例:', examples)


if __name__ == '__main__':
    main()
