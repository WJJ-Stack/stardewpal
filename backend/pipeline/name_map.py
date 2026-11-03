"""英文物品名 → 中文名映射，从知识库自身构建。

为什么不用手工词表
------------------
wiki 的 {{name|Wood|1350}} 用的是英文键，但每个物品页面都有
`eng` 字段（如 防风草 页的 fields.eng = "Parsnip"）。
直接拿这个字段建映射，比手写词表可靠，也不会漏。

加载时机：在 crawl 清洗之前，从已有 pages.jsonl 建映射；
若没有则返回空表（退化为保留英文）。
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / 'data' / 'raw'

# 高频但可能没有独立页面的词（资源类）
_EXTRA = {
    'wood': '木材', 'stone': '石头', 'fiber': '纤维', 'clay': '黏土',
    'coal': '煤', 'copper ore': '铜矿石', 'iron ore': '铁矿石',
    'gold ore': '金矿石', 'iridium ore': '铱矿石',
    'copper bar': '铜锭', 'iron bar': '铁锭', 'gold bar': '金锭',
    'iridium bar': '铱锭', 'refined quartz': '精炼石英',
    'hardwood': '硬木', 'sap': '树液', 'hay': '干草',
    'energy': '能量', 'health': '生命', 'speed': '速度',
    'defense': '防御', 'weight': '重量', 'immunity': '免疫',
    'attack': '攻击', 'luck': '运气', 'magnetic': '磁性',
}


@lru_cache(maxsize=1)
def get_name_map() -> dict[str, str]:
    """构建 {英文名小写: 中文名} 映射。"""
    mapping: dict[str, str] = dict(_EXTRA)

    src = RAW / 'pages.jsonl'
    if not src.exists():
        return mapping

    try:
        with src.open(encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                fields = d.get('fields') or {}
                eng = str(fields.get('eng') or fields.get('engname') or '').strip()
                zh = str(fields.get('name') or d.get('title') or '').strip()
                if eng and zh:
                    key = eng.lower()
                    # 只保留干净的英文名（排除模板残留）
                    if re.fullmatch(r"[a-z0-9 '\-\.&()]{1,40}", key):
                        mapping.setdefault(key, zh)
    except Exception:  # noqa: BLE001
        pass

    return mapping


def translate_item(name: str) -> str:
    """把英文物品名转成中文；查不到时原样返回。"""
    if not name:
        return name
    m = get_name_map()
    key = name.strip().lower()
    if key in m:
        return m[key]
    # 去掉复数 s 再试
    if key.endswith('s') and key[:-1] in m:
        return m[key[:-1]]
    return name
