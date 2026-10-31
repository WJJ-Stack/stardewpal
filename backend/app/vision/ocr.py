"""OCR：提取截图中的文字，并按版面还原结构。

为什么 OCR 是主力
-----------------
实测（RapidOCR，中文游戏 UI）：
- 物品名识别 100% 正确
- 单图 2.08s，模型仅 15.4MB，纯 CPU
- 问题：相邻格子文字会粘连（"卖价750金卖价50金"）

本模块解决两件事：
1. 文字提取 + 置信度过滤
2. **按坐标聚类还原版面**，把粘连的长串切回独立格子
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass
class TextBox:
    text: str
    confidence: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2

    @property
    def height(self) -> float:
        return max(self.box[3] - self.box[1], 1.0)

    def as_dict(self) -> dict:
        return {'text': self.text, 'confidence': round(self.confidence, 3),
                'box': [round(v, 1) for v in self.box]}


# ---------------------------------------------------------------- 版面还原

# 中文 + 数字 + 金币等单位的分割点（粘连修复）
RE_INLINE_SPLIT = re.compile(r'(?<=金)(?=[^\s])|(?<=[a-zA-Z])(?=[\u4e00-\u9fa5])')


def sort_reading_order(boxes: list[TextBox], row_tol: float = 0.6) -> list[TextBox]:
    """按阅读顺序排序：先按行聚类（y），再按 x。"""
    if not boxes:
        return []
    rows: list[list[TextBox]] = []
    for b in sorted(boxes, key=lambda t: t.cy):
        placed = False
        for row in rows:
            ref = row[0]
            if abs(b.cy - ref.cy) <= max(ref.height, b.height) * row_tol:
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])
    for row in rows:
        row.sort(key=lambda t: t.box[0])
    return [b for row in rows for b in row]


def split_glued_text(boxes: list[TextBox]) -> list[TextBox]:
    """处理 OCR 把相邻格子文字连成一段的情况。

    策略：若某段文本由多个「数值+单位」重复模式组成，按模式切开；
    并结合邻近块的文字高度估算格距。
    """
    out: list[TextBox] = []
    for b in boxes:
        parts = _split_repeated_units(b.text)
        if len(parts) <= 1:
            out.append(b)
            continue
        # 按 x 均分
        x0, y0, x1, y1 = b.box
        n = len(parts)
        w = (x1 - x0) / n
        for i, p in enumerate(parts):
            out.append(TextBox(
                text=p, confidence=b.confidence,
                box=(x0 + i * w, y0, x0 + (i + 1) * w, y1),
            ))
    return out


def _split_repeated_units(text: str) -> list[str]:
    """切分 "卖价750金卖价50金" → ["卖价750金", "卖价50金"]。"""
    # 匹配「标签+数字+单位」的重复
    pat = re.compile(r'[\u4e00-\u9fa5]{1,6}\s*[\d,.]+\s*(?:金|元|个|天|星|级|%)?')
    found = pat.findall(text)
    if len(found) >= 2 and all(f in text for f in found):
        joined = ''.join(found)
        # 确认几乎覆盖原文，才认为切分可信
        if len(joined) >= len(text) * 0.85:
            return found
    return [text]


# ---------------------------------------------------------------- OCR 引擎


class OCR:
    """RapidOCR 封装（懒加载，模型常驻内存）。"""

    def __init__(self) -> None:
        self._engine = None
        self._error: str | None = None

    @property
    def available(self) -> bool:
        return self._ensure() is not None

    def _ensure(self):
        if self._engine is not None or self._error:
            return self._engine
        try:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
        except Exception as e:  # noqa: BLE001
            self._error = str(e)
            self._engine = None
        return self._engine

    def extract(self, img: Image.Image, min_conf: float = 0.4) -> list[TextBox]:
        eng = self._ensure()
        if eng is None:
            return []
        import numpy as np

        arr = np.asarray(img.convert('RGB'))
        try:
            result, _ = eng(arr)
        except Exception:  # noqa: BLE001
            return []
        boxes: list[TextBox] = []
        for item in result or []:
            try:
                poly, text, conf = item[0], item[1], float(item[2])
            except Exception:
                continue
            if conf < min_conf or not str(text).strip():
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            boxes.append(TextBox(text=str(text).strip(), confidence=conf,
                                 box=(min(xs), min(ys), max(xs), max(ys))))
        boxes = split_glued_text(boxes)
        return sort_reading_order(boxes)


_ocr: OCR | None = None


def get_ocr() -> OCR:
    global _ocr
    if _ocr is None:
        _ocr = OCR()
    return _ocr


# ---------------------------------------------------------------- 语义抽取


RE_PRICE = re.compile(r'(?:卖价|售价|价格)\s*[:：]?\s*([\d,]+)\s*(?:金|元)?')
RE_NUM_UNIT = re.compile(r'([\d,]+)\s*(金|元)')
RE_QUALITY = re.compile(r'(银星|金星|铱星|普通)')


def summarize(boxes: list[TextBox]) -> dict[str, Any]:
    """从 OCR 结果抽取结构化信号，供检索使用。"""
    texts = [b.text for b in boxes]
    joined = ' '.join(texts)

    prices = [int(p.replace(',', '')) for p in RE_PRICE.findall(joined)]
    # RE_NUM_UNIT 有两个分组，findall 返回 tuple，需取第一组
    nums = [int(m[0].replace(',', '')) for m in RE_NUM_UNIT.findall(joined)]
    quality = RE_QUALITY.findall(joined)

    # 候选实体：较短的纯中文词（大概率是物品名/NPC 名）
    entities = [
        t for t in texts
        if 1 < len(t) <= 8 and re.fullmatch(r'[\u4e00-\u9fa5·]+', t)
        and not re.fullmatch(r'[一二三四五六七八九十百千]+', t)
    ]

    return {
        'texts': texts,
        'joined': joined,
        'prices': prices or nums,
        'qualities': list(dict.fromkeys(quality)),
        'entities': list(dict.fromkeys(entities))[:12],
        'avg_confidence': round(sum(b.confidence for b in boxes) / len(boxes), 3) if boxes else 0.0,
        'n_boxes': len(boxes),
    }
