"""截图辅助分析：OCR 抽文字 + 图标向量匹配。

**这里的结果不参与回答**，只用于前端面板展示（"识别到什么"）。
真正决定答案的链路在 `Agent.run()`：
    图片 → VLM 只抄画面文字 → 拼进检索词 → 查 wiki → 模型据原文作答

为什么要说清楚
--------------
早期把这条链路当成主力：OCR 出文字 → 交给 VLM → VLM 转述 → 检索。
每转述一道就丢一层信息，实测把沙漠节认成夜市，然后检索一路跟着错。
后来改成「图片直接进 messages，VLM 只做客观的文字抄写，判断交给检索和回答」，
准确率才上来。

保留下来的价值
--------------
`analyze()` 提供 OCR 文本、实体、价格、图标候选，
用于：① 前端展示识别结果；② 排查「为什么这次检索偏了」。
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from backend.app.config import settings
from backend.app.llm import get_llm
from backend.app.vision.icons import get_icon_index
from backend.app.vision.ocr import get_ocr, summarize


@dataclass
class VisionResult:
    """轻量视觉分析结果（供检索与展示，不参与最终回答的生成）。"""
    description: str = ''
    texts: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    prices: list[int] = field(default_factory=list)
    qualities: list[str] = field(default_factory=list)
    items: list[dict[str, Any]] = field(default_factory=list)
    ocr_confidence: float = 0.0
    vlm_answer: str = ''
    mocked: bool = False
    trace: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            'description': self.description,
            'texts': self.texts[:20],
            'entities': self.entities,
            'prices': self.prices[:10],
            'qualities': self.qualities,
            'items': self.items,
            'ocr_confidence': self.ocr_confidence,
            'vlm_answer': self.vlm_answer,
            'mocked': self.mocked,
        }


class VisionPipeline:
    def __init__(self) -> None:
        self.ocr = get_ocr()
        self.icons = get_icon_index()
        self.llm = get_llm()

    # ---------------------------------------------------------------- 轻量分析

    def analyze(self, image_bytes: bytes) -> VisionResult:
        """轻量分析：OCR 提取文字实体 + 图标候选。

        注意：**结果只用于给检索提供线索**，不参与最终回答。
        v1 曾把 OCR 文字塞进 VLM 提示词，结果模型去描述网页界面文字了。
        """
        out = VisionResult()
        try:
            img = _load(image_bytes)
        except Exception as e:  # noqa: BLE001
            out.description = f'图片解析失败：{e}'
            return out

        img = _fit(img, 1400)
        out.trace.append({'step': 'load', 'size': img.size})

        if settings.ocr_enabled and self.ocr.available:
            try:
                summary = summarize(self.ocr.extract(img))
                out.texts = summary['texts']
                out.entities = summary['entities']
                out.prices = summary['prices']
                out.qualities = summary['qualities']
                out.ocr_confidence = summary['avg_confidence']
                out.trace.append({'step': 'ocr', 'boxes': summary['n_boxes'],
                                  'confidence': summary['avg_confidence']})
            except Exception as e:  # noqa: BLE001
                out.trace.append({'step': 'ocr_error', 'error': str(e)[:120]})
        else:
            out.trace.append({'step': 'ocr', 'skipped': '引擎不可用'})

        try:
            matches = self._match_icons(img)
            if matches:
                out.items = matches
                out.trace.append({'step': 'icons', 'count': len(matches),
                                  'top': [(m['name'], m['score']) for m in matches[:3]]})
        except Exception as e:  # noqa: BLE001
            out.trace.append({'step': 'icons_error', 'error': str(e)[:120]})

        return out

    def _match_icons(self, img: Image.Image) -> list[dict[str, Any]]:
        """图标候选（辅助信号，仅在 OCR 有实体时使用）。"""
        if self.icons.size == 0:
            return []
        out = []
        for m in self.icons.match(img, top_k=2):
            out.append(m.as_dict())
        return out

# ---------------------------------------------------------------- 工具函数

def _load(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes))
    img.load()
    if img.mode not in ('RGB', 'RGBA'):
        return img.convert('RGB')
    return img


def _fit(img: Image.Image, max_side: int) -> Image.Image:
    if max(img.size) <= max_side:
        return img
    r = max_side / max(img.size)
    return img.resize((int(img.width * r), int(img.height * r)), Image.LANCZOS)


def _to_b64(img: Image.Image, max_side: int = 1100) -> str:
    """编码为 base64 PNG（像素图无损）。超大时退化为 JPEG。"""
    img = _fit(img, max_side)
    buf = io.BytesIO()
    rgb = img.convert('RGB')
    rgb.save(buf, format='PNG', optimize=True)
    if buf.tell() > 1200 * 1024:
        buf = io.BytesIO()
        rgb.save(buf, format='JPEG', quality=88, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


_pipeline: VisionPipeline | None = None


def get_vision() -> VisionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = VisionPipeline()
    return _pipeline

