"""截图理解管线：OCR + 图标匹配 + VLM 分层路由。

分层策略（基于实测数据的决策）
------------------------------
实测：纯图像匹配在真实扰动下 Top-1 仅 35%，而 OCR 对中文 UI 文字几乎 100% 正确。
因此优先级是 **OCR > 图标匹配**，VLM 只做兜底和界面理解。

四层
----
1. OCR      文字信号（可靠、免费、可解释）
2. 图标匹配  视觉信号（Top-K 召回，不作为最终裁决）
3. VLM      界面类型判断 + 低置信兜底（选择题形式，不给开放题）
4. 用户确认  置信不足时提供候选，兼顾体验与标注采集
"""
from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from backend.app.config import settings
from backend.app.llm import LLMError, get_llm
from backend.app.vision.icons import get_icon_index
from backend.app.vision.ocr import get_ocr, summarize

SCREEN_SYSTEM = """你是一个界面理解助手。你看到的是《星露谷物语》的游戏截图。
根据界面布局判断这是什么界面，并输出 JSON：
{"screen": "inventory|shop|fishing|dialogue|quest|farm|cooking|other",
 "description": "一句话描述界面内容",
 "has_items": true/false}

只输出 JSON，不要解释。不要尝试识别具体物品名称。"""

VLM_CHOICE_PROMPT = """这是《星露谷物语》截图中的一个物品图标。
请从候选里选出最匹配的一个。候选：{candidates}

只输出 JSON：{{"choice": "候选名", "confidence": 0.0-1.0, "reason": "简短理由"}}
如果都不像，choice 填 "都不是"。"""


@dataclass
class VisionResult:
    screen: str = 'unknown'
    description: str = ''
    texts: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    prices: list[int] = field(default_factory=list)
    qualities: list[str] = field(default_factory=list)
    items: list[dict[str, Any]] = field(default_factory=list)
    ocr_confidence: float = 0.0
    vlm_used: bool = False
    needs_confirm: bool = False
    mocked: bool = False
    trace: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            'screen': self.screen,
            'description': self.description,
            'texts': self.texts[:20],
            'entities': self.entities,
            'prices': self.prices[:10],
            'qualities': self.qualities,
            'items': self.items,
            'ocr_confidence': self.ocr_confidence,
            'vlm_used': self.vlm_used,
            'needs_confirm': self.needs_confirm,
            'mocked': self.mocked,
        }


class VisionPipeline:
    def __init__(self) -> None:
        self.ocr = get_ocr()
        self.icons = get_icon_index()
        self.llm = get_llm()

    async def analyze(self, image_bytes: bytes, *, question: str = '',
                      use_vlm: bool = True) -> VisionResult:
        out = VisionResult()

        # ---- 加载图片（限制尺寸，控制耗时）
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.load()
        except Exception as e:  # noqa: BLE001
            out.description = f'图片解析失败：{e}'
            return out
        if img.mode not in ('RGB', 'RGBA'):
            img = img.convert('RGB')
        max_side = 1400
        if max(img.size) > max_side:
            ratio = max_side / max(img.size)
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
        out.trace.append({'step': 'load', 'size': img.size})

        # ---- 第一层：OCR
        if settings.ocr_enabled and self.ocr.available:
            boxes = self.ocr.extract(img)
            summary = summarize(boxes)
            out.texts = summary['texts']
            out.entities = summary['entities']
            out.prices = summary['prices']
            out.qualities = summary['qualities']
            out.ocr_confidence = summary['avg_confidence']
            out.trace.append({'step': 'ocr', 'boxes': summary['n_boxes'],
                              'confidence': summary['avg_confidence']})
        else:
            out.trace.append({'step': 'ocr', 'skipped': '引擎不可用'})

        # ---- 第二层：图标匹配（整图 + 网格切片）
        icon_matches = self._match_icons(img)
        if icon_matches:
            out.items = icon_matches
            scores = [m['score'] for m in icon_matches]
            high = sum(1 for s in scores if s >= settings.icon_confidence_high)
            mid = sum(1 for s in scores if settings.icon_confidence_mid <= s
                      < settings.icon_confidence_high)
            out.trace.append({'step': 'icons', 'candidates': len(icon_matches),
                              'high': high, 'mid': mid,
                              'top': [(m['name'], m['score']) for m in icon_matches[:3]]})
            # 有中置信候选但无高置信 → 需要 VLM 或用户确认
            if mid and not high:
                out.needs_confirm = True

        # ---- 第三层：VLM（界面理解 + 低置信兜底）
        if use_vlm and self.llm.live:
            await self._vlm_stage(img, out, question)
        elif use_vlm:
            out.mocked = True

        # 无任何信号 → 标记需确认
        if not out.texts and not out.items:
            out.needs_confirm = True
        return out

    # ---------------------------------------------------------------- 图标

    def _match_icons(self, img: Image.Image) -> list[dict[str, Any]]:
        """整图匹配 + 网格切片匹配，返回候选（去重后按分数排序）。"""
        if self.icons.size == 0:
            return []
        cands: list[dict[str, Any]] = []

        # 整图（适用于用户已裁好的单图标截图）
        for m in self.icons.match(img, top_k=2):
            cands.append(m.as_dict())

        # 网格切片（适用于整屏截图）：按 48 的整数倍滑动
        for tile in _slice_tiles(img):
            for m in self.icons.match(tile, top_k=1):
                if m.score >= settings.icon_confidence_mid:
                    cands.append(m.as_dict())

        # 按 key 去重，保留最高分
        best: dict[str, dict[str, Any]] = {}
        for c in cands:
            k = c['key']
            if k not in best or c['score'] > best[k]['score']:
                best[k] = c
        ranked = sorted(best.values(), key=lambda x: -x['score'])
        return ranked[: settings.icon_match_topk]

    # ---------------------------------------------------------------- VLM

    async def _vlm_stage(self, img: Image.Image, out: VisionResult,
                         question: str) -> None:
        b64 = _to_b64(img, max_side=900)

        # (a) 界面类型
        try:
            res = await self.llm.vision(SCREEN_SYSTEM, b64, max_tokens=200)
            data = _json(res.content) or {}
            out.screen = data.get('screen', 'unknown')
            out.description = data.get('description', '')
            out.vlm_used = True
            out.mocked = out.mocked or res.mocked
            out.trace.append({'step': 'vlm_screen', 'screen': out.screen,
                              'mocked': res.mocked})
        except LLMError as e:
            out.trace.append({'step': 'vlm_screen_error', 'error': str(e)[:150]})

        # (b) 图标低置信兜底：给候选做选择题
        if out.needs_confirm and out.items:
            cands = [i['name'] for i in out.items[:3]]
            extra = _nearby_entity_candidates(out.entities, cands)
            prompt = VLM_CHOICE_PROMPT.format(candidates='、'.join(extra))
            try:
                res = await self.llm.vision(prompt, b64, max_tokens=200)
                data = _json(res.content) or {}
                choice = str(data.get('choice') or '').strip()
                conf = float(data.get('confidence') or 0.0)
                out.trace.append({'step': 'vlm_choice', 'choice': choice,
                                  'confidence': conf, 'mocked': res.mocked})
                if choice and choice != '都不是':
                    out.items.insert(0, {'key': choice, 'name': choice,
                                         'score': round(conf, 3), 'source': 'vlm'})
                    out.needs_confirm = conf < 0.7
            except LLMError as e:
                out.trace.append({'step': 'vlm_choice_error', 'error': str(e)[:150]})


# ---------------------------------------------------------------- 工具函数

def _slice_tiles(img: Image.Image) -> list[Image.Image]:
    """把整屏截图切成候选小格。

    星露谷 UI 格子多为 48/64 的整数倍；这里用滑动窗口覆盖常见格子尺寸，
    步长取格子尺寸的一半，避免图标正好卡在边界。
    """
    tiles: list[Image.Image] = []
    W, H = img.size
    if W < 48 or H < 48:
        return [img]
    sizes = [48, 64, 80]
    max_tiles = 24  # 控制开销
    for s in sizes:
        step = max(s // 2, 24)
        for y in range(0, H - s + 1, step):
            for x in range(0, W - s + 1, step):
                if len(tiles) >= max_tiles:
                    return tiles
                tiles.append(img.crop((x, y, x + s, y + s)))
    return tiles


def _to_b64(img: Image.Image, max_side: int = 900) -> str:
    if max(img.size) > max_side:
        ratio = max_side / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert('RGB').save(buf, format='PNG', optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def _json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    m = re.search(r'\{[\s\S]*\}', text)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None


def _nearby_entity_candidates(entities: list[str], base: list[str]) -> list[str]:
    """把 OCR 抽到的中文实体加入候选，提升 VLM 选择题命中率。"""
    out = list(dict.fromkeys(base))
    for e in entities:
        if e not in out:
            out.append(e)
        if len(out) >= 5:
            break
    return out


_pipeline: VisionPipeline | None = None


def get_vision() -> VisionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = VisionPipeline()
    return _pipeline
