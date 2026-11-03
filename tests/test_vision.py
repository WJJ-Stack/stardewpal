"""视觉链路单元测试：OCR 切分、图标特征、置信度路由。"""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from backend.app.vision.icons import IconIndex, feat_color, feat_phash, feat_pixel, feat_shape
from backend.app.vision.ocr import TextBox, sort_reading_order, split_glued_text, summarize


def _solid(color=(200, 120, 60, 255), size=(48, 48)) -> Image.Image:
    return Image.new('RGBA', size, color)


class TestIconFeatures:
    def test_phash_length(self):
        h = feat_phash(_solid())
        assert h.shape == (64,)
        assert h.dtype == np.uint8

    def test_pixel_length(self):
        v = feat_pixel(_solid())
        assert v.shape == (16 * 16 * 4,)

    def test_color_histogram(self):
        v = feat_color(_solid())
        assert v.shape == (12,)
        assert pytest.approx(v.sum(), abs=0.01) == 3.0  # 三通道各归一到 1

    def test_shape_mask(self):
        v = feat_shape(_solid((0, 0, 0, 255)))
        assert v.shape == (16 * 16,)
        assert v.sum() == 16 * 16  # 全不透明

    def test_identical_images_match(self):
        idx = IconIndex()
        a = _solid((200, 120, 60, 255))
        idx.keys, idx.names, idx.files = ['a'], ['甲'], ['a.png']
        idx.phash = feat_phash(a)[None, :]
        idx.pixel = feat_pixel(a)[None, :]
        idx.color = feat_color(a)[None, :]
        idx.shape = feat_shape(a)[None, :]
        idx._loaded = True
        m = idx.match(a, top_k=1)
        assert m[0].key == 'a'
        assert m[0].score > 0.95

    def test_empty_index_returns_empty(self):
        idx = IconIndex()
        idx._loaded = True
        assert idx.match(_solid()) == []


class TestTextLayout:
    def test_reading_order_top_to_bottom(self):
        boxes = [
            TextBox('下', 0.9, (0, 50, 20, 60)),
            TextBox('上', 0.9, (0, 10, 20, 20)),
        ]
        out = sort_reading_order(boxes)
        assert [b.text for b in out] == ['上', '下']

    def test_reading_order_left_to_right(self):
        boxes = [
            TextBox('右', 0.9, (60, 10, 80, 20)),
            TextBox('左', 0.9, (0, 10, 20, 20)),
        ]
        out = sort_reading_order(boxes)
        assert [b.text for b in out] == ['左', '右']

    def test_split_glued_price(self):
        """OCR 把相邻格子连成 '卖价750金卖价50金' 时应能切开。"""
        boxes = [TextBox('卖价750金卖价50金', 0.9, (0, 0, 200, 30))]
        out = split_glued_text(boxes)
        assert len(out) == 2
        assert '750' in out[0].text
        assert '50' in out[1].text

    def test_no_split_when_single(self):
        boxes = [TextBox('防风草', 0.9, (0, 0, 50, 30))]
        out = split_glued_text(boxes)
        assert len(out) == 1


class TestSummarize:
    def test_extract_entities_and_prices(self):
        boxes = [
            TextBox('背包', 0.95, (0, 0, 40, 20)),
            TextBox('防风草', 0.92, (0, 30, 60, 50)),
            TextBox('卖价35金', 0.88, (0, 55, 80, 75)),
            TextBox('金星', 0.90, (0, 80, 40, 100)),
        ]
        s = summarize(boxes)
        assert '防风草' in s['entities']
        assert 35 in s['prices']
        assert '金星' in s['qualities']
        assert s['n_boxes'] == 4

    def test_empty(self):
        s = summarize([])
        assert s['n_boxes'] == 0
        assert s['avg_confidence'] == 0.0


class TestToB64:
    def test_png_roundtrip(self):
        from backend.app.vision.pipeline import _to_b64

        img = Image.new('RGB', (60, 60), (10, 20, 30))
        b64 = _to_b64(img)
        assert isinstance(b64, str) and len(b64) > 50
        raw = __import__('base64').b64decode(b64)
        assert raw[:4] == b'\x89PNG'


class TestImageHelpers:
    """v3 极简管线的图像工具（不再有切片/裁剪逻辑）。"""

    def test_fit_downscales_large(self):
        from backend.app.vision.pipeline import _fit

        img = Image.new('RGB', (3000, 2000), (0, 0, 0))
        out = _fit(img, 1400)
        assert max(out.size) == 1400

    def test_fit_keeps_small(self):
        from backend.app.vision.pipeline import _fit

        img = Image.new('RGB', (640, 480), (0, 0, 0))
        assert _fit(img, 1400).size == (640, 480)

    def test_load_converts_mode(self):

        from backend.app.vision.pipeline import _load

        buf = io.BytesIO()
        Image.new('P', (32, 32)).save(buf, format='PNG')
        out = _load(buf.getvalue())
        assert out.mode in ('RGB', 'RGBA')
