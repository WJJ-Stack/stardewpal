"""图标匹配：pHash 召回 + 多特征精排。

为什么这样设计
--------------
实测结论（320 个真实图标）：
- 干净输入：native 像素特征 100%，pHash 94.7%
- 真实扰动（缩放+模糊+JPEG+亮度）：native 35%，pHash 19.7%

单靠任一特征都不够。因此采用「召回-精排」两级：
1. pHash 粗筛取 Top-20（快，且对亮度/压缩鲁棒）
2. 多特征加权精排（像素结构 + 颜色分布 + 形状矩）
3. 输出 Top-K + 置信度，低置信则交给上层降级

明确边界：48x48 像素图标信息量有限，且同一图标可能被多个物品复用
（实测 Armored_Bug 与 Armored_Bug_Dangerous 像素完全相同），
所以本模块只提供候选，**不做最终裁决**。
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn


@dataclass
class Match:
    key: str
    name: str
    score: float
    icon_file: str | None = None

    def as_dict(self) -> dict:
        return {'key': self.key, 'name': self.name,
                'score': round(self.score, 4), 'icon': self.icon_file}


def _load_rgba(img: Image.Image) -> Image.Image:
    return img.convert('RGBA') if img.mode != 'RGBA' else img


def feat_phash(img: Image.Image, hash_size: int = 8) -> np.ndarray:
    """感知哈希：DCT 低频符号。对亮度/压缩鲁棒，用于召回。"""
    bg = Image.new('RGBA', img.size, (128, 128, 128, 255))
    comp = Image.alpha_composite(bg, _load_rgba(img)).convert('L')
    a = np.asarray(comp.resize((32, 32), Image.LANCZOS), dtype=np.float32)
    d = np.abs(dctn(a, axes=(0, 1)))[:hash_size, :hash_size]
    return (d > np.median(d)).astype(np.uint8).flatten()


def feat_pixel(img: Image.Image, size: int = 16) -> np.ndarray:
    """像素结构特征：降采样到 16x16，alpha 加权。对同源图最准。"""
    a = np.asarray(_load_rgba(img).resize((size, size), Image.NEAREST))
    rgb = a[:, :, :3].astype(np.float32) / 255.0
    alpha = a[:, :, 3:4].astype(np.float32) / 255.0
    return np.concatenate([rgb * alpha, alpha], axis=2).reshape(-1)


def feat_color(img: Image.Image, bins: int = 4) -> np.ndarray:
    """颜色直方图（RGB 分桶）。对形状变化鲁棒。"""
    arr = np.asarray(_load_rgba(img))
    mask = arr[:, :, 3] > 128
    px = arr[:, :, :3][mask] if mask.sum() > 0 else arr[:, :, :3].reshape(-1, 3)
    hist = []
    for c in range(3):
        h, _ = np.histogram(px[:, c], bins=bins, range=(0, 256))
        hist.extend(h / max(len(px), 1))
    return np.array(hist, dtype=np.float32)


def feat_shape(img: Image.Image, size: int = 16) -> np.ndarray:
    """形状特征：alpha 轮廓的二值投影，捕捉外形。"""
    a = np.asarray(_load_rgba(img).resize((size, size), Image.NEAREST))[:, :, 3]
    return (a > 40).astype(np.float32).flatten()


class IconIndex:
    """内存图标索引（图标量级小，无需落库）。"""

    def __init__(self) -> None:
        self.keys: list[str] = []
        self.names: list[str] = []
        self.files: list[str] = []
        self.phash = np.zeros((0, 64), dtype=np.uint8)
        self.pixel = np.zeros((0, 16 * 16 * 4), dtype=np.float32)
        self.color = np.zeros((0, 12), dtype=np.float32)
        self.shape = np.zeros((0, 16 * 16), dtype=np.float32)
        self._loaded = False

    @property
    def size(self) -> int:
        return len(self.keys)

    def load(self, icons_dir: Path, key_map: dict[str, str] | None = None,
             limit: int | None = None) -> int:
        """从目录加载图标。key_map: 文件名(stem小写) → 中文名。"""
        icons_dir = Path(icons_dir)
        key_map = key_map or {}
        ph, px, cl, sh = [], [], [], []
        for f in sorted(icons_dir.glob('*.png')):
            if limit and len(ph) >= limit:
                break
            try:
                im = Image.open(f)
                im.load()
                if min(im.size) < 8:
                    continue
            except Exception:
                continue
            stem = f.stem
            self.keys.append(stem)
            self.names.append(key_map.get(stem.lower(), stem))
            self.files.append(f.name)
            ph.append(feat_phash(im))
            px.append(feat_pixel(im))
            cl.append(feat_color(im))
            sh.append(feat_shape(im))

        if ph:
            self.phash = np.vstack(ph).astype(np.uint8)
            self.pixel = np.vstack(px).astype(np.float32)
            self.color = np.vstack(cl).astype(np.float32)
            self.shape = np.vstack(sh).astype(np.float32)
        self._loaded = True
        return self.size

    def _ensure(self) -> None:
        if not self._loaded:
            from backend.app.config import settings

            self.load(settings.icons_dir)

    def match(self, img: Image.Image, top_k: int = 3,
              recall: int = 20) -> list[Match]:
        """两级匹配：pHash 召回 → 多特征精排。"""
        self._ensure()
        if self.size == 0:
            return []

        qh = feat_phash(img)
        # 第一级：汉明距离召回
        ham = (self.phash != qh).sum(axis=1)
        cand = np.argsort(ham)[: min(recall, self.size)]

        # 第二级：多特征加权精排
        qp = feat_pixel(img)
        qc = feat_color(img)
        qs = feat_shape(img)

        scored: list[tuple[float, int]] = []
        for idx in cand:
            d_pix = float(np.abs(qp - self.pixel[idx]).mean())
            d_col = float(np.abs(qc - self.color[idx]).sum() / 2.0)
            d_shp = float(np.abs(qs - self.shape[idx]).mean())
            d_ham = float(ham[idx]) / 64.0
            # 权重：像素结构最重（同源图最准），其余补偿扰动
            score = (0.42 * (1.0 - d_pix)
                     + 0.18 * (1.0 - d_col)
                     + 0.20 * (1.0 - d_shp)
                     + 0.20 * (1.0 - d_ham))
            scored.append((score, int(idx)))

        scored.sort(reverse=True)
        return [
            Match(key=self.keys[i], name=self.names[i], score=s, icon_file=self.files[i])
            for s, i in scored[:top_k]
        ]


_index: IconIndex | None = None


def get_icon_index() -> IconIndex:
    global _index
    if _index is None:
        _index = IconIndex()
    return _index
