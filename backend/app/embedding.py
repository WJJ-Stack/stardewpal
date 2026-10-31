"""本地 ONNX 嵌入器：bge-small-zh-v1.5（INT8 量化，512 维）。

设计取舍
--------
不引入 torch —— 模型 23MB、纯 CPU、1024 条/秒，
在免费 CPU 容器里足够，且镜像体积可控。
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

DIM = 512


class Embedder:
    """线程安全的 ONNX 文本嵌入器（mean pooling + L2 归一化）。"""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self._lock = threading.Lock()
        self._session = None
        self._tokenizer = None

    # ---------------------------------------------------------------- 懒加载
    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            import onnxruntime as ort
            from tokenizers import Tokenizer

            model_path = self.model_dir / "model_quantized.onnx"
            tok_path = self.model_dir / "tokenizer.json"
            if not model_path.exists() or not tok_path.exists():
                raise FileNotFoundError(
                    f"嵌入模型缺失: {model_path}. 先运行 python -m backend.pipeline.fetch_model"
                )

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, min(4, self._cpu_count()))
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(model_path), opts, providers=["CPUExecutionProvider"]
            )
            self._tokenizer = Tokenizer.from_file(str(tok_path))
            self._tokenizer.enable_truncation(max_length=512)
            self._input_names = {i.name for i in self._session.get_inputs()}

    @staticmethod
    def _cpu_count() -> int:
        import os

        try:
            return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
        except AttributeError:
            return os.cpu_count() or 1

    @property
    def ready(self) -> bool:
        try:
            self._ensure_loaded()
            return True
        except Exception:
            return False

    # ---------------------------------------------------------------- 编码
    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """返回 (n, 512) 的 L2 归一化向量。"""
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        self._ensure_loaded()
        out: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            out.append(self._encode_batch(texts[i : i + batch_size]))
        return np.vstack(out).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        encs = self._tokenizer.encode_batch(texts)
        max_len = max(len(e.ids) for e in encs)
        n = len(texts)
        ids = np.zeros((n, max_len), dtype=np.int64)
        mask = np.zeros((n, max_len), dtype=np.int64)
        types = np.zeros((n, max_len), dtype=np.int64)
        for j, e in enumerate(encs):
            k = len(e.ids)
            ids[j, :k] = e.ids
            mask[j, :k] = e.attention_mask
            types[j, :k] = e.type_ids

        feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}
        hidden = self._session.run(None, feeds)[0]

        m = mask[:, :, None].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled / np.maximum(norms, 1e-9)


_embedder: Embedder | None = None


def get_embedder(model_dir: Path | None = None) -> Embedder:
    global _embedder
    if _embedder is None:
        from backend.app.config import settings

        _embedder = Embedder(model_dir or settings.model_dir)
    return _embedder
