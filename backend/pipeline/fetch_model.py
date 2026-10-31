"""下载嵌入模型（bge-small-zh-v1.5 ONNX 量化版）。

只依赖标准库，无需预先安装 onnxruntime，方便首次部署。

    python -m backend.pipeline.fetch_model
"""
from __future__ import annotations

import os
import ssl
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DEST = ROOT / 'backend' / 'data' / 'models' / 'bge-small-zh-v1.5'

REPO = 'Xenova/bge-small-zh-v1.5'
FILES = {
    'onnx/model_quantized.onnx': 'model_quantized.onnx',
    'tokenizer.json': 'tokenizer.json',
    'tokenizer_config.json': 'tokenizer_config.json',
    'config.json': 'config.json',
}

MIRRORS = ['https://hf-mirror.com', 'https://huggingface.co']

UA = {'User-Agent': 'StardewPal/0.1 (model download)'}


def _ctx():
    c = ssl.create_default_context()
    c.check_hostname = False
    c.verify_mode = ssl.CERT_NONE
    return c


def download(remote: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1000:
        print(f'  ✓ {dest.name} 已存在 ({dest.stat().st_size/1048576:.1f} MB)')
        return
    last = ''
    for base in MIRRORS:
        url = f'{base}/{REPO}/resolve/main/{remote}'
        try:
            print(f'  下载 {remote} ← {base.split("//")[1]}', end=' ', flush=True)
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=180, context=_ctx()) as r:
                data = r.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            print(f'✓ {len(data)/1048576:.1f} MB')
            return
        except Exception as e:  # noqa: BLE001
            last = str(e)[:120]
            print(f'✗ {last}')
    raise SystemExit(f'下载失败: {remote} — {last}')


def main() -> None:
    print(f'目标目录: {DEST}')
    DEST.mkdir(parents=True, exist_ok=True)
    for remote, local in FILES.items():
        download(remote, DEST / local)

    total = sum(f.stat().st_size for f in DEST.iterdir() if f.is_file())
    print(f'\n完成，模型体积 {total/1048576:.1f} MB')

    need = DEST / 'model_quantized.onnx'
    if not need.exists():
        sys.exit('缺少 model_quantized.onnx')


if __name__ == '__main__':
    main()
