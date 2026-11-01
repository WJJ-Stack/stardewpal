"""打包部署产物：把图标库压成单个文件，减小仓库体积与文件数。

用法
----
    python -m backend.pipeline.bundle_assets          # 打包
    python -m backend.pipeline.bundle_assets --unpack  # 解包（容器启动时调用）

设计考虑
--------
1200 个图标分成 1200 个 Git 对象会让 clone 变慢、仓库列表变乱。
打包成单个 .zip 后：
- 文件数从 1200 → 1
- 体积基本不变（PNG 已压缩，zip 不再压缩只是归档）
- 容器启动时解包约 100ms，可接受
"""
from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ICONS = ROOT / 'backend' / 'data' / 'icons'
ARCHIVE = ROOT / 'backend' / 'data' / 'icons.zip'


def bundle() -> None:
    if not ICONS.exists():
        raise SystemExit(f'图标目录不存在：{ICONS}')
    files = sorted(ICONS.glob('*.png'))
    if not files:
        raise SystemExit(f'图标目录为空：{ICONS}')

    t0 = time.time()
    with zipfile.ZipFile(ARCHIVE, 'w', zipfile.ZIP_STORED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)

    size = ARCHIVE.stat().st_size
    print(f'打包完成：{len(files)} 个图标 → {ARCHIVE.name} ({size/1048576:.2f} MB, '
          f'{time.time()-t0:.2f}s)')


def unpack() -> None:
    if not ARCHIVE.exists():
        print(f'归档不存在，跳过解包：{ARCHIVE}')
        return
    ICONS.mkdir(parents=True, exist_ok=True)
    existing = len(list(ICONS.glob('*.png')))
    if existing:
        print(f'图标已存在 {existing} 个，跳过解包')
        return

    t0 = time.time()
    with zipfile.ZipFile(ARCHIVE) as zf:
        zf.extractall(ICONS)
    n = len(list(ICONS.glob('*.png')))
    print(f'解包完成：{n} 个图标 ({time.time()-t0:.2f}s)')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--unpack', action='store_true', help='解包而不是打包')
    args = ap.parse_args()
    if args.unpack:
        unpack()
    else:
        bundle()


if __name__ == '__main__':
    sys.exit(main())
