"""启动入口：uvicorn 封装 + 启动期自检（索引/图标）。

    python -m backend.app.serve          # 默认 0.0.0.0:7860
    python -m backend.app.serve --port 8000

启动期自检为什么必要
--------------------
部署环境（如魔搭创空间）在构建镜像时，Git LFS 文件可能未被正确还原，
导致 stardewpal.db 变成 LFS 指针文本，SQLite 打开时报
"file is not a database"。

因此启动前做三件事：
1. 校验 db 文件头，不是 SQLite 就删除
2. 若 db 缺失但有语料（pages.jsonl），自动重建索引
3. 若图标是 zip 未解包，先解包
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _is_valid_sqlite(p: Path) -> bool:
    """SQLite 文件头是 'SQLite format 3\\x00'（16 字节）。"""
    try:
        with p.open('rb') as f:
            return f.read(16) == b'SQLite format 3\x00'
    except OSError:
        return False


def _looks_like_lfs_pointer(p: Path) -> bool:
    """Git LFS 指针文件的特征：开头是 'version https://git-lfs...'。"""
    try:
        with p.open('rb') as f:
            head = f.read(120)
        return head.startswith(b'version https://git-lfs')
    except OSError:
        return False


def _is_valid_onnx(p: Path, min_bytes: int = 1_000_000) -> bool:
    """ONNX 量化模型应远大于 1MB；指针文件只有几百字节。"""
    try:
        return p.stat().st_size >= min_bytes and not _looks_like_lfs_pointer(p)
    except OSError:
        return False


def _preflight() -> None:
    from backend.app.config import settings

    db = settings.db_path
    corpus = settings.raw_dir / 'pages.jsonl'
    data_dir = settings.db_path.parent

    # ---- 1. 校验索引（LFS 指针 / 损坏 → 删除后重建）
    if db.exists() and not _is_valid_sqlite(db):
        size = db.stat().st_size
        kind = 'LFS 指针' if _looks_like_lfs_pointer(db) else '损坏'
        print(f'[preflight] 索引文件是{kind}（{size} 字节），删除后重建')
        db.unlink()

    # ---- 2. 校验嵌入模型（LFS 指针 → 重新下载）
    model = settings.model_dir / 'model_quantized.onnx'
    if model.exists() and not _is_valid_onnx(model):
        print(f'[preflight] 嵌入模型是 LFS 指针（{model.stat().st_size} 字节），重新下载')
        try:
            model.unlink()
            subprocess.run([sys.executable, '-m', 'backend.pipeline.fetch_model'], check=False)
        except Exception as e:  # noqa: BLE001
            print(f'[preflight] 模型下载失败: {e}')

    # ---- 3. 解包图标（zip 若为指针则跳过，OCR 仍可用）
    icons_zip = data_dir / 'icons.zip'
    n_icons = len(list(settings.icons_dir.glob('*.png'))) if settings.icons_dir.exists() else 0
    if n_icons == 0 and icons_zip.exists():
        if _looks_like_lfs_pointer(icons_zip) or icons_zip.stat().st_size < 100_000:
            print('[preflight] icons.zip 是 LFS 指针，跳过图标解包（OCR 仍可用）')
        else:
            print('[preflight] 解包图标...')
            subprocess.run([sys.executable, '-m', 'backend.pipeline.bundle_assets', '--unpack'],
                           check=False)

    # ---- 4. 必要时重建索引
    if not db.exists():
        if not corpus.exists():
            print('[preflight] 索引缺失且无语料，服务将以空索引启动')
            return
        print(f'[preflight] 从 {corpus.name} 重建索引（约 2 分钟）...')
        r = subprocess.run([sys.executable, '-m', 'backend.pipeline.build_index', '--rebuild'],
                           check=False)
        if r.returncode != 0:
            print(f'[preflight] 索引重建失败（exit {r.returncode}），将以空索引启动')
        else:
            print('[preflight] 索引重建完成')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default=None)
    ap.add_argument('--port', type=int, default=None)
    ap.add_argument('--reload', action='store_true')
    ap.add_argument('--skip-preflight', action='store_true',
                    help='跳过启动自检（本地开发用）')
    args = ap.parse_args()

    from backend.app.config import settings

    if not args.skip_preflight:
        try:
            _preflight()
        except Exception as e:  # noqa: BLE001
            print(f'[preflight] 自检异常，继续启动: {e}')

    import uvicorn

    uvicorn.run(
        'backend.app.main:app',
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
        log_level='info',
        proxy_headers=True,
        forwarded_allow_ips='*',
    )


if __name__ == '__main__':
    main()
