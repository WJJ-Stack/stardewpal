"""把准备好的项目目录上传到魔搭创空间（走 HTTP API，不依赖 Git 凭据）。

为什么不用 git push
-------------------
Windows 上 Git Credential Manager 在非交互环境会卡住弹窗等待输入。
SDK 的 upload_file / upload_folder 走 HTTP API，用已登录的令牌即可，
更适合自动化。

用法
----
    python -m backend.pipeline.push_modelscope --dry-run    # 只列出文件
    python -m backend.pipeline.push_modelscope              # 实际上传
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT.parent / 'ms-push'
REPO = 'wujaijie/stardewpal'

# 需要上传的（相对路径 glob），按优先级排序：先小文件后大文件
PRIORITY_DIRS = ['frontend', 'scripts']
PRIORITY_FILES = ['Dockerfile', 'requirements.txt', 'pyproject.toml', 'README.md',
                  '.gitattributes']


SKIP_DIRS = {'__pycache__', '.pytest_cache', '.ruff_cache', '.git', '.venv'}
SKIP_SUFFIX = {'.pyc', '.pyo', '.db-shm', '.db-wal'}


def _keep(f: Path) -> bool:
    if any(p in SKIP_DIRS for p in f.parts):
        return False
    return f.suffix not in SKIP_SUFFIX


def collect(src: Path) -> list[Path]:
    files = []
    for name in PRIORITY_FILES:
        p = src / name
        if p.exists():
            files.append(p)
    for d in PRIORITY_DIRS:
        base = src / d
        if base.exists():
            files.extend(sorted(f for f in base.rglob('*') if f.is_file() and _keep(f)))
    # backend：先小文件（代码），后大文件（数据）
    backend = src / 'backend'
    if backend.exists():
        allb = [f for f in backend.rglob('*') if f.is_file() and _keep(f)]
        small = [f for f in allb if f.stat().st_size < 1_000_000]
        large = [f for f in allb if f.stat().st_size >= 1_000_000]
        files.extend(sorted(small))
        files.extend(sorted(large, key=lambda x: x.stat().st_size))
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--skip-existing', action='store_true',
                    help='跳过已上传的文件（按大小粗判）')
    args = ap.parse_args()

    if not SRC.exists():
        raise SystemExit(f'源目录不存在：{SRC}，先运行 prepare_modelscope')

    files = collect(SRC)
    total = sum(f.stat().st_size for f in files)
    print(f'待上传 {len(files)} 个文件, {total/1048576:.1f} MB')

    if args.dry_run:
        for f in files[:20]:
            print(f'  {f.relative_to(SRC)}  ({f.stat().st_size/1024:.0f} KB)')
        if len(files) > 20:
            print(f'  ... 其余 {len(files)-20} 个')
        return

    from modelscope.hub.api import HubApi

    api = HubApi()
    print('开始上传（并行、支持断点续传）...')
    t0 = time.time()

    def _progress(info: dict) -> None:
        done = info.get('completed_files') or info.get('files_done')
        total_f = info.get('total_files') or info.get('files_total')
        if done and total_f:
            pct = done / total_f * 100
            print(f'  {done}/{total_f} ({pct:.0f}%)', flush=True)

    try:
        result = api.upload_folder(
            repo_id=REPO,
            folder_path=str(SRC),
            repo_type='studio',
            commit_message='deploy: StardewPal multimodal RAG',
            ignore_patterns=['__pycache__/*', '*.pyc', '.pytest_cache/*', '*.log'],
            max_workers=8,
            progress_callback=_progress,
        )
        elapsed = time.time() - t0
        print(f'\n完成，用时 {elapsed:.0f}s')
        print(f'结果: {str(result)[:300]}')
    except Exception as e:  # noqa: BLE001
        print(f'上传失败: {str(e)[:400]}')
        sys.exit(1)


if __name__ == '__main__':
    main()
