"""模拟「LFS 未还原」的容器环境，验证自检修复是否有效。

复现方式：把所有 LFS 文件替换成指针文本，然后跑 preflight，
看它能否自动恢复（重新下载模型 + 用 pages.jsonl 重建索引）。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SIM = ROOT.parent / 'ms-lfs-sim'

LFS_POINTER = (
    'version https://git-lfs.github.com/spec/v1\n'
    'oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n'
    'size %d\n'
)


def make_sim() -> None:
    if SIM.exists():
        shutil.rmtree(SIM)
    print(f'创建模拟环境: {SIM}')

    # 复制代码与语料（不含 LFS 大文件）
    for d in ['backend']:
        src = ROOT / d
        for f in src.rglob('*'):
            if not f.is_file():
                continue
            if '__pycache__' in f.parts:
                continue
            rel = f.relative_to(ROOT)
            # 把 LFS 文件替换成指针
            if f.suffix in {'.onnx', '.zip'} or f.name.endswith('.db'):
                dst = SIM / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(LFS_POINTER % f.stat().st_size)
                print(f'  指针化: {rel}')
            elif f.suffix in {'.pyc'}:
                continue
            else:
                dst = SIM / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)

    for d in ['frontend']:
        src = ROOT / d
        if src.exists():
            shutil.copytree(src, SIM / d)

    for f in ['Dockerfile', 'requirements.txt', 'pyproject.toml']:
        if (ROOT / f).exists():
            shutil.copy2(ROOT / f, SIM / f)

    print('模拟环境就绪')


def check_state() -> None:
    db = SIM / 'backend/data/stardewpal.db'
    model = SIM / 'backend/data/models/bge-small-zh-v1.5/model_quantized.onnx'
    zipf = SIM / 'backend/data/icons.zip'
    corpus = SIM / 'backend/data/raw/pages.jsonl'

    for label, p in [('db', db), ('model', model), ('icons.zip', zipf),
                     ('pages.jsonl', corpus)]:
        state = '缺失'
        if p.exists():
            head = p.open('rb').read(60)
            if head.startswith(b'version https://git-lfs'):
                state = f'LFS 指针 ({p.stat().st_size} B)'
            else:
                state = f'真实文件 ({p.stat().st_size/1048576:.2f} MB)'
        print(f'  {label:14s} {state}')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--check':
        check_state()
    else:
        make_sim()
        print('\n模拟后的状态：')
        check_state()
