"""组装魔搭创空间的推送目录。

用法
----
    python -m backend.pipeline.prepare_modelscope
    python -m backend.pipeline.prepare_modelscope --out D:\\ms-push

为什么需要这个脚本
------------------
魔搭创空间是一个独立的 Git 仓库，直接推整个项目目录会把
.git / .venv / .archive 等无关内容一起推上去。

本脚本只拷贝部署必需的子集到临时目录，然后你在那里执行 git push。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
# 暂存目录放在仓库内（.archive 已 gitignore）。
# 不放仓库外是为了避免清理时需要跨工作区递归删除。
DEFAULT_OUT = ROOT / '.archive' / 'ms-push'

# 部署必需的顶层条目
INCLUDE_DIRS = ['backend', 'frontend', 'scripts']
INCLUDE_FILES = ['Dockerfile', 'requirements.txt', 'pyproject.toml']

# 需要排除的子路径（相对项目根）
EXCLUDE_PATTERNS = [
    '__pycache__',
    '.pytest_cache',
    '.ruff_cache',
    '*.pyc',
    'data/runs',
    'data/eval',
    '*.db-wal',
    '*.db-shm',
    # 原始 wikitext（7.7MB）只在离线重跑清洗时用得到，
    # 运行时建索引只读 pages.jsonl。不传给线上，省带宽和构建时间。
    'wikitext.jsonl',
]


def should_skip(rel: Path) -> bool:
    parts = rel.parts
    if any(p in EXCLUDE_PATTERNS for p in parts):
        return True
    name = rel.name
    return any(
        name == pat or (pat.startswith('*') and name.endswith(pat[1:]))
        for pat in EXCLUDE_PATTERNS
    )


def copytree(src: Path, dst: Path) -> int:
    count = 0
    for item in src.rglob('*'):
        if not item.is_file():
            continue
        rel = item.relative_to(src)
        if should_skip(rel):
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out)

    if out.exists():
        print(f'清空已存在的目录：{out}')
        shutil.rmtree(out)
    out.mkdir(parents=True)

    total = 0
    for d in INCLUDE_DIRS:
        src = ROOT / d
        if src.exists():
            n = copytree(src, out / d)
            print(f'  {d:12s} {n:5d} 个文件')
            total += n
        else:
            print(f'  {d:12s} 跳过（不存在）')

    for f in INCLUDE_FILES:
        src = ROOT / f
        if src.exists():
            shutil.copy2(src, out / f)
            print(f'  {f:12s}     1 个文件')
            total += 1
        else:
            print(f'  {f:12s} 跳过（不存在）')

    # 魔搭专用 README（带 frontmatter）
    ms_readme = ROOT / 'docs' / 'modelscope-README.md'
    if ms_readme.exists():
        shutil.copy2(ms_readme, out / 'README.md')
        print('  README.md    (魔搭专用版，含 frontmatter)')
        total += 1

    # .gitattributes：沿用魔搭模板，让 .onnx / .zip / .db 自动走 LFS
    gitattr = ROOT / '.gitattributes'
    if gitattr.exists():
        shutil.copy2(gitattr, out / '.gitattributes')
    else:
        print('  警告：缺 .gitattributes，大文件可能无法走 LFS')

    # 汇总
    size = sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
    print(f'\n合计 {total} 个文件, {size/1048576:.1f} MB')
    print(f'输出目录：{out}')

    # 体积检查
    big = [(f, f.stat().st_size) for f in out.rglob('*')
           if f.is_file() and f.stat().st_size > 10 * 1048576]
    if big:
        print('\n超过 10MB 的文件（需 LFS）：')
        for f, s in sorted(big, key=lambda x: -x[1]):
            print(f'  {s/1048576:6.1f} MB  {f.relative_to(out)}')

    print('\n下一步（在该目录执行）：')
    print(f'  cd /d "{out}"')
    print('  git init -b master')
    print('  git remote add origin https://www.modelscope.cn/studios/wujaijie/stardewpal.git')
    print('  git add -A')
    print('  git commit -m "deploy: StardewPal multimodal RAG"')
    print('  git push -u origin master --force')


if __name__ == '__main__':
    sys.exit(main())
