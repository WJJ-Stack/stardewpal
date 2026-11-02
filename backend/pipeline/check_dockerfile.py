"""本地验证 Dockerfile 里那段构建期自检逻辑（不依赖 Docker）。"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT.parent / 'ms-lfs-sim'   # 已模拟好 LFS 指针的目录


def extract_build_script(dockerfile: Path) -> str:
    text = dockerfile.read_text(encoding='utf-8')
    m = re.search(r"RUN python - <<'PY'\n(.*?)\nPY", text, re.S)
    if not m:
        raise SystemExit('未在 Dockerfile 中找到自检脚本')
    return m.group(1)


def main() -> None:
    print('=== 模拟目录当前状态 ===')
    data = SRC / 'backend/data'
    for name, p in [
        ('模型', data / 'models/bge-small-zh-v1.5/model_quantized.onnx'),
        ('索引', data / 'stardewpal.db'),
        ('图标包', data / 'icons.zip'),
        ('语料', data / 'raw/pages.jsonl'),
    ]:
        if p.exists():
            head = p.open('rb').read(40)
            kind = 'LFS 指针' if head.startswith(b'version https://git-lfs') else '真实文件'
            print(f'  {name:6s} {kind:8s} {p.stat().st_size:>10,} B')
        else:
            print(f'  {name:6s} 缺失')

    script = extract_build_script(ROOT / 'Dockerfile')
    # 把 /app 替换成模拟目录
    script = script.replace('/app', str(SRC).replace('\\', '/'))

    print('\n=== 执行 Dockerfile 自检脚本 ===')
    r = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    print(r.stdout or '(无输出)')
    if r.stderr:
        print('STDERR:', r.stderr[:500])
    print(f'退出码: {r.returncode}')

    print('\n=== 执行后状态 ===')
    for name, p in [
        ('模型', data / 'models/bge-small-zh-v1.5/model_quantized.onnx'),
        ('索引', data / 'stardewpal.db'),
        ('图标包', data / 'icons.zip'),
        ('语料', data / 'raw/pages.jsonl'),
    ]:
        if p.exists():
            head = p.open('rb').read(40)
            kind = 'LFS 指针' if head.startswith(b'version https://git-lfs') else '真实文件'
            print(f'  {name:6s} {kind:8s} {p.stat().st_size:>10,} B')
        else:
            print(f'  {name:6s} 已删除（等待运行时自愈）')


if __name__ == '__main__':
    main()
