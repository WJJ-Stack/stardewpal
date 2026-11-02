"""通过 GitHub REST API 推送本地提交（绕过 github.com:443 被墙）。

背景
----
本网络环境下 github.com:443 不可达（git push 报
"Failed to connect to github.com port 443"），但 api.github.com 正常。
因此用 Git Data API 手动构造一次提交：
  1. 为每个改动文件创建 blob
  2. 创建 tree（基于父提交的 tree）
  3. 创建 commit
  4. 更新 refs/heads/master

用法
----
    python -m backend.pipeline.push_github            # 推送 HEAD 与远程的差异
    python -m backend.pipeline.push_github --dry-run
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

OWNER = 'WJJ-Stack'
REPO = 'stardewpal'
BRANCH = 'master'
API = 'https://api.github.com'


def _token() -> str:
    r = subprocess.run(['gh', 'auth', 'token'], capture_output=True, text=True, check=False)
    tok = r.stdout.strip()
    if not tok:
        raise SystemExit('无法获取 gh token，请先 gh auth login')
    return tok


def _req(method: str, path: str, token: str, payload: dict | None = None) -> dict:
    url = path if path.startswith('http') else f'{API}{path}'
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header('Authorization', f'Bearer {token}')
    req.add_header('Accept', 'application/vnd.github+json')
    req.add_header('User-Agent', 'stardewpal-deployer')
    if data:
        req.add_header('Content-Type', 'application/json')
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode('utf-8', 'ignore')[:400]
        raise SystemExit(f'{method} {path} -> HTTP {e.code}: {detail}') from e


def _git(*args: str) -> str:
    r = subprocess.run(['git', *args], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise SystemExit(f'git {" ".join(args)} 失败: {r.stderr[:300]}')
    return r.stdout.strip()


def changed_files(base: str, head: str) -> list[tuple[str, str]]:
    """返回 [(状态, 路径)]，状态为 A/M/D。"""
    out = _git('diff', '--name-status', base, head)
    items = []
    for line in out.splitlines():
        parts = line.split('\t')
        if len(parts) >= 2:
            items.append((parts[0], parts[-1]))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    token = _token()
    local_head = _git('rev-parse', 'HEAD')

    # 远程当前 master
    ref = _req('GET', f'/repos/{OWNER}/{REPO}/git/ref/heads/{BRANCH}', token)
    remote_sha = ref['object']['sha']
    print(f'远程 master : {remote_sha[:8]}')
    print(f'本地 HEAD   : {local_head[:8]}')

    if remote_sha == local_head:
        print('已同步，无需推送')
        return

    # 找分叉点
    try:
        merge_base = _git('merge-base', remote_sha, local_head)
    except SystemExit:
        merge_base = remote_sha
    print(f'分叉点      : {merge_base[:8]}')

    files = changed_files(merge_base, local_head)
    print(f'变更文件    : {len(files)} 个')

    # 只处理文本文件（大文件单独跳过并警告）
    blobs: dict[str, str | None] = {}
    skipped = []
    for status, path in files:
        if status == 'D':
            blobs[path] = None
            continue
        local = Path(path)
        if not local.exists():
            continue
        size = local.stat().st_size
        if size > 5 * 1024 * 1024:
            skipped.append((path, size))
            continue
        blobs[path] = str(local)

    if skipped:
        print('\n以下文件超过 5MB，API 推送会跳过（需 git push 或 LFS）：')
        for p, s in skipped:
            print(f'  {s/1048576:6.1f} MB  {p}')

    if args.dry_run:
        for p in list(blobs)[:20]:
            print(f'  {"DELETE" if blobs[p] is None else "WRITE "} {p}')
        if len(blobs) > 20:
            print(f'  ... 其余 {len(blobs)-20} 个')
        return

    # 1) 父提交的 tree
    parent_commit = _req('GET', f'/repos/{OWNER}/{REPO}/git/commits/{remote_sha}', token)
    base_tree = parent_commit['tree']['sha']

    # 2) 创建 blobs 与 tree entries
    entries = []
    for i, (path, local) in enumerate(blobs.items(), 1):
        if local is None:
            entries.append({'path': path, 'mode': '100644', 'type': 'blob', 'sha': None})
            continue
        content = base64.b64encode(Path(local).read_bytes()).decode()
        blob = _req('POST', f'/repos/{OWNER}/{REPO}/git/blobs', token,
                    {'content': content, 'encoding': 'base64'})
        entries.append({'path': path, 'mode': '100644', 'type': 'blob', 'sha': blob['sha']})
        if i % 10 == 0:
            print(f'  blob {i}/{len(blobs)}')

    tree = _req('POST', f'/repos/{OWNER}/{REPO}/git/trees', token,
                {'base_tree': base_tree, 'tree': entries})
    print(f'tree 创建完成: {tree["sha"][:8]}')

    # 3) 创建 commit
    msg = _git('log', '-1', '--pretty=%B')
    commit = _req('POST', f'/repos/{OWNER}/{REPO}/git/commits', token,
                  {'message': msg, 'tree': tree['sha'], 'parents': [remote_sha]})
    print(f'commit 创建完成: {commit["sha"][:8]}')

    # 4) 更新 ref
    _req('PATCH', f'/repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}', token,
         {'sha': commit['sha'], 'force': False})
    print(f'\n已推送 → {commit["sha"][:8]}')
    print(f'https://github.com/{OWNER}/{REPO}/commit/{commit["sha"]}')


if __name__ == '__main__':
    sys.exit(main())
