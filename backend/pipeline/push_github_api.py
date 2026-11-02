"""通过 `gh api` 把本地提交推送到 GitHub（绕过 github.com:443 被墙）。

背景
----
本网络环境下：
  - github.com:443       不可达（git push 失败）
  - api.github.com      可达（gh api 正常）

因此用 Git Data API 手动构造提交。相比直接 git push，
多几步但能在受限网络下完成同步。

用法
----
    python -m backend.pipeline.push_github_api --dry-run
    python -m backend.pipeline.push_github_api
    python -m backend.pipeline.push_github_api --remote-sha <sha>   # 指定基点
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

OWNER_REPO = 'WJJ-Stack/stardewpal'
BRANCH = 'master'
MAX_FILE = 20 * 1024 * 1024  # 超过则跳过（API 单文件上限更严）
# 这些大文件已通过其他方式同步，重复写入易触发 HTTP/2 流错误
SKIP_LARGE_MARKERS = ('.onnx', '.zip', '.db')


def gh_api(method: str, path: str, payload: dict | None = None) -> dict:
    cmd = ['gh', 'api', '--method', method, path]
    if payload is not None:
        cmd += ['--input', '-']
    r = subprocess.run(cmd, input=json.dumps(payload) if payload is not None else None,
                       capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise SystemExit(f'{method} {path} 失败: {r.stderr[:400]}')
    out = r.stdout.strip()
    return json.loads(out) if out else {}


def git(*args: str) -> str:
    r = subprocess.run(['git', *args], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise SystemExit(f'git {" ".join(args)} 失败: {r.stderr[:300]}')
    return r.stdout.strip()


def _files_to_sync(remote_sha: str, local_head: str) -> list[list[str]]:
    """远程提交不在本地时，列出需要写入的文件。

    做法：取本地 HEAD 的全量文件，逐个与远程 blob 的 sha 比对，
    只保留内容不同的。这样避免无谓地重写上千个未改动文件。
    """
    remote_tree = gh_api(
        'GET', f'repos/{OWNER_REPO}/git/trees/{remote_sha}?recursive=1')['tree']
    remote_blobs = {item['path']: item['sha'] for item in remote_tree
                    if item['type'] == 'blob'}

    tracked = git('ls-tree', '-r', '--name-only', local_head).splitlines()
    out = []
    for path in tracked:
        local_sha = git('hash-object', path) if Path(path).exists() else None
        if local_sha and remote_blobs.get(path) == local_sha:
            continue   # 内容一致，跳过
        status = 'M' if path in remote_blobs else 'A'
        out.append([status, path])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--remote-sha', default=None)
    args = ap.parse_args()

    local_head = git('rev-parse', 'HEAD')
    if args.remote_sha:
        remote_sha = args.remote_sha
    else:
        ref = gh_api('GET', f'repos/{OWNER_REPO}/git/ref/heads/{BRANCH}')
        remote_sha = ref['object']['sha']

    print(f'远程 {BRANCH}: {remote_sha[:8]}')
    print(f'本地 HEAD  : {local_head[:8]}')
    if remote_sha == local_head:
        print('已同步')
        return

    # 远程提交可能不在本地对象库（例如通过 API 推送过）。
    # 这种情况无法 merge-base，改为「以远程 tree 为基底，覆盖本地差异文件」。
    has_remote_obj = subprocess.run(
        ['git', 'cat-file', '-e', remote_sha], capture_output=True).returncode == 0

    if has_remote_obj:
        base = git('merge-base', remote_sha, local_head)
        files = [line.split('\t') for line in
                 git('diff', '--name-status', base, local_head).splitlines()]
        print(f'分叉点 {base[:8]}，变更 {len(files)} 个文件')
    else:
        # 回退：把本地 HEAD 相对「上一个已知同步点」的改动全部覆盖上去。
        # 用 --files 模式：取本地 HEAD 的全量文件，但只写入与远程不同的部分。
        print('远程提交不在本地对象库，改用「全量对齐」模式')
        files = _files_to_sync(remote_sha, local_head)
        print(f'对齐 {len(files)} 个文件')

    parent = gh_api('GET', f'repos/{OWNER_REPO}/git/commits/{remote_sha}')
    base_tree = parent['tree']['sha']

    entries, skipped = [], []
    for item in files:
        status, path = item[0], item[-1]
        if status == 'D':
            entries.append({'path': path, 'mode': '100644', 'type': 'blob', 'sha': None})
            continue
        f = Path(path)
        if not f.exists():
            continue
        if f.stat().st_size > MAX_FILE or path.endswith(SKIP_LARGE_MARKERS):
            skipped.append((path, f.stat().st_size))
            continue
        if args.dry_run:
            entries.append({'path': path, 'mode': '100644', 'type': 'blob', 'sha': 'DRYRUN'})
            continue
        content = base64.b64encode(f.read_bytes()).decode()
        blob = gh_api('POST', f'repos/{OWNER_REPO}/git/blobs',
                      {'content': content, 'encoding': 'base64'})
        entries.append({'path': path, 'mode': '100644', 'type': 'blob', 'sha': blob['sha']})
        print(f'  blob {path}')

    if skipped:
        print('\n跳过（过大）：')
        for p, s in skipped:
            print(f'  {s/1048576:6.1f} MB  {p}')

    if args.dry_run:
        print(f'\n[dry-run] 将写入 {len(entries)} 项')
        return

    tree = gh_api('POST', f'repos/{OWNER_REPO}/git/trees',
                  {'base_tree': base_tree, 'tree': entries})
    print(f'tree: {tree["sha"][:8]}')

    message = git('log', '-1', '--pretty=%B')
    commit = gh_api('POST', f'repos/{OWNER_REPO}/git/commits',
                    {'message': message, 'tree': tree['sha'], 'parents': [remote_sha]})
    print(f'commit: {commit["sha"][:8]}')

    gh_api('PATCH', f'repos/{OWNER_REPO}/git/refs/heads/{BRANCH}',
           {'sha': commit['sha']})

    # 本地 ref 跟上，避免下次又算一遍
    subprocess.run(['git', 'update-ref', f'refs/remotes/origin/{BRANCH}', commit['sha']],
                   check=False)

    print(f'\n推送成功 → https://github.com/{OWNER_REPO}/commit/{commit["sha"]}')


if __name__ == '__main__':
    sys.exit(main())
