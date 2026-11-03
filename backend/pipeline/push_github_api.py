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
import time
from pathlib import Path

OWNER_REPO = 'WJJ-Stack/stardewpal'
BRANCH = 'master'
MAX_FILE = 20 * 1024 * 1024  # 普通 blob 上限
# .gitattributes 里声明走 LFS 的后缀。这些文件不能当普通 blob 推，
# 否则远程拿到的是真实字节而非 LFS 指针，与 .gitattributes 冲突。
LFS_MARKERS = ('.onnx', '.zip', '.db')
LFS_BATCH = 'https://github.com/{repo}.git/info/lfs/objects/batch'


LFS_CACHE = Path(__file__).resolve().parents[2] / '.archive' / 'lfs_cache.json'


def _load_cache() -> dict:
    try:
        return json.loads(LFS_CACHE.read_text(encoding='utf-8'))
    except Exception:  # noqa: BLE001
        return {}


def _save_cache(cache: dict) -> None:
    try:
        LFS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        LFS_CACHE.write_text(json.dumps(cache, indent=1), encoding='utf-8')
    except Exception:  # noqa: BLE001
        pass


def gh_api(method: str, path: str, payload: dict | None = None) -> dict:
    cmd = ['gh', 'api', '--method', method, path]
    if payload is not None:
        cmd += ['--input', '-']
    body = json.dumps(payload, ensure_ascii=False) if payload is not None else None
    # 同样必须显式 utf-8：payload 里有中文提交信息，
    # 不指定编码时 Windows 按 GBK 处理 stdin/stdout 会出错。
    r = subprocess.run(cmd, input=body, capture_output=True, text=True,
                       encoding='utf-8', errors='replace', check=False)
    if r.returncode != 0:
        raise SystemExit(f'{method} {path} 失败: {(r.stderr or "")[:400]}')
    out = (r.stdout or '').strip()
    return json.loads(out) if out else {}


def _lfs_batch(repo: str, objects: list[dict], attempts: int = 4) -> dict:
    """POST /lfs/objects/batch，返回各文件的上传地址。

    github.com:443 在本网络很不稳（偶发 wsarecv 超时），
    失败就退避重试；连续失败才真报错。
    """
    url = LFS_BATCH.format(repo=repo)
    payload = {
        'operation': 'upload',
        'transfers': ['basic'],
        'objects': objects,
    }
    last_err = ''
    for i in range(attempts):
        r = subprocess.run(
            ['gh', 'api', '--method', 'POST', url,
             '-H', 'Accept: application/vnd.git-lfs+json',
             '-H', 'Content-Type: application/vnd.git-lfs+json',
             '--input', '-'],
            input=json.dumps(payload), capture_output=True, text=True, check=False)
        if r.returncode == 0 and r.stdout.strip():
            return json.loads(r.stdout)
        last_err = (r.stderr or '')[:300]
        if i < attempts - 1:
            wait = 3 * (2 ** i)
            print(f'    LFS batch 重试 {i+1}/{attempts-1}（{wait}s 后）')
            time.sleep(wait)
    raise SystemExit(f'LFS batch 失败（重试 {attempts} 次）: {last_err}')


def lfs_upload(repo: str, path: str, use_cache: bool = True) -> tuple:
    """把单个文件按 Git LFS 协议上传，返回 (oid, size)。

    带本地缓存：已成功上传过的文件直接复用 oid，不再联网。
    github.com 在本网络不稳，这一步能省掉绝大部分重试成本。
    """
    raw = Path(path).read_bytes()
    oid = __import__('hashlib').sha256(raw).hexdigest()
    size = len(raw)

    if use_cache:
        hit = _load_cache().get(path)
        if hit and hit.get('oid') == oid and hit.get('size') == size:
            print('    (缓存命中，跳过上传)')
            return oid, size

    resp = _lfs_batch(repo, [{'oid': oid, 'size': size}])
    obj = (resp.get('objects') or [{}])[0]
    actions = obj.get('actions') or {}
    upload = actions.get('upload')

    if upload:
        # upload.href 是预签名的对象存储地址，不需要再附 token，
        # 只需要服务端在 batch 响应里给出的 header（通常是 Content-Type）。
        import urllib.request

        req = urllib.request.Request(upload['href'], data=raw, method='PUT')
        for k, v in (upload.get('header') or {}).items():
            req.add_header(k, v)
        if not (upload.get('header') or {}):
            req.add_header('Content-Type', 'application/octet-stream')
        with urllib.request.urlopen(req, timeout=600) as r:
            if r.status not in (200, 201, 204):
                raise SystemExit(f'LFS PUT 失败 {path}: HTTP {r.status}')

    if use_cache:
        cache = _load_cache()
        cache[path] = {'oid': oid, 'size': size}
        _save_cache(cache)

    # 验证（部分服务端要求）
    verify = actions.get('verify')
    if verify:
        import urllib.request

        body = json.dumps({'oid': oid, 'size': size}).encode()
        req = urllib.request.Request(verify['href'], data=body, method='POST')
        for k, v in (verify.get('header') or {}).items():
            req.add_header(k, v)
        try:
            urllib.request.urlopen(req, timeout=120)
        except Exception:
            pass

    return oid, size


def _lfs_pointer(oid: str, size: int) -> bytes:
    """LFS 指针文件的字节内容。git checkout 时由 smudge filter 替换为真身。"""
    return (b'version https://git-lfs.github.com/spec/v1\n'
            b'oid sha256:' + oid.encode() + b'\n'
            b'size ' + str(size).encode() + b'\n')


def git(*args: str) -> str:
    """执行 git 命令并返回 stdout。

    必须显式指定 encoding='utf-8'：
    Windows 默认按 GBK 解码，提交信息里一有中文/emoji 就抛
    UnicodeDecodeError，`r.stdout` 变成 None，
    后续 `git log -1 --pretty=%B` 拿到 None，
    在 commit 那步静默失败——排查了很久才发现是这个。
    """
    r = subprocess.run(['git', *args], capture_output=True, text=True,
                       encoding='utf-8', errors='replace', check=False)
    if r.returncode != 0:
        raise SystemExit(f'git {" ".join(args)} 失败: {(r.stderr or "")[:300]}')
    return (r.stdout or '').strip()


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

        # LFS 文件：走 LFS 协议上传，tree 里用 LFS oid 作 blob sha
        if path.endswith(LFS_MARKERS):
            if args.dry_run:
                entries.append({'path': path, 'mode': '100644', 'type': 'blob',
                                'sha': 'DRYRUN-LFS'})
                continue
            oid, size = lfs_upload(OWNER_REPO, path)
            # 提交的是指针文本（普通 blob），不是 LFS oid。
            # trees API 只认已存在的 git 对象，传 oid 会 422。
            ptr = base64.b64encode(_lfs_pointer(oid, size)).decode()
            pblob = gh_api('POST', f'repos/{OWNER_REPO}/git/blobs',
                           {'content': ptr, 'encoding': 'base64'})
            entries.append({'path': path, 'mode': '100644', 'type': 'blob',
                            'sha': pblob['sha']})
            print(f'  lfs  {path}  ({size/1048576:.1f} MB)')
            continue

        if f.stat().st_size > MAX_FILE:
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
