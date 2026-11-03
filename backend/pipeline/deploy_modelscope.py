"""魔搭创空间全自动部署（不依赖 Git 凭据，全走 HTTP API）。

用法
----
    # 前提：已执行 ms login 或 modelscope login
    python -m backend.pipeline.deploy_modelscope --check     # 只看状态
    python -m backend.pipeline.deploy_modelscope --deploy    # 上传并部署
    python -m backend.pipeline.deploy_modelscope --logs      # 看构建日志

为什么写这个脚本
----------------
1. Windows 上 git push 会被 Credential Manager 卡住（非交互环境弹窗）
2. 魔搭创空间的 SDK 类型默认是 gradio，必须显式改成 docker
3. 密钥要通过 secret 注入，不能写进仓库

这些坑踩过一次就够了。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / '.archive' / 'ms-push'
STUDIO_ID = 'wujaijie/stardewpal'
HARDWARE = 'platform/2v-cpu-8g-mem'   # 免费 CPU 2 核 8G

# 运行时注入的环境变量。
# 文本模型与视觉模型**分别**配置——本地实测 qwen3-vl-plus 远强于 glm-4v-flash，
# 所以 VLM 单独走通义。两份 key 都要注入，只注入 LLM_API_KEY 会让截图功能失效。
#
# 这里的值是兜底默认；真实值从本地 .env 读取（见 inject_secrets）。
VARIABLES = {
    'PORT': '7860',
    'LLM_PROVIDER': 'zhipu',
    'LLM_MODEL': 'glm-4-flash',
    'LLM_BASE_URL': 'https://open.bigmodel.cn/api/paas/v4',
    'VLM_PROVIDER': 'qwen',
    'VLM_MODEL': 'qwen3-vl-plus',
    'VLM_BASE_URL': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
}

# 需要以 secret 形式注入的键（不能进仓库）
SECRET_KEYS = ('LLM_API_KEY', 'VLM_API_KEY')


def _api():
    from modelscope.hub.api import HubApi

    return HubApi()


def _read_local_env() -> dict[str, str]:
    """读取本地 .env（键值对）。缺失返回空字典。"""
    env = ROOT / '.env'
    if not env.exists():
        return {}
    out: dict[str, str] = {}
    for line in env.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        out[k.strip()] = v.strip()
    return out


def show_status() -> None:
    api = _api()
    info = api._api_for({}).get_repo(STUDIO_ID, repo_type='studio')
    print(f'空间      : {info.id}')
    print(f'名称      : {info.display_name}')
    print(f'SDK 类型  : {info.sdk_type}')
    print(f'硬件      : {info.hardware}')
    print(f'许可证    : {info.license}')
    print(f'运行状态  : {(info.runtime or {}).get("status")}')
    print(f'访问地址  : https://modelscope.cn/studios/{STUDIO_ID}')


def ensure_settings() -> None:
    """确保是 Docker 类型 + 免费 CPU。"""
    api = _api()
    inner = api._api_for({})
    info = inner.get_repo(STUDIO_ID, repo_type='studio')

    if info.sdk_type != 'docker':
        print(f'SDK 类型 {info.sdk_type} → docker')
        inner.update_repo_settings(STUDIO_ID, 'studio', sdk_type='docker')
    else:
        print('SDK 类型已是 docker')

    if info.hardware != HARDWARE:
        print(f'硬件 {info.hardware} → {HARDWARE}')
        inner.update_repo_settings(STUDIO_ID, 'studio', hardware=HARDWARE)
    else:
        print(f'硬件已是 {HARDWARE}')


def inject_secrets() -> None:
    """注入 API Key（secret）与普通配置（variable）。

    secret 与 variable 要分开处理：
      · API Key 走 secret（加密存储，日志里不可见）
      · provider/model/base_url 走 variable（普通环境变量）
    两份 key（文本 + 视觉）都要注入，只给文本 key 会导致截图问答静默降级。
    """
    api = _api()
    local = _read_local_env()

    # 1) secrets
    for k in SECRET_KEYS:
        val = local.get(k)
        if not val:
            print(f'未找到本地 {k}，跳过')
            continue
        try:
            api.add_studio_secret(STUDIO_ID, k, val)
            print(f'secret {k} 已设置')
        except Exception as e:  # noqa: BLE001
            try:
                api.update_studio_secret(STUDIO_ID, k, val)
                print(f'secret {k} 已更新')
            except Exception as e2:  # noqa: BLE001
                print(f'{k} 设置失败: {str(e)[:80]} / {str(e2)[:80]}')

    # 2) variables（本地 .env 优先，缺失用默认）
    variables = dict(VARIABLES)
    for k in list(variables):
        if local.get(k):
            variables[k] = local[k]

    for k, v in variables.items():
        try:
            api.add_studio_variable(STUDIO_ID, k, v)
        except Exception:  # noqa: BLE001
            try:
                api.update_studio_variable(STUDIO_ID, k, v)
            except Exception:  # noqa: BLE001
                pass
    print(f'variables 已设置: {sorted(variables)}')


def upload() -> None:
    if not SRC.exists():
        raise SystemExit(f'源目录不存在: {SRC}，先运行 prepare_modelscope')
    api = _api()
    print(f'上传 {SRC} → {STUDIO_ID} ...')
    t0 = time.time()
    api.upload_folder(
        repo_id=STUDIO_ID,
        folder_path=str(SRC),
        repo_type='studio',
        commit_message='deploy: StardewPal',
        ignore_patterns=['__pycache__/*', '*.pyc', '.pytest_cache/*', '*.log',
                         '*.db-shm', '*.db-wal'],
        max_workers=8,
    )
    print(f'上传完成，用时 {time.time()-t0:.0f}s')


def deploy() -> None:
    api = _api()
    r = api.deploy_studio(STUDIO_ID)
    print(f'部署已触发: {json.dumps(r, ensure_ascii=False, default=str)}')


def watch_logs(interval: int = 20, max_minutes: int = 15) -> None:
    api = _api()
    inner = api._api_for({})
    deadline = time.time() + max_minutes * 60
    last_len = 0
    while time.time() < deadline:
        try:
            info = inner.get_repo(STUDIO_ID, repo_type='studio')
            st = (info.runtime or {}).get('status')
            logs = api.get_studio_logs(STUDIO_ID)
            items = logs.get('logs') or []
            if len(items) > last_len:
                for it in items[last_len:]:
                    line = it.get('message') if isinstance(it, dict) else str(it)
                    print(f'  {line}')
                last_len = len(items)
            print(f'[{time.strftime("%H:%M:%S")}] status={st} logs={len(items)}', flush=True)
            if st in ('Running', 'Success'):
                print('\n构建成功')
                return
            if st in ('BuildFailed', 'Failed', 'RunningFailed'):
                print('\n构建失败，查看上方日志')
                return
        except Exception as e:  # noqa: BLE001
            print(f'  查询出错: {str(e)[:120]}')
        time.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='查看当前状态')
    ap.add_argument('--deploy', action='store_true', help='上传 + 配置 + 部署')
    ap.add_argument('--logs', action='store_true', help='跟踪构建日志')
    ap.add_argument('--no-upload', action='store_true', help='只部署不上传')
    ap.add_argument('--src', default=None, help='上传源目录（默认 .archive/ms-push）')
    ap.add_argument('--skip-watch', action='store_true', help='触发部署后不跟踪日志')
    args = ap.parse_args()

    if args.check or not any([args.deploy, args.logs]):
        show_status()
        return

    global SRC
    if args.src:
        SRC = Path(args.src)

    if args.deploy:
        ensure_settings()
        inject_secrets()
        if not args.no_upload:
            upload()
        deploy()
        if not args.skip_watch:
            watch_logs()
    elif args.logs:
        watch_logs()


if __name__ == '__main__':
    sys.exit(main())
