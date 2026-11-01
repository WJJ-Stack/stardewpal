"""模型服务连通性自检。

用法
----
    python -m backend.pipeline.check_llm              # 测试当前配置
    python -m backend.pipeline.check_llm --provider zhipu
    python -m backend.pipeline.check_llm --all        # 依次试每个已配置的 provider

各 provider 免费额度入口
------------------------
    zhipu       https://open.bigmodel.cn/           注册即送，glm-4-flash 长期免费
    modelscope  https://modelscope.cn/              绑定阿里云后每日免费调用
    siliconflow https://cloud.siliconflow.cn/       注册送额度
    qwen        https://bailian.console.aliyun.com/ 新用户 100 万 token
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # E:\rag
PROJECT_ROOT = ROOT.parent if ROOT.name == 'backend' else ROOT
sys.path.insert(0, str(ROOT))

PROVIDERS = {
    'zhipu': {
        'env': 'ZHIPU_API_KEY',
        'base_url': 'https://open.bigmodel.cn/api/paas/v4',
        'model': 'glm-4-flash',
        'console': 'https://open.bigmodel.cn/',
    },
    'modelscope': {
        'env': 'MODELSCOPE_API_KEY',
        'base_url': 'https://api-inference.modelscope.cn/v1',
        'model': 'Qwen/Qwen3.5-27B',
        'console': 'https://modelscope.cn/',
    },
    'siliconflow': {
        'env': 'SILICONFLOW_API_KEY',
        'base_url': 'https://api.siliconflow.cn/v1',
        'model': 'Qwen/Qwen3-8B',
        'console': 'https://cloud.siliconflow.cn/',
    },
    'qwen': {
        'env': 'DASHSCOPE_API_KEY',
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'model': 'qwen-plus',
        'console': 'https://bailian.console.aliyun.com/',
    },
}


async def probe(name: str, cfg: dict, key: str) -> bool:
    import httpx

    print(f'\n[{name}] {cfg["base_url"]}')
    print(f'  model = {cfg["model"]}')
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                f'{cfg["base_url"].rstrip("/")}/chat/completions',
                headers={'Authorization': f'Bearer {key}',
                         'Content-Type': 'application/json'},
                json={'model': cfg['model'],
                      'messages': [{'role': 'user', 'content': '只回复：OK'}],
                      'max_tokens': 10},
            )
        if r.status_code == 200:
            data = r.json()
            msg = data['choices'][0]['message']['content']
            usage = data.get('usage', {})
            print(f'  ✅ 可用 | 回复={msg[:30]!r} | tokens={usage.get("total_tokens")}')
            return True
        detail = r.text[:180]
        print(f'  ❌ HTTP {r.status_code} | {detail}')
        if 'quota' in detail.lower() or 'insufficient' in detail.lower():
            print('     额度不足，需充值或换服务')
        elif r.status_code in (401, 403):
            print('     key 无效或无权限')
        return False
    except Exception as e:  # noqa: BLE001
        print(f'  ❌ 连接失败 | {str(e)[:120]}')
        return False


def load_env_file() -> None:
    """把 .env 里的变量读进 os.environ（不覆盖已有值）。"""
    env = ROOT / '.env'
    if not env.exists():
        return
    for line in env.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip())


async def main_async(args) -> None:
    load_env_file()

    if args.all:
        targets = list(PROVIDERS.items())
    elif args.provider:
        cfg = PROVIDERS.get(args.provider)
        if not cfg:
            print(f'未知 provider: {args.provider}，可选：{list(PROVIDERS)}')
            return
        targets = [(args.provider, cfg)]
    else:
        # 用当前 .env 配置
        from backend.app.config import resolve_provider

        cfg = resolve_provider()
        key = cfg['api_key']
        if not key:
            print('当前未配置 API Key（LLM_API_KEY 或 DASHSCOPE_API_KEY）。')
            print('可获取免费额度的服务：')
            for n, c in PROVIDERS.items():
                print(f'  · {n:12s} {c["console"]}  (env: {c["env"]})')
            print('\n配置后重跑：python -m backend.pipeline.check_llm')
            return
        await probe(cfg['name'], cfg, key)
        return

    results = {}
    for name, cfg in targets:
        key = os.environ.get(cfg['env'], '') or os.environ.get('LLM_API_KEY', '')
        if not key:
            print(f'\n[{name}] 跳过：环境变量 {cfg["env"]} 未设置')
            print(f'  申请地址 {cfg["console"]}')
            results[name] = None
            continue
        results[name] = await probe(name, cfg, key)

    print('\n' + '=' * 60)
    ok = [n for n, v in results.items() if v]
    if ok:
        print(f'可用：{ok}')
        print(f'\n切换方式（写入 .env）：')
        print(f'  LLM_PROVIDER={ok[0]}')
        print(f'  LLM_API_KEY=<对应 key>')
    else:
        print('没有可用的 provider，请先申请免费额度')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--provider', help='指定 provider')
    ap.add_argument('--all', action='store_true', help='测试全部已知 provider')
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
