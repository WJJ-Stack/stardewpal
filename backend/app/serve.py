"""启动入口：uvicorn 封装，便于本地与容器统一调用。

    python -m backend.app.serve          # 默认 0.0.0.0:7860
    python -m backend.app.serve --port 8000
"""
from __future__ import annotations

import argparse

import uvicorn

from backend.app.config import settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default=settings.host)
    ap.add_argument('--port', type=int, default=settings.port)
    ap.add_argument('--reload', action='store_true')
    args = ap.parse_args()

    uvicorn.run(
        'backend.app.main:app',
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level='info',
        # 魔搭创空间的反向代理需要 forwarded headers 才能正确识别
        proxy_headers=True,
        forwarded_allow_ips='*',
    )


if __name__ == '__main__':
    main()
