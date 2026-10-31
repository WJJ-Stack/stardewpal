#!/usr/bin/env bash
# 一键构建部署产物：下载模型 → 抓取 → 建索引 → 建图标索引
# 用于生产镜像构建，或本地首次初始化。
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> 1/4 下载嵌入模型"
python -m backend.pipeline.fetch_model

echo "==> 2/4 抓取 wiki（严格限速，首次约 1.5 小时）"
if [ -f backend/data/raw/pages.jsonl ]; then
  echo "    已存在抓取结果，跳过。如需重抓请删除 backend/data/raw/pages.jsonl"
else
  python -m backend.pipeline.crawl
fi

echo "==> 3/4 构建检索索引"
python -m backend.pipeline.build_index --rebuild

echo "==> 4/4 构建图标索引"
python -m backend.pipeline.build_icons

echo
echo "完成。启动服务："
echo "  python -m backend.app.serve"
