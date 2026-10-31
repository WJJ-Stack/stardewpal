# StardewPal 数据准备（Windows）
# 用法：powershell -ExecutionPolicy Bypass -File scripts\prepare_data.ps1

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "==> 1/4 下载嵌入模型" -ForegroundColor Cyan
python -m backend.pipeline.fetch_model

Write-Host "==> 2/4 抓取 wiki" -ForegroundColor Cyan
if (Test-Path "backend\data\raw\pages.jsonl") {
    Write-Host "    已存在抓取结果，跳过"
} else {
    python -m backend.pipeline.crawl
}

Write-Host "==> 3/4 构建检索索引" -ForegroundColor Cyan
python -m backend.pipeline.build_index --rebuild

Write-Host "==> 4/4 构建图标索引" -ForegroundColor Cyan
python -m backend.pipeline.build_icons

Write-Host ""
Write-Host "完成。启动：python -m backend.app.serve" -ForegroundColor Green
