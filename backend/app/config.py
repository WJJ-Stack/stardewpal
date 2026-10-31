"""全局配置。所有可调参数集中在此，便于线上调整与面试讲解。"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- 模型服务 ----
    llm_provider: str = "qwen"          # qwen | mock
    dashscope_api_key: str = ""
    llm_model: str = "qwen-plus"
    vlm_model: str = "qwen-vl-plus"
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_timeout: float = 60.0
    llm_max_retries: int = 3

    # ---- 检索 ----
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_final: int = 6
    rrf_k: int = 60
    min_score: float = 0.0

    # ---- 视觉 ----
    ocr_enabled: bool = True
    icon_match_topk: int = 3
    icon_confidence_high: float = 0.82
    icon_confidence_mid: float = 0.55
    ocr_confidence_low: float = 0.55

    # ---- Agent ----
    max_agent_steps: int = 4
    agent_timeout: float = 30.0
    enable_calculator: bool = True

    # ---- 存储 ----
    db_path: Path = DATA_DIR / "stardewpal.db"
    model_dir: Path = DATA_DIR / "models" / "bge-small-zh-v1.5"
    icons_dir: Path = DATA_DIR / "icons"
    raw_dir: Path = DATA_DIR / "raw"
    runs_dir: Path = DATA_DIR / "runs"

    # ---- 服务 ----
    host: str = "0.0.0.0"
    port: int = 7860
    cors_origins: list[str] = ["*"]
    rate_limit_per_min: int = 30
    max_image_mb: int = 8

    @property
    def has_llm(self) -> bool:
        return bool(self.dashscope_api_key) and self.llm_provider != "mock"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    for p in (s.db_path.parent, s.model_dir, s.icons_dir, s.raw_dir, s.runs_dir):
        os.makedirs(p, exist_ok=True)
    return s


settings = get_settings()
