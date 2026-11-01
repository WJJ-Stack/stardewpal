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
        env_file=str(ROOT.parent / ".env"),  # 兼容从仓库根目录运行
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- 模型服务 ----
    # provider 决定预设的 base_url / 默认模型；显式设置 LLM_BASE_URL 可覆盖
    #   qwen   | 阿里云百炼 DashScope
    #   zhipu  | 智谱 GLM（GLM-4-Flash 长期免费）
    #   modelscope | 魔搭免费推理
    #   siliconflow | 硅基流动
    #   mock   | 完全离线，用于 CI 与链路验证
    llm_provider: str = "qwen"
    llm_api_key: str = ""              # 通用 key（优先）
    dashscope_api_key: str = ""        # 兼容旧配置
    llm_model: str = ""
    vlm_model: str = ""
    llm_base_url: str = ""             # 留空则用 provider 预设
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
    def api_key(self) -> str:
        """统一取 key：LLM_API_KEY 优先，兼容旧的 DASHSCOPE_API_KEY。"""
        return self.llm_api_key or self.dashscope_api_key

    @property
    def has_llm(self) -> bool:
        return bool(self.api_key) and self.llm_provider != "mock"


# provider 预设：base_url 与默认模型（均兼容 OpenAI 协议）
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "vlm_model": "qwen-vl-plus",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
        "vlm_model": "glm-4v-flash",
    },
    "modelscope": {
        "base_url": "https://api-inference.modelscope.cn/v1",
        "model": "Qwen/Qwen3.5-27B",
        "vlm_model": "Qwen/Qwen3.5-VL-30B-A3B-Instruct",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen3-8B",
        "vlm_model": "Qwen/Qwen2.5-VL-7B-Instruct",
    },
    "mock": {
        "base_url": "http://127.0.0.1:1",
        "model": "mock",
        "vlm_model": "mock",
    },
}


def resolve_provider() -> dict[str, str]:
    """合并 provider 预设与显式配置。"""
    name = (settings.llm_provider or "qwen").strip().lower()
    preset = PROVIDER_PRESETS.get(name, PROVIDER_PRESETS["qwen"])
    return {
        "name": name,
        "base_url": (settings.llm_base_url or preset["base_url"]).rstrip("/"),
        "model": settings.llm_model or preset["model"],
        "vlm_model": settings.vlm_model or preset["vlm_model"],
        "api_key": settings.api_key,
    }


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    for p in (s.db_path.parent, s.model_dir, s.icons_dir, s.raw_dir, s.runs_dir):
        os.makedirs(p, exist_ok=True)
    return s


settings = get_settings()
