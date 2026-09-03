"""Configuration models and loader for ARC-AGI-3 Agent."""

from pathlib import Path
from typing import List, Optional
import os
import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    model_id: str = Field(
        default="nvidia/Gemma-4-26B-A4B-NVFP4",
        description="HuggingFace model ID or local directory path",
    )
    fallback_model_ids: List[str] = Field(
        default_factory=lambda: [
            "google/gemma-4-26b-a4b",
            "google/gemma-3-27b-it",
            "Qwen/Qwen2.5-VL-7B-Instruct",
        ]
    )
    device: str = Field(default="cuda:0", description="Target device (e.g., cuda:0)")
    torch_dtype: str = Field(default="bfloat16", description="bfloat16, float16, or float32")
    attn_implementation: str = Field(default="sdpa", description="sdpa or flash_attention_2")
    max_context_length: int = Field(default=8192, description="Maximum context length")
    trust_remote_code: bool = Field(default=True)
    temperature: float = Field(default=0.0)
    top_p: float = Field(default=0.95)
    repeat_penalty: float = Field(default=1.05)
    max_new_tokens_eye: int = Field(default=256)
    max_new_tokens_debug: int = Field(default=96)
    max_new_tokens_brain: int = Field(default=32)
    max_new_tokens_review: int = Field(default=256)


class AgentConfig(BaseModel):
    stuck_threshold: int = Field(default=3, description="Consecutive NOOP steps before breaking")
    consecutive_action_threshold: int = Field(default=5, description="Warning threshold for same action")
    max_iterations_per_level: int = Field(default=5)
    max_levels_per_game: int = Field(default=10)
    speculative_plan_max_steps: int = Field(default=15)
    time_budget_hours: float = Field(default=8.5)
    memory_root: str = Field(default="./agent_memory")
    vision_cache_dir: str = Field(default="/tmp/agent_vision")


class EnvironmentConfig(BaseModel):
    mode: str = Field(default="offline", description="'offline', 'competition', or 'auto'")
    environments_dir: str = Field(
        default="/kaggle/input/arc-prize-2026-arc-agi-3/environment_files"
    )
    base_url: str = Field(default="http://gateway:8001/")
    submission_parquet: str = Field(default="./submission.parquet")


class AppConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def apply_env_overrides(self) -> None:
        """Applies environment variable overrides if present."""
        if os.getenv("MODEL_ID"):
            self.model.model_id = os.environ["MODEL_ID"]
        if os.getenv("DEVICE"):
            self.model.device = os.environ["DEVICE"]
        if os.getenv("TORCH_DTYPE"):
            self.model.torch_dtype = os.environ["TORCH_DTYPE"]
        if os.getenv("ENVIRONMENTS_DIR"):
            self.environment.environments_dir = os.environ["ENVIRONMENTS_DIR"]
        if os.getenv("ARC_BASE_URL"):
            self.environment.base_url = os.environ["ARC_BASE_URL"]
        if os.getenv("OPERATION_MODE"):
            self.environment.mode = os.environ["OPERATION_MODE"].lower()
        if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
            self.environment.mode = "competition"
