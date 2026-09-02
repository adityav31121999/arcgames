"""Automatic file and model weight locator for local and Kaggle environments."""

from pathlib import Path
from typing import List, Optional
import os


def locate_hf_model_dir(
    keyword_tokens: List[str], search_root: str | Path = "/kaggle/input"
) -> Optional[str]:
    """Finds a directory containing config.json and safetensors matching keyword tokens."""
    root_path = Path(search_root)
    if not root_path.exists():
        return None

    # Look for config.json files
    config_files = list(root_path.rglob("config.json"))
    for cfg in config_files:
        model_dir = cfg.parent
        dir_str = str(model_dir).lower()
        if all(tok.lower() in dir_str for tok in keyword_tokens):
            has_weights = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin"))
            if has_weights:
                return str(model_dir)

    # Secondary search on parent directory names
    for cfg in config_files:
        model_dir = cfg.parent
        dir_str = str(model_dir).lower()
        if any(tok.lower() in dir_str for tok in keyword_tokens):
            return str(model_dir)

    return None


def find_arc_agi_wheels(
    search_root: str | Path = "/kaggle/input",
) -> List[Path]:
    """Finds offline arc_agi competition wheels."""
    comp_dir = Path(search_root) / "arc-prize-2026-arc-agi-3" / "arc_agi_3_wheels"
    if comp_dir.exists():
        whls = list(comp_dir.rglob("arc_agi-*.whl"))
        if whls:
            return whls

    root_path = Path(search_root)
    if root_path.exists():
        excluded = "arc-agi-for-offline"
        return [
            w
            for w in root_path.rglob("arc_agi-*.whl")
            if excluded not in str(w)
        ]
    return []
