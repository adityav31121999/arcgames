"""
VisualWorldModelIndex: persists each level's S0 screenshot alongside the text
hypothesis produced by compare_assume, so future levels can be visually
grounded against a real prior image instead of a purely textual memory.
"""

import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image


LEVEL_STATES_DIR = "level_states"
INDEX_FILE = "index.json"


@dataclass
class VisualWorldModelIndex:
    agent_memory_dir: str
    _index: Dict[str, dict] = field(default_factory=dict)

    def __post_init__(self):
        os.makedirs(self.states_dir, exist_ok=True)
        self._index = self._load_index()

    @property
    def states_dir(self) -> str:
        return os.path.join(self.agent_memory_dir, LEVEL_STATES_DIR)

    @property
    def index_path(self) -> str:
        return os.path.join(self.states_dir, INDEX_FILE)

    def _load_index(self) -> dict:
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_index(self) -> None:
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2)

    def save_level_s0(self, level: int, image_data: Any) -> str:
        """Persist the raw S0 screenshot for a level.
        image_data can be raw PNG/JPEG bytes or a PIL.Image.Image.
        """
        path = os.path.join(self.states_dir, f"level_{level}_s0.png")
        if isinstance(image_data, bytes):
            with open(path, "wb") as f:
                f.write(image_data)
        elif hasattr(image_data, "save"):
            image_data.save(path, format="PNG")
        else:
            raise TypeError(f"Unsupported image_data type: {type(image_data)}")

        entry = self._index.get(str(level), {})
        entry["image"] = path
        self._index[str(level)] = entry
        self._save_index()
        return path

    def save_world_model_text(self, level: int, world_model_text: str) -> None:
        """Persist the parsed 'World model:' section produced by
        compare_assume for this level, linked to its S0 image."""
        entry = self._index.get(str(level), {})
        entry["world_model"] = world_model_text
        self._index[str(level)] = entry
        self._save_index()

    def get_prior_level(self, current_level: int) -> Optional[dict]:
        """Return the most recently completed level's {image, world_model},
        if any, for use as the comparison anchor in compare_assume."""
        prior = current_level - 1
        return self._index.get(str(prior))

    def get_level(self, level: int) -> Optional[dict]:
        return self._index.get(str(level))

    def get_prior_pil_image(self, current_level: int) -> Optional[Image.Image]:
        """Loads and returns the prior level's S0 PIL image if available."""
        prior = self.get_prior_level(current_level)
        if prior and prior.get("image") and os.path.exists(prior["image"]):
            try:
                return Image.open(prior["image"]).convert("RGB")
            except Exception:
                return None
        return None


def build_compare_assume_payload(
    index: VisualWorldModelIndex,
    current_level: int,
    current_s0_image: Any,
) -> dict:
    """
    Assemble the multimodal payload for compare_assume: current level's S0
    image plus the single most recent prior level's S0 image + its stored
    world-model text. Capped to one prior level to control latency/cost.
    """
    prior = index.get_prior_level(current_level)
    prior_image = index.get_prior_pil_image(current_level)
    prior_world_model_text = prior.get("world_model") if prior else None

    images = []
    if prior_image is not None:
        images.append(("Prior level S0", prior_image))
    if current_s0_image is not None:
        images.append(("Current level S0", current_s0_image))

    return {
        "images": images,
        "prior_world_model_text": prior_world_model_text,
    }
