"""Unit tests for VisualWorldModelIndex."""

from pathlib import Path
import numpy as np
from PIL import Image

from arc_agent.memory.visual_index import VisualWorldModelIndex, build_compare_assume_payload


def test_visual_world_model_index_persists_s0_and_world_model(tmp_path):
    index = VisualWorldModelIndex(agent_memory_dir=str(tmp_path))

    # Save level 0 S0
    img_l0 = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
    p0 = index.save_level_s0(0, img_l0)
    assert Path(p0).exists()

    # Save level 0 world model
    wm_text_0 = "World model: maze with player and exit.\nGoal model: reach exit."
    index.save_world_model_text(0, wm_text_0)

    # Check index persistence across instances
    index_reloaded = VisualWorldModelIndex(agent_memory_dir=str(tmp_path))
    prior = index_reloaded.get_prior_level(current_level=1)
    assert prior is not None
    assert prior["image"] == p0
    assert prior["world_model"] == wm_text_0

    # Build compare_assume payload for level 1
    img_l1 = Image.fromarray(np.ones((10, 10, 3), dtype=np.uint8) * 255)
    payload = build_compare_assume_payload(index_reloaded, current_level=1, current_s0_image=img_l1)

    assert len(payload["images"]) == 2
    assert payload["images"][0][0] == "Prior level S0"
    assert payload["images"][1][0] == "Current level S0"
    assert payload["prior_world_model_text"] == wm_text_0
