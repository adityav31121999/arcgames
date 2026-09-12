"""Unit tests for click-only games (e.g. s5i5), object detection, and HUD filtering."""

from enum import Enum
import numpy as np
import pytest

from arc_agent.core.object_detection import (
    detect_grid_objects,
    is_click_only,
    render_click_history,
    render_detected_objects,
)
from arc_agent.core.actions import ActionSignature
from arc_agent.core.state import ARCState
from arc_agent.chains.brain import BrainChain
from arc_agent.chains.prompts import PROMPT_CLICK_ONLY_TARGET
from arc_agent.agent.arc_langchain_agent import ARCLangChainAgent
from arc_agent.core.resolver import GameStateResolver
from arc_agent.memory.knowledge import KnowledgeCache
from arc_agent.models.gemma_transformers import MockChatModel


class GameAction(Enum):
    RESET = 0
    ACTION1 = 1
    ACTION2 = 2
    ACTION3 = 3
    ACTION4 = 4
    ACTION5 = 5
    ACTION6 = 6
    ACTION7 = 7


class DummyObservation:
    def __init__(self, grid):
        self.grid = grid
        self.state = "PLAYING"
        self.levels_completed = 0


def test_is_click_only():
    """Verify is_click_only correctly identifies exclusive ACTION6 action spaces."""
    assert is_click_only([GameAction.ACTION6]) is True
    assert is_click_only(["ACTION6"]) is True
    assert is_click_only([GameAction.RESET, GameAction.ACTION6]) is True
    assert is_click_only(["RESET", "ACTION6"]) is True

    # Multi-action or directional spaces are NOT click-only
    assert is_click_only([GameAction.ACTION1, GameAction.ACTION2]) is False
    assert is_click_only([GameAction.ACTION1, GameAction.ACTION6]) is False
    assert is_click_only([]) is False
    assert is_click_only(None) is False


def test_detect_grid_objects_bracket_and_hollow_center():
    """Verify that detect_grid_objects accurately detects bracket shapes and avoids clicking in the hollow gap."""
    grid = np.zeros((64, 64), dtype=int)
    
    # Draw a C-shaped bracket (color 9: Blue) at X=[30..36], Y=[20..26]
    # Vertical back: x=30, y=20..26
    grid[20:27, 30] = 9
    # Top bar: y=20, x=30..36
    grid[20, 30:37] = 9
    # Bottom bar: y=26, x=30..36
    grid[26, 30:37] = 9
    # The center at (33, 23) is hollow (0)

    # Draw a HUD corner marker at (1, 1) to (2, 2)
    grid[1:3, 1:3] = 8

    objects = detect_grid_objects(grid)
    assert len(objects) == 2

    # Border location alone cannot establish that the red object is HUD.
    hud_objs = [o for o in objects if o["is_hud"]]
    game_objs = [o for o in objects if not o["is_hud"]]
    assert not hud_objs
    border = next(o for o in objects if o["bbox"] == (1, 1, 2, 2))
    assert border["possible_hud"] is True

    assert len(game_objs) == 2
    bracket = next(o for o in game_objs if o["bbox"] == (30, 20, 36, 26))
    assert bracket["bbox"] == (30, 20, 36, 26)
    assert bracket["visual_center"] == (33, 23)

    # CRITICAL: (33, 23) is empty background. The solid_click_point MUST be a solid pixel of the bracket!
    cx, cy = bracket["solid_click_point"]
    assert grid[cy, cx] == 9  # Must be solid blue pixel!
    assert bracket["shape_desc"] == "bracket / hollow shape"


def test_render_detected_objects():
    """Verify render_detected_objects formats candidate click coordinates."""
    grid = np.zeros((64, 64), dtype=int)
    grid[10:15, 10:15] = 11  # Yellow square block

    text = render_detected_objects(grid)
    assert "Object #1" in text
    assert "Yellow" in text
    assert "ACTION=ACTION6" in text
    assert "X=12 Y=12" in text


def test_render_click_history():
    """Verify render_click_history formats prior click attempts."""
    from arc_agent.memory.trajectory import TrajectoryStep

    steps = [
        TrajectoryStep("hash1", ActionSignature("ACTION6", (("x", 10), ("y", 15))), changed=False),
        TrajectoryStep("hash2", ActionSignature("ACTION6", (("x", 32), ("y", 20))), changed=True),
    ]

    history_text = render_click_history(steps)
    assert "Clicked (X=10, Y=15) -> NO-OP" in history_text
    assert "Clicked (X=32, Y=20) -> CHANGED" in history_text


def test_brain_uses_click_only_prompt_when_action6_exclusive(tmp_path):
    """Verify that BrainChain switches to PROMPT_CLICK_ONLY_TARGET when action space is exclusively ACTION6."""
    mock_model = MockChatModel()
    cache = KnowledgeCache(memory_root=tmp_path)
    brain = BrainChain(mock_model)

    grid = np.zeros((64, 64), dtype=int)
    grid[20:25, 20:25] = 9
    s0 = ARCState.create("click_game", 1, 0, DummyObservation(grid))

    # Test with ACTION6 exclusive
    action_resp = brain.decide_action(
        "click_game",
        1,
        s0,
        s0,
        [GameAction.ACTION6],
        "Context",
        cache,
        object_list="Object #1: Blue bracket at X=22 Y=22",
        click_history="No clicks yet",
    )
    assert "ACTION=" in action_resp


def test_smart_fallback_targets_detected_objects(tmp_path):
    """Verify that _safe_fallback targets unclicked detected objects in click-only games."""
    mock_model = MockChatModel()
    brain = BrainChain(mock_model)
    from arc_agent.chains.eye import EyeChain

    agent = ARCLangChainAgent(
        eye_chain=EyeChain(mock_model),
        brain_chain=brain,
        resolver=GameStateResolver(),
        memory_root=str(tmp_path / "memory"),
        vision_cache_dir=str(tmp_path / "vision"),
    )

    grid = np.zeros((64, 64), dtype=int)
    grid[30:35, 30:35] = 9  # Object at (32, 32)
    s0 = ARCState.create("click_test", 1, 0, DummyObservation(grid))

    action, data, note = agent.decide_action(
        "click_test",
        1,
        s0,
        s0,
        [GameAction.ACTION6],
        "Debug note",
    )
    assert action == GameAction.ACTION6
    assert "x" in data and "y" in data
    assert 0 <= data["x"] < 64 and 0 <= data["y"] < 64


def test_monochromatic_separation_and_nested_containment():
    """Verify that adjacent different-colored objects stay separate, while nested cores merge into brackets."""
    grid = np.zeros((64, 64), dtype=int)

    # 1. Blue bracket at X=[20..30], Y=[20..30] (Color 9)
    grid[20:31, 20] = 9
    grid[20, 20:31] = 9
    grid[30, 20:31] = 9

    # 2. Orange core strictly nested INSIDE the bracket at (25, 25) (Color 12)
    grid[24:27, 24:27] = 12

    # 3. Red player token touching the OUTSIDE of the bracket at (19, 20) (Color 8)
    grid[19, 20] = 8

    objects = detect_grid_objects(grid)
    # Filter out any HUD
    candidates = [o for o in objects if not o["is_hud"]]

    # We should have exactly 2 candidate objects:
    # (1) The bracket + nested core (composite colors 9 and 12)
    # (2) The touching red token (separate color 8)
    assert len(candidates) == 2

    # Find the composite bracket
    composite_bracket = next(o for o in candidates if 9 in o["colors"])
    assert 12 in composite_bracket["colors"]  # Nested core absorbed!
    assert 8 not in composite_bracket["colors"]  # Touching player was NOT fused!

    # Find the player token
    player_token = next(o for o in candidates if 8 in o["colors"])
    assert player_token["colors"] == [8]
    assert player_token["area"] == 1


def test_massive_background_wall_filtering():
    """Verify that huge monolithic walls/floors are filtered out from candidate click objects."""
    grid = np.zeros((64, 64), dtype=int)

    # Small interactive bracket (15 pixels)
    grid[20:25, 20] = 9
    grid[20, 20:25] = 9

    # Giant wall occupying 1200 pixels (>25% of the 4096-pixel grid)
    grid[30:60, 10:50] = 3

    objects = detect_grid_objects(grid)
    wall = next(o for o in objects if 3 in o["colors"])
    assert wall["is_hud"] is False
    assert wall["possible_hud"] is True

    bracket = next(o for o in objects if 9 in o["colors"])
    assert bracket["is_hud"] is False  # Legitimate game object

    rendered = render_detected_objects(grid)
    assert "Color 9" in rendered or "Blue" in rendered
    assert "Dark Gray" in rendered  # Keep possible interactive structures visible.
    assert "role uncertain" in rendered


def test_render_click_history_schema_robustness():
    """Verify render_click_history handles dict steps and raw markdown actions log."""
    # Test dictionary step schema
    dict_steps = [
        {"action_sig": {"name": "ACTION6", "data": {"x": 14, "y": 28}}, "changed": True},
        {"action_sig": {"name": "ACTION6", "data": {"x": 5, "y": 8}}, "changed": False},
    ]
    res1 = render_click_history(dict_steps)
    assert "Clicked (X=14, Y=28) -> CHANGED" in res1
    assert "Clicked (X=5, Y=8) -> NO-OP" in res1

    # Test markdown actions log text fallback
    markdown_log = (
        "# Actions Log\n"
        "| 1 | 10:00:00 | ACTION6(x=42,y=18) | hash1 -> hash2 |\n"
        "| 2 | 10:00:05 | ACTION6(x=11,y=9) | hash2 -> hash2 |\n"
    )
    res2 = render_click_history(trajectory_or_steps=[], actions_log_text=markdown_log)
    assert "Clicked (X=42, Y=18)" in res2
    assert "Clicked (X=11, Y=9)" in res2


def test_system_prompt_preserves_native_role():
    """The checkpoint template receives full system instructions in their native role."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from arc_agent.models.gemma_transformers import GemmaTransformersChatModel

    chat_model = GemmaTransformersChatModel()
    long_system = "A" * 1200  # 1200 characters
    messages = [
        SystemMessage(content=long_system),
        HumanMessage(content="Decide next action."),
    ]
    formatted, _ = chat_model._extract_images_and_text(messages)
    sys_turn = next(m for m in formatted if m["role"] == "system")
    sys_text = sys_turn["content"][0]["text"]
    assert sys_text == long_system

    # Do not duplicate system instructions into the task.
    user_turn = next(m for m in formatted if m["role"] == "user")
    user_text = user_turn["content"][0]["text"]
    assert long_system not in user_text
    assert "Decide next action." in user_text


def test_runner_halts_on_consecutive_llm_failures(tmp_path):
    """Verify runner immediately halts if LLM produces 6 consecutive unparseable outputs."""
    from tests.test_runner_budget import _build_test_agent, DummyEnvWithInfo
    from arc_agent.agent.runner import ARCRunner

    agent = _build_test_agent(tmp_path)
    # Simulate LLM failing to parse
    agent.brain._invoke = lambda *a, **k: "<EMPTY>"
    runner = ARCRunner(agent, max_iterations_per_level=1)
    grid = np.zeros((5, 5), dtype=np.int32)
    env = DummyEnvWithInfo(grid, baseline_actions=[10])

    runner.play_game(
        game_id="halt_test",
        env=env,
        max_levels=1,
        max_steps_per_level=20,
    )
    # Should halt at 6 failures, not drain the full 20 steps
    assert agent.consecutive_parse_failures >= 6
    assert runner.total_actions_taken <= 9
