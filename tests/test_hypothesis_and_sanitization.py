"""Unit tests for hypothesis creation, non-English token sanitization, and hash-free context."""

from enum import Enum
import numpy as np
import pytest

from arc_agent.chains.brain import BrainChain
from arc_agent.chains.debugger import DebuggerChain
from arc_agent.chains.eye import EyeChain
from arc_agent.core.actions import ActionSignature
from arc_agent.core.state import ARCState, compute_transition
from arc_agent.memory.knowledge import KnowledgeCache
from arc_agent.memory.trajectory import TrajectoryMemory
from arc_agent.memory.world_model import WorldModel
from arc_agent.models.gemma_transformers import MockChatModel, _sanitize_llm_text


class Action(Enum):
    ACTION1 = 1
    ACTION2 = 2
    ACTION6 = 6


class DummyObservation:
    def __init__(self, grid):
        self.grid = grid
        self.state = "PLAYING"
        self.levels_completed = 0


def test_sanitize_llm_text_removes_foreign_scripts():
    """Verify that non-English/CJK/Cyrillic scripts and repetitive loops are cleanly sanitized."""
    # 1. Foreign script flooding (Chinese / Japanese / Russian)
    cjk_text = "World model: 这是一个测试。 Goal model: 达到目标。"
    sanitized_cjk = _sanitize_llm_text(cjk_text)
    assert "这是一个测试" not in sanitized_cjk
    assert "World model:" in sanitized_cjk
    assert "Goal model:" in sanitized_cjk

    # 2. Degenerate token repetition loop
    loop_text = "ACTION=ACTION1 ACTION=ACTION1 ACTION=ACTION1 ACTION=ACTION1 ACTION=ACTION1"
    sanitized_loop = _sanitize_llm_text(loop_text)
    words = sanitized_loop.split()
    assert len(words) <= 12

    # 3. Clean English text passes through untouched
    clean_text = "World model: 2D grid with blue player.\nGoal model: Reach green tile."
    assert _sanitize_llm_text(clean_text) == clean_text


def test_hypothesis_creates_structured_world_model(tmp_path):
    """Verify that EyeChain.assume produces structured labeled output that updates WorldModel."""
    mock_model = MockChatModel()
    cache = KnowledgeCache(memory_root=tmp_path)
    eye = EyeChain(mock_model)
    wm = WorldModel()

    grid = np.zeros((10, 10), dtype=int)
    grid[2, 3] = 9
    s0 = ARCState.create("test_game", 1, 0, DummyObservation(grid))

    hyp = eye.assume("test_game", 1, s0, cache)
    assert "World model:" in hyp
    assert "Goal model:" in hyp
    assert "Action model:" in hyp

    wm.update_from_text(hyp)
    assert not wm.is_empty()
    assert "2D grid puzzle" in wm.world_model
    assert "green goal block" in wm.goal_model
    assert "ACTION1" in wm.action_model


def test_no_raw_hashes_in_context(tmp_path):
    """Verify that raw hashes (e.g. SHA-256 strings) do not appear in LLM context or logs."""
    cache = KnowledgeCache(memory_root=tmp_path)
    mem = TrajectoryMemory()
    mem.reset("hash_abc123456789def")

    # 1. Trajectory text must not contain raw hash
    sig = ActionSignature.from_action(Action.ACTION1)
    mem.record_transition("hash_abc123456789def", sig, "hash_xyz987654321fed", changed=True)
    traj_text = mem.recent_trajectory_text()
    assert "hash_abc" not in traj_text
    assert "hash_xyz" not in traj_text
    assert "ACTION1" in traj_text

    # 2. Actions log line must not contain raw hash
    from arc_agent.agent.arc_langchain_agent import ARCLangChainAgent
    from arc_agent.core.resolver import GameStateResolver
    mock_model = MockChatModel()
    agent = ARCLangChainAgent(
        eye_chain=EyeChain(mock_model),
        debugger_chain=DebuggerChain(mock_model),
        brain_chain=BrainChain(mock_model),
        reviewer_chain=None,
        resolver=GameStateResolver(),
        memory_root=str(tmp_path / "memory"),
    )

    agent.log_action("game_test", 1, 1, sig, "hash_before_12345", "hash_after_67890")
    log_content = agent.cache.actions_log("game_test", 1)
    assert "hash_before" not in log_content
    assert "hash_after" not in log_content
    assert "Changed" in log_content
