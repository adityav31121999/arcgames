"""Unit tests for LangChain perceptual, debugging, planner, and review chains."""

from enum import Enum
import numpy as np
import pytest

from arc_agent.chains.brain import BrainChain
from arc_agent.chains.debugger import DebuggerChain
from arc_agent.chains.eye import EyeChain
from arc_agent.chains.reviewer import ReviewerChain
from arc_agent.core.state import ARCState, compute_transition
from arc_agent.core.actions import ActionSignature
from arc_agent.memory.knowledge import KnowledgeCache
from arc_agent.models.gemma_transformers import MockChatModel


class DummyObservation:
    def __init__(self, grid):
        self.grid = grid
        self.state = "PLAYING"
        self.levels_completed = 0


class Action(Enum):
    ACTION1 = 1
    ACTION2 = 2


def test_langchain_chains_with_mock_model(tmp_path):
    mock_model = MockChatModel()
    cache = KnowledgeCache(memory_root=tmp_path)

    eye_chain = EyeChain(mock_model)
    debugger_chain = DebuggerChain(mock_model)
    brain_chain = BrainChain(mock_model)
    reviewer_chain = ReviewerChain(mock_model)

    grid = np.zeros((12, 12), dtype=int)
    grid[5, 5] = 9
    obs = DummyObservation(grid)
    s0 = ARCState.create("game_test", 1, 0, obs)

    # 1. Test EyeChain assume
    hyp = eye_chain.assume("game_test", 1, s0, cache)
    assert len(hyp) > 0
    assert cache.scratch("game_test") != ""

    # 2. Test BrainChain decide action
    action_resp = brain_chain.decide_action(
        "game_test", 1, s0, s0, [Action.ACTION1, Action.ACTION2], "No warning", cache
    )
    assert "ACTION=" in action_resp

    # 3. Test DebuggerChain validate
    s1 = ARCState.create("game_test", 1, 1, obs)
    transition = compute_transition(s0, s1, action_sig=ActionSignature("ACTION1"))
    verdict = debugger_chain.validate("game_test", 1, s0, transition, "0 pixels changed", "", cache)
    assert len(verdict) > 0

    # 4. Test ReviewerChain
    review = reviewer_chain.review("game_test", 1, 1, s0, s1, cache)
    assert len(review) > 0
