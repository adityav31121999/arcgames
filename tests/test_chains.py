"""Unit tests for Eye perception and Brain planning and review."""

from enum import Enum
import numpy as np
import pytest

from arc_agent.chains.brain import BrainChain
from arc_agent.chains.eye import EyeChain
from arc_agent.chains.prompts import build_system_prompt
from arc_agent.agent.arc_langchain_agent import ARCLangChainAgent
from arc_agent.core.resolver import GameStateResolver
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
    brain_chain = BrainChain(mock_model)

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

    # 3. Test Eye transition analysis
    s1 = ARCState.create("game_test", 1, 1, obs)
    transition = compute_transition(s0, s1, action_sig=ActionSignature("ACTION1"))
    verdict = eye_chain.analyse_visual("game_test", s0, transition, "0 pixels changed")
    assert len(verdict) > 0

    # 4. Brain also reviews failed attempts
    review = brain_chain.review("game_test", 1, 1, s0, s1, cache)
    assert len(review) > 0


def test_dynamic_system_prompt_builder():
    """Verify build_system_prompt only includes allowed actions for a specific game."""
    # 1. Default (no action space specified) should include all standard actions
    default_prompt = build_system_prompt()
    for act in ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]:
        assert act in default_prompt

    # 2. ls20 scenario: only ACTION1..4 allowed
    ls20_actions = [Action.ACTION1, Action.ACTION2]
    custom_prompt = build_system_prompt(ls20_actions)
    assert "ACTION1: Upward movement" in custom_prompt
    assert "ACTION2: Downward movement" in custom_prompt
    # Must NOT contain actions that don't exist for this game
    assert "ACTION5" not in custom_prompt
    assert "ACTION6" not in custom_prompt
    assert "ACTION7" not in custom_prompt
    assert "RESET:" not in custom_prompt
    numeric_prompt = build_system_prompt([1, 2, 3, 4, 5, 6])
    assert "ACTION1: Upward movement" in numeric_prompt
    assert "ACTION6: Click" in numeric_prompt


def test_agent_action_space_propagation(tmp_path):
    """Verify agent.set_action_space propagates dynamic prompts across all chains."""
    mock_model = MockChatModel()
    eye = EyeChain(mock_model)
    brain = BrainChain(mock_model)

    agent = ARCLangChainAgent(
        eye_chain=eye,
        brain_chain=brain,
        resolver=GameStateResolver(),
        memory_root=str(tmp_path / "memory"),
        vision_cache_dir=str(tmp_path / "vision"),
    )

    # Restrict action space to only ACTION1 and ACTION2
    agent.set_action_space([Action.ACTION1, Action.ACTION2])

    for chain in [agent.eye, agent.brain]:
        assert "ACTION1" in chain.system_prompt
        assert "ACTION2" in chain.system_prompt
        assert "ACTION5" not in chain.system_prompt
        assert "ACTION6" not in chain.system_prompt
        assert "ACTION7" not in chain.system_prompt


def test_budget_context_in_chains(tmp_path):
    """Verify that budget context is properly accepted by Brain."""
    mock_model = MockChatModel()
    cache = KnowledgeCache(memory_root=tmp_path)
    brain = BrainChain(mock_model)

    grid = np.zeros((10, 10), dtype=int)
    s0 = ARCState.create("budget_game", 1, 0, DummyObservation(grid))
    transition = compute_transition(s0, s0, action_sig=ActionSignature("ACTION1"))

    # Pass budget context into Brain
    budget_ctx = "You have used 66 of 2328 total allowed moves (Step 5/30 in Try 1/3)."
    
    action_resp = brain.decide_action(
        "budget_game", 1, s0, s0, [Action.ACTION1], "Context", cache, budget_context=budget_ctx
    )
    assert len(action_resp) > 0

