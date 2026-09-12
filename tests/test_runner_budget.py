"""Unit tests for move limit budget and tries (retry iterations) in ARCRunner."""

from enum import Enum
from types import SimpleNamespace
import numpy as np
import pytest

from arc_agent.agent.runner import ARCRunner, get_max_steps_for_level
from arc_agent.agent.arc_langchain_agent import ARCLangChainAgent
from arc_agent.chains.brain import BrainChain
from arc_agent.chains.eye import EyeChain
from arc_agent.core.resolver import GameStateResolver
from arc_agent.models.gemma_transformers import MockChatModel


class Action(Enum):
    RESET = 0
    ACTION1 = 1
    ACTION2 = 2
    ACTION3 = 3

    def is_complex(self):
        return False


class DummyObservation:
    def __init__(self, grid, state="PLAYING", levels_completed=0):
        self.grid = grid
        self.state = state
        self.levels_completed = levels_completed
        self.available_actions = [Action.ACTION1, Action.ACTION2]


class DummyEnvWithInfo:
    def __init__(self, grid, baseline_actions=None):
        self.grid = grid
        self.action_space = [Action.ACTION1, Action.ACTION2]
        if baseline_actions is not None:
            self.environment_info = SimpleNamespace(
                baseline_actions=baseline_actions,
                win_levels=len(baseline_actions),
            )
        else:
            self.environment_info = None

    def reset(self):
        return DummyObservation(self.grid)

    def step(self, action, **kwargs):
        new_grid = self.grid.copy()
        new_grid[0, 0] = (new_grid[0, 0] + 1) % 10
        return DummyObservation(new_grid)


def test_get_max_steps_for_level_multiplier():
    """Verify that get_max_steps_for_level scales by baseline_multiplier (default 3.0x, or custom)."""
    env = DummyEnvWithInfo(np.zeros((10, 10)), baseline_actions=[12, 25, 40])
    
    # Default multiplier (3.0x) so agent is not starved of exploration steps
    assert get_max_steps_for_level(env, 1) == 36
    assert get_max_steps_for_level(env, 2) == 75
    assert get_max_steps_for_level(env, 3) == 120

    # Explicit 1.0x multiplier
    assert get_max_steps_for_level(env, 1, baseline_multiplier=1.0) == 12
    assert get_max_steps_for_level(env, 2, baseline_multiplier=1.0) == 25

    # With remaining_budget cap
    assert get_max_steps_for_level(env, 1, remaining_budget=20) == 20

    # Level out of bounds falls back to dynamic
    obs = DummyObservation(np.zeros((10, 10)))
    steps = get_max_steps_for_level(env, 99, fallback_obs=obs)
    assert steps > 0


def _build_test_agent(tmp_path):
    mock_llm = MockChatModel()
    return ARCLangChainAgent(
        eye_chain=EyeChain(mock_llm),
        brain_chain=BrainChain(mock_llm),
        resolver=GameStateResolver(),
        stuck_threshold=10,
        memory_root=str(tmp_path / "memory"),
        vision_cache_dir=str(tmp_path / "vision"),
    )


def test_runner_budget_exhaustion(tmp_path):
    """Test that runner stops when max_total_actions is reached."""
    agent = _build_test_agent(tmp_path)
    runner = ARCRunner(agent, max_iterations_per_level=2)
    grid = np.zeros((5, 5), dtype=np.int32)
    env = DummyEnvWithInfo(grid, baseline_actions=[10])

    # Budget capped at 5 total actions
    obs = runner.play_game(
        game_id="budget_test",
        env=env,
        max_levels=1,
        max_total_actions=5,
    )
    assert runner.total_actions_taken == 5
    assert runner.should_stop_game() is True


def test_runner_auto_derives_budget(tmp_path):
    """Test that play_game auto-derives budget as sum(baseline_actions) * max_iterations_per_level."""
    agent = _build_test_agent(tmp_path)
    runner = ARCRunner(agent, max_iterations_per_level=3)
    grid = np.zeros((5, 5), dtype=np.int32)
    env = DummyEnvWithInfo(grid, baseline_actions=[4, 6])

    # Without max_total_actions explicitly passed, budget should be (4 + 6) * 3 = 30
    obs = runner.play_game(
        game_id="auto_budget_test",
        env=env,
        max_levels=1,
        max_steps_per_level=2,  # per-try cap
    )
    assert runner._current_game_budget == 30


def test_runner_tries_exhaustion(tmp_path):
    """Test that runner respects max_iterations_per_level (tries/lives per level)."""
    agent = _build_test_agent(tmp_path)
    runner = ARCRunner(agent, max_iterations_per_level=3)
    grid = np.zeros((5, 5), dtype=np.int32)
    env = DummyEnvWithInfo(grid, baseline_actions=[2])

    # Level 1 has max_steps=2, runs out of steps on each try, should execute exactly 3 tries = 6 total actions
    obs = runner.play_game(
        game_id="tries_test",
        env=env,
        max_levels=1,
    )
    # 3 tries * 2 steps = 6 actions total
    assert runner.total_actions_taken == 6


def test_runner_fast_step_eval(tmp_path):
    """Test that fast_step_eval bypasses redundant LLM calls on NO-OP and routine steps."""
    agent = _build_test_agent(tmp_path)
    runner = ARCRunner(agent, max_iterations_per_level=1, fast_step_eval=True)
    grid = np.zeros((5, 5), dtype=np.int32)
    env = DummyEnvWithInfo(grid, baseline_actions=[4])

    obs = runner.play_game(
        game_id="fast_eval_test",
        env=env,
        max_levels=1,
        max_steps_per_level=2,
    )
    assert runner.total_actions_taken == 2
    assert runner.fast_step_eval is True


def test_runner_macro_repeat_execution(tmp_path):
    """Test that runner executes macro repeat actions sequentially without extra LLM calls."""
    agent = _build_test_agent(tmp_path)
    # Mock decide_action to return ACTION1 with repeat=3
    agent.brain.decide_action = lambda *args, **kwargs: "Plan: move up\nACTION=ACTION1 REPEAT=3"
    runner = ARCRunner(agent, max_iterations_per_level=1, fast_step_eval=True, speculative_plan_max_steps=0)
    grid = np.zeros((5, 5), dtype=np.int32)
    env = DummyEnvWithInfo(grid, baseline_actions=[10])

    obs = runner.play_game(
        game_id="macro_repeat_test",
        env=env,
        max_levels=1,
        max_steps_per_level=5,
    )
    # In 1 LLM decision, it executed 3 steps. Then the next decision takes the remaining 2 steps.
    assert runner.total_actions_taken == 5


def test_runner_macro_repeat_stops_on_noop(tmp_path):
    """Test that runner halts macro repeat immediately if an action causes NO-OP (wall collision)."""
    agent = _build_test_agent(tmp_path)
    agent.brain.decide_action = lambda *args, **kwargs: "Plan: move into wall\nACTION=ACTION1 REPEAT=5"
    runner = ARCRunner(agent, max_iterations_per_level=1, fast_step_eval=True, speculative_plan_max_steps=0)

    class StaticEnv(DummyEnvWithInfo):
        def step(self, action, **kwargs):
            # NO-OP: board does not change at all
            return DummyObservation(self.grid)

    grid = np.zeros((5, 5), dtype=np.int32)
    env = StaticEnv(grid, baseline_actions=[10])

    obs = runner.play_game(
        game_id="macro_noop_test",
        env=env,
        max_levels=1,
        max_steps_per_level=10,
    )
    # The first decision requested repeat=5, but because step 1 was NO-OP (grid unchanged),
    # it immediately stopped repeat and returned to closed-loop!
    # Stuck threshold is 3, so it halts after 3 NO-OP decisions = 3 total actions taken (not 15!)
    assert runner.total_actions_taken == agent.stuck_threshold

