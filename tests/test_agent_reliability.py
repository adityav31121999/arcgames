"""Regression tests for progression, visual evidence, and stage recovery."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from arc_agent.agent.runner import ARCRunner
from arc_agent.chains.eye import EyeChain
from arc_agent.chains.inference import StageResult
from arc_agent.core.actions import ActionSignature
from arc_agent.core.diff import detect_real_change, get_hud_pixels, register_hud_pixel
from arc_agent.core.object_detection import render_detected_objects
from arc_agent.core.state import ARCState, compute_transition
from arc_agent.memory.knowledge import KnowledgeCache
from tests.test_runner_budget import _build_test_agent


class Environment:
    action_space = ["ACTION1"]

    def __init__(self, complete=False, changing=False):
        self.ticks = self.resets = self.completed = 0
        self.complete, self.changing = complete, changing

    def obs(self):
        grid = np.zeros((8, 8), dtype=int)
        if self.changing:
            grid[4, 4] = self.ticks % 15
        return SimpleNamespace(grid=grid, state="PLAYING", levels_completed=self.completed,
                               available_actions=self.action_space, ticks=self.ticks)

    def reset(self):
        self.resets += 1
        return self.obs()

    def step(self, action, **kwargs):
        self.ticks += 1
        self.completed += int(self.complete)
        return self.obs()


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setattr("arc_agent.agent.runner.render_live", lambda *a, **k: None)
    monkeypatch.setattr("arc_agent.agent.arc_langchain_agent.render_live", lambda *a, **k: None)
    monkeypatch.setattr("arc_agent.agent.runner.is_time_budget_exhausted", lambda *a: False)
    monkeypatch.delenv("FAST_STEP_EVAL", raising=False)
    agent = _build_test_agent(tmp_path)
    agent.stuck_threshold = 3
    agent.brain.one_shot_plan = Mock(return_value="")
    return agent


def test_completed_count_advances_levels_without_false_review_or_reset(agent):
    env = Environment(complete=True)
    agent.review_failed_iteration = Mock()
    runner = ARCRunner(agent, max_iterations_per_level=2, fast_step_eval=True)
    final = runner.play_game("progress", env, obs=env.obs(), max_levels=2, max_steps_per_level=2)
    assert final.levels_completed == 2
    assert env.ticks == 2
    assert env.resets == 0
    agent.review_failed_iteration.assert_not_called()


def test_speculative_stuck_handoff_returns_last_observation(agent):
    env = Environment()
    agent.brain.one_shot_plan.return_value = "ACTION=ACTION1\n" * 3
    runner = ARCRunner(agent)
    initial = agent.enter_level("handoff", 1, env.obs(), True, env.action_space)
    current, previous, _, used, solved = runner.attempt_one_shot(
        "handoff", 1, env, initial, env.action_space, 10,
    )
    assert used == env.ticks == current.raw_obs.ticks == 3
    assert previous.raw_obs.ticks == 2
    assert not solved


def test_speculative_handoff_evaluates_before_brain_replans(agent):
    env = Environment(changing=True)
    agent.brain.one_shot_plan.return_value = "ACTION=ACTION1\n" * 10
    runner = ARCRunner(agent, fast_step_eval=True, full_eval_interval=2,
                       max_iterations_per_level=1)
    events = Mock()
    events.attach_mock(Mock(wraps=agent.eye.analyse_visual), "vision")
    events.attach_mock(Mock(wraps=agent.decide_action), "decide")
    agent.eye.analyse_visual = events.vision
    agent.decide_action = events.decide
    current, _, steps = runner.play_level("handoff_eval", 1, env, env.obs(), env.action_space,
                                         max_steps=3, is_first_level_of_game=True)
    assert steps == current.ticks == 3
    assert [call[0] for call in events.mock_calls[:2]] == ["vision", "decide"]
    assert not hasattr(agent, "debugger") and not hasattr(agent, "reviewer")
    transition = events.vision.call_args.args[2]
    assert transition.previous.raw_obs.ticks == 1
    assert transition.current.raw_obs.ticks == 2


def test_speculation_can_be_disabled_without_model_call(agent):
    env = Environment()
    runner = ARCRunner(agent, speculative_plan_max_steps=0)
    initial = agent.enter_level("no_speculation", 1, env.obs(), True, env.action_space)
    current, previous, _, used, solved = runner.attempt_one_shot(
        "no_speculation", 1, env, initial, env.action_space, 10,
    )
    assert current is initial and previous is None
    assert used == env.ticks == 0 and not solved
    agent.brain.one_shot_plan.assert_not_called()


def test_speculation_hands_off_when_visual_comparison_is_unavailable(agent):
    class MissingBoard(Environment):
        def obs(self):
            obs = super().obs()
            obs.grid = None
            return obs

    env = MissingBoard()
    agent.brain.one_shot_plan.return_value = "ACTION=ACTION1\n" * 5
    runner = ARCRunner(agent)
    initial = agent.enter_level("missing_board", 1, env.obs(), True, env.action_space)
    current, previous, _, used, solved = runner.attempt_one_shot(
        "missing_board", 1, env, initial, env.action_space, 10,
    )
    assert used == current.raw_obs.ticks == 1
    assert previous is initial and not solved


def test_retry_counts_all_steps_and_uses_prior_analysis(agent):
    env = Environment(changing=True)
    runner = ARCRunner(agent, fast_step_eval=True, max_iterations_per_level=2)
    agent.review_failed_iteration = Mock()
    agent.eye.assume = Mock(wraps=agent.eye.assume)
    agent.eye.compare_assume = Mock(wraps=agent.eye.compare_assume)
    current, _, steps = runner.play_level("retry", 1, env, env.obs(), env.action_space,
                                         max_steps=2, is_first_level_of_game=True)
    assert steps == env.ticks == current.ticks == 4
    assert env.resets == 1
    agent.eye.assume.assert_called_once()
    agent.eye.compare_assume.assert_called_once()
    assert agent.review_failed_iteration.call_count == 2


def test_fast_mode_reassesses_stuck_and_passes_action_intent(agent):
    env = Environment()
    runner = ARCRunner(agent, fast_step_eval=True, max_iterations_per_level=1)
    agent.eye.analyse_visual = Mock(wraps=agent.eye.analyse_visual)
    runner.play_level("stuck", 1, env, env.obs(), env.action_space, max_steps=4, is_first_level_of_game=True)
    assert agent.eye.analyse_visual.call_count >= 1
    # First move carries model intent; later fallback moves must not inherit it.
    for call in agent.eye.analyse_visual.call_args_list:
        assert "intended_plan" in call.kwargs
        assert "expected_effect" in call.kwargs
        assert "world_model_block" in call.kwargs


def test_fast_mode_periodically_checks_novel_changes(agent):
    env = Environment(changing=True)
    runner = ARCRunner(agent, fast_step_eval=True, full_eval_interval=2, max_iterations_per_level=1)
    agent.eye.analyse_visual = Mock(wraps=agent.eye.analyse_visual)
    runner.play_level("periodic", 1, env, env.obs(), env.action_space, max_steps=4, is_first_level_of_game=True)
    assert agent.eye.analyse_visual.call_count == 2
    first = agent.eye.analyse_visual.call_args_list[0].kwargs
    assert first["intended_plan"] == "Test upward movement."
    assert first["expected_effect"] == "Player moves upward."


def test_real_border_motion_is_not_registered_as_hud():
    before = np.zeros((64, 64), dtype=int)
    after = before.copy()
    before[10, 1] = 9
    after[10, 2] = 9
    assert detect_real_change(before, after, "ACTION4")
    assert not get_hud_pixels()


def test_level_entry_clears_verified_hud_registry(agent):
    register_hud_pixel(1, 1)
    agent.enter_level("hud", 2, Environment().obs(), False, ["ACTION1"])
    assert not get_hud_pixels()


def test_border_targets_remain_in_object_guidance():
    grid = np.zeros((64, 64), dtype=int)
    grid[1:3, 1:3] = 8
    grid[20:23, 20:23] = 9
    text = render_detected_objects(grid)
    assert "Red" in text and "Blue" in text
    assert "role uncertain" in text


def make_transition():
    before = np.zeros((8, 8), dtype=int)
    after = before.copy()
    before[3, 2] = 9
    after[3, 3] = 9
    def state(grid, step):
        return ARCState.create("visual", 1, step, SimpleNamespace(grid=grid, state="PLAYING", levels_completed=0))
    initial, current = state(before, 0), state(after, 1)
    return initial, compute_transition(initial, current, ActionSignature("ACTION6", (("x", 3), ("y", 3))))


def test_vision_receives_before_after_and_action_coordinates():
    initial, transition = make_transition()
    model = Mock()
    model.invoke.return_value = SimpleNamespace(content="Recent findings: Token moved.\nOpen questions: Did it advance the goal?")
    eye = EyeChain(model)
    eye.analyse_visual("visual", initial, transition, "2 changed cells")
    parts = model.invoke.call_args.args[0][-1].content
    images = [p["image"] for p in parts if p["type"] == "image"]
    assert images == [initial.get_pil_image(), transition.current.get_pil_image()]
    assert str(transition.action_sig) in parts[0]["text"]
    assert eye.last_result.ok


def test_eye_receives_explicit_prediction_and_both_boards(tmp_path):
    initial, transition = make_transition()
    model = Mock()
    model.invoke.return_value = SimpleNamespace(content="Recent findings: Prediction supported.\nOpen questions: Is the rule general?\nPlan: Repeat elsewhere.")
    EyeChain(model).analyse_visual("visual", initial, transition, "2 cells",
                      intended_plan="Test switch", expected_effect="Door becomes green")
    parts = model.invoke.call_args.args[0][-1].content
    assert "Door becomes green" in parts[0]["text"]
    assert "Test switch" in parts[0]["text"]
    assert len([p for p in parts if p["type"] == "image"]) == 2


def test_vision_retries_malformed_response_once():
    initial, transition = make_transition()
    model = Mock()
    model.invoke.side_effect = [SimpleNamespace(content="ACTION=ACTION1"),
                               SimpleNamespace(content="Recent findings: Moved.\nOpen questions: Goal unknown.")]
    eye = EyeChain(model)
    assert "Moved" in eye.analyse_visual("visual", initial, transition, "2 cells")
    assert eye.last_result.ok and eye.last_result.attempts == 2


def test_failed_vision_is_not_saved_as_invented_mechanics(tmp_path):
    initial, _ = make_transition()
    model = Mock()
    model.invoke.side_effect = RuntimeError("backend unavailable")
    eye = EyeChain(model)
    cache = KnowledgeCache(tmp_path)
    output = eye.assume("visual", 1, initial, cache, object_list="Object 1")
    assert not eye.last_result.ok and eye.last_result.attempts == 2
    assert "interpretation unavailable" in output
    assert "Goal model:" not in output
    assert "backend unavailable" not in cache.scratch("visual")


def test_failed_evaluation_cannot_pollute_world_model(agent):
    env = Environment(changing=True)
    agent.eye.analyse_visual = Mock(return_value="")
    runner = ARCRunner(agent, fast_step_eval=True, full_eval_interval=1, max_iterations_per_level=1)
    runner.play_level("failure", 1, env, env.obs(), env.action_space, max_steps=2, is_first_level_of_game=True)
    assert "Board changed" in agent.world_model.recent_findings
    assert "INFERENCE FAILED" not in agent.world_model.to_prompt_block()
    assert agent.eye.analyse_visual.call_count == 2


def test_eye_observation_feeds_next_brain_decision(agent):
    env = Environment(changing=True)
    observation = "Recent findings: Blue token moved right.\nOpen questions: Goal unknown."
    agent.eye.analyse_visual = Mock(return_value=observation)
    agent.decide_action = Mock(wraps=agent.decide_action)
    runner = ARCRunner(agent, speculative_plan_max_steps=0, fast_step_eval=False,
                       max_iterations_per_level=1)
    runner.play_level("eye_brain", 1, env, env.obs(), env.action_space,
                      max_steps=2, is_first_level_of_game=True)
    assert observation in agent.decide_action.call_args_list[1].args[5]
    assert agent.eye.analyse_visual.call_count == 2
    assert set(runner.last_stage_status) == {"vision"}


def test_failed_attempt_review_uses_brain_and_saves_hypotheses(agent):
    initial, transition = make_transition()
    agent.brain.review = Mock(return_value="FAILURE_REASON: Repeated moves\nRULES:\n- Test another direction")
    agent.review_failed_iteration("visual", 1, 1, initial, transition.current)
    agent.brain.review.assert_called_once()
    assert "Test another direction" in agent.cache.scratch("visual")
    assert "REVIEW HYPOTHESES (UNVERIFIED)" in agent.cache.scratch("visual")


def test_retry_preserves_review_world_model_and_experiment(agent):
    env = Environment()
    initial = agent.enter_level("retry_memory", 1, env.obs(), True, env.action_space)
    agent.brain.review = Mock(return_value=(
        "Action model: Movement buttons affect both tokens.\n"
        "Hypotheses: H1 shared movement; H2 independent movement.\n"
        "Plan: Test a perpendicular movement.\nExpected effect: Observe both tokens.\n"
        "FAILURE_REASON: Same experiment repeated\nRULES:\n- Compare both tokens"))
    agent.review_failed_iteration("retry_memory", 1, 1, initial, initial)
    agent.enter_level("retry_memory", 1, env.obs(), False, env.action_space)
    assert agent.world_model.action_model == "Movement buttons affect both tokens."
    assert agent.world_model.current_plan == "Test a perpendicular movement."
    assert agent.world_model.expected_effect == "Observe both tokens."
    assert "FAILURE_REASON" not in agent.world_model.to_prompt_block()
    from pathlib import Path
    assert "H1 shared movement" in (Path(agent.memory_root) / "retry_memory/world_model.json").read_text()


def test_dynamic_temperature_stuck_vs_normal(agent):
    env = Environment()
    s0 = agent.enter_level("temp_test", 1, env.obs(), True, env.action_space)
    agent.brain.decide_action = Mock(return_value="Plan: move\nACTION=ACTION1")

    # 1. Normal move: temperature should be 0.0
    agent.decide_action("temp_test", 1, s0, s0, env.action_space, "", is_stuck=False)
    assert agent.brain.decide_action.call_args.kwargs.get("temperature") == 0.0

    # 2. Stuck / NO-OP move: temperature should switch to 1.0
    agent.decide_action("temp_test", 1, s0, s0, env.action_space, "", is_stuck=True)
    assert agent.brain.decide_action.call_args.kwargs.get("temperature") == 1.0