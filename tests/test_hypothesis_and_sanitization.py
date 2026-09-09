"""Unit tests for hypothesis creation, non-English token sanitization, and hash-free context."""

from enum import Enum
import contextlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest

from arc_agent.chains.brain import BrainChain
from arc_agent.chains.eye import EyeChain
from arc_agent.config import AppConfig, ModelConfig
from arc_agent.core.actions import ActionSignature, ARCActionMapper
from arc_agent.core.state import ARCState, compute_transition
from arc_agent.memory.knowledge import KnowledgeCache
from arc_agent.memory.trajectory import TrajectoryMemory
from arc_agent.memory.world_model import WorldModel
from arc_agent.models import gemma_transformers
from arc_agent.models.gemma_transformers import (
    GemmaTransformersChatModel, MockChatModel, _sanitize_llm_text,
)


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


@pytest.mark.parametrize("raw, expected_action, expected_data", [
    ("Plan: 移动向上 ACTION=ACTION1", Action.ACTION1, {}),
    ("Plan: 点击目标\nACTION=ACTION6 X=2 Y=3", Action.ACTION6, {"x": 2, "y": 3}),
])
def test_sanitizer_preserves_valid_actions(raw, expected_action, expected_data, monkeypatch):
    rescue = Mock(wraps=gemma_transformers._rescue_action_from_corrupted_text)
    monkeypatch.setattr(gemma_transformers, "_rescue_action_from_corrupted_text", rescue)
    clean = _sanitize_llm_text(raw, action_response=True)
    assert ARCActionMapper.parse(clean, list(Action), (10, 10)) == (expected_action, expected_data)
    rescue.assert_not_called()


@pytest.mark.parametrize("raw", [
    "Plan: 移动向上\nUP",  # Readable text remains but ACTION= is absent.
    "Plan: 移动向上 ACTION=???\nUP",  # A keyword alone is insufficient.
    "Plan: Move toward the target at the top of the board. 中\nUP",  # Low foreign ratio.
])
def test_sanitizer_rescues_action_despite_readable_fragments(raw, monkeypatch, capsys):
    rescue = Mock(wraps=gemma_transformers._rescue_action_from_corrupted_text)
    monkeypatch.setattr(gemma_transformers, "_rescue_action_from_corrupted_text", rescue)
    clean = _sanitize_llm_text(raw, action_response=True)
    rescue.assert_called_once_with(raw)
    assert clean == "ACTION=ACTION1"
    assert ARCActionMapper.parse(clean, list(Action)) == (Action.ACTION1, {})
    assert repr(raw) in capsys.readouterr().out


def test_sanitizer_rescues_multiline_corrupted_output():
    """Verify that multi-line text with foreign characters before and after ACTION= is cleanly handled and parsed."""
    raw = (
        "Plan: 探索这个谜题并向上移动。\n"
        "ACTION=ACTION1\n"
        "目标是到达绿色方块。"
    )
    clean = _sanitize_llm_text(raw)
    assert ARCActionMapper.parse(clean, list(Action)) == (Action.ACTION1, {})

    # Test multi-line text where ACTION= is missing and only alias UP exists
    raw_alias = (
        "Plan: 探索这个谜题并向上移动。\n"
        "UP\n"
        "目标是到达绿色方块。"
    )
    clean_alias = _sanitize_llm_text(raw_alias, action_response=True)
    assert clean_alias == "ACTION=ACTION1"
    assert ARCActionMapper.parse(clean_alias, list(Action)) == (Action.ACTION1, {})

    # Test integer action in ACTION= line
    raw_int = "Plan: 移动\nACTION=1\n继续移动"
    clean_int = _sanitize_llm_text(raw_int)
    assert ARCActionMapper.parse(clean_int, list(Action)) == (Action.ACTION1, {})

    # Test coordinate action rescue with multi-line surrounding text
    raw_coord = "Plan: 点击\nACTION=ACTION6 X=5 Y=7\n完成"
    clean_coord = _sanitize_llm_text(raw_coord)
    assert ARCActionMapper.parse(clean_coord, list(Action), (10, 10)) == (Action.ACTION6, {"x": 5, "y": 7})


def test_sanitizer_foreign_only_output_is_empty():
    assert _sanitize_llm_text("移动向上") == ""


def test_repetition_penalty_defaults_are_disabled():
    assert ModelConfig().repeat_penalty == 1.0
    assert GemmaTransformersChatModel().repeat_penalty == 1.0
    config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    assert AppConfig.from_yaml(config_path).model.repeat_penalty == 1.0


@pytest.mark.parametrize("actions", [[Action.ACTION1, Action.ACTION2], [Action.ACTION6]])
def test_brain_passes_stops_to_model(tmp_path, actions):
    response = "Plan: Select target.\nACTION=ACTION6 X=2 Y=3"
    model = Mock()
    model.invoke.return_value = SimpleNamespace(content=response)
    state = ARCState.create("test_game", 1, 0, DummyObservation(np.zeros((10, 10), dtype=int)))
    result = BrainChain(model).decide_action(
        "test_game", 1, state, state, actions, "", KnowledgeCache(tmp_path),
    )
    assert result == response
    kwargs = model.invoke.call_args.kwargs
    assert kwargs["stop"] == ["[END_ACTION]"]
    assert kwargs["action_response"] is True
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == 128
    assert not any(stop in response for stop in kwargs["stop"])


@pytest.mark.parametrize("raw", [
    "Plan: 点击\nACTION=ACTION6 X=２ Y=３",
    "Plan: 点击\nＡＣＴＩＯＮ＝ＡＣＴＩＯＮ６ Ｘ＝２ Ｙ＝３",
])
def test_sanitizer_normalizes_fullwidth_action_and_coordinates(raw):
    clean = _sanitize_llm_text(raw, action_response=True)
    assert ARCActionMapper.parse(clean, list(Action), (10, 10)) == (Action.ACTION6, {"x": 2, "y": 3})


@pytest.mark.parametrize("raw", [
    "Plan: 移动\nAvoid ACTION1; choose ACTION2.",
    "Plan: 移动\nACTION1\nACTION2",
    "Plan: 移动\nUP\nDOWN",
    "Plan: 移动\nDo not go UP.",
    "Plan: 移动向上 UP",  # An inline prose mention is not a standalone choice.
])
def test_sanitizer_retries_ambiguous_or_prose_choices(raw):
    assert gemma_transformers._rescue_action_from_corrupted_text(raw) == ""
    assert _sanitize_llm_text(raw, action_response=True) == ""


@pytest.mark.parametrize("actions", ["ACTION1 moves up; ACTION2 moves down.", "UP", "ACTION1"])
def test_sanitizer_preserves_world_model_action_mentions(actions):
    raw = f"World model: 移动 player at top.\nAction model: {actions}\nGoal model: Reach the target."
    clean = _sanitize_llm_text(raw)
    assert clean == raw.replace("移动", "")


@pytest.mark.parametrize("choice, expected", [
    ("ACTION=ACTION1", "ACTION=ACTION1"),
    ("**ACTION**=1", "ACTION=ACTION1"),
    ("ACTION=ACTION6 X=5 Y=7", "ACTION=ACTION6 X=5 Y=7"),
    ("CLICK X=２ Y=３", "ACTION=ACTION6 X=2 Y=3"),
    ("ACTION=ACTION6 X=2 Y=-3", ""),
    ("ACTION6", ""),
])
def test_rescue_directly_handles_multiline_choices(choice, expected):
    raw = f"Plan: 移动\n{choice}\nExtra trailing text."
    assert gemma_transformers._rescue_action_from_corrupted_text(raw) == expected


def test_sanitizer_diagnostics_work_on_cp1252_stream():
    buffer = io.BytesIO()
    with io.TextIOWrapper(buffer, encoding="cp1252") as stream:
        with contextlib.redirect_stdout(stream):
            clean = _sanitize_llm_text("Plan: 移动向上\nUP", action_response=True)
            gemma_transformers._log_llm("🔍 [LLM RAW RESPONSE] text='中'")
        stream.flush()
        logged = buffer.getvalue().decode("cp1252")
    assert clean == "ACTION=ACTION1"
    assert "Raw text preview" in logged
    assert "Rescued ACTION" in logged
    assert "LLM RAW RESPONSE" in logged


@pytest.mark.parametrize("leading, separator", [("\n", "\n"), ("", "\n\n"), ("\n\n", "\n\n")])
def test_brain_stops_after_action_despite_blank_lines(tmp_path, leading, separator):
    captured = {}

    class StoppingModel:
        def invoke(self, messages, **kwargs):
            captured.update(kwargs)
            text = leading + "Plan: Click the target." + separator + "ACTION=ACTION6 X=2 Y=3[END_ACTION]"
            text += "\nPlan: Repeat.\nACTION=ACTION1"
            # Emulate a provider honoring the supplied stop strings during decoding.
            for stop in kwargs.get("stop", []):
                text = text.split(stop)[0]
            return SimpleNamespace(content=text)

    state = ARCState.create("test_game", 1, 0, DummyObservation(np.zeros((10, 10), dtype=int)))
    result = BrainChain(StoppingModel()).decide_action(
        "test_game", 1, state, state, [Action.ACTION6], "", KnowledgeCache(tmp_path),
    )
    assert ARCActionMapper.parse(result, list(Action), (10, 10)) == (Action.ACTION6, {"x": 2, "y": 3})
    assert "Repeat" not in result
    assert captured["action_response"] is True


@pytest.mark.parametrize("action_response, raw, expected", [
    (True, "Plan: 移动\nUP", "ACTION=ACTION1"),
    (False, "Plan: 移动\nUP", "Plan: \nUP"),
    (False, "World model: 移动\nAction model: ACTION1 moves up.",
     "World model: \nAction model: ACTION1 moves up."),
])
def test_generate_scopes_rescue_without_forwarding_flag_to_transformers(monkeypatch, action_response, raw, expected):
    # Exercise the wrapper with CPU array doubles; no model weights or torch installation needed.
    class DeviceArray(np.ndarray):
        def to(self, device):
            return self

    class Processor:
        pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            return "Prompt"

        def __call__(self, **kwargs):
            return {"input_ids": np.array([[1]]).view(DeviceArray)}

        def decode(self, ids, **kwargs):
            return raw

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        device=str,
        cuda=SimpleNamespace(is_available=lambda: False),
        inference_mode=contextlib.nullcontext,
    ))
    monkeypatch.delenv("ENABLE_PROMPT_LOOKUP", raising=False)
    backend = Mock()
    backend.generate.return_value = np.array([[1, 2, 3]])
    model = GemmaTransformersChatModel(model=backend, processor=Processor())
    result = model._generate([], action_response=action_response)
    assert result.generations[0].message.content == expected
    assert "action_response" not in backend.generate.call_args.kwargs


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
        brain_chain=BrainChain(mock_model),
        resolver=GameStateResolver(),
        memory_root=str(tmp_path / "memory"),
    )

    agent.log_action("game_test", 1, 1, sig, "hash_before_12345", "hash_after_67890")
    log_content = agent.cache.actions_log("game_test", 1)
    assert "hash_before" not in log_content
    assert "hash_after" not in log_content
    assert "Changed" in log_content
