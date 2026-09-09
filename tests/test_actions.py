"""Unit tests for ARCActionMapper and coordinate parsing heuristics."""

from enum import Enum
import pytest

from arc_agent.core.actions import ARCActionMapper, ActionSignature, validate_coordinates


class Action(Enum):
    RESET = 0
    ACTION1 = 1
    ACTION2 = 2
    ACTION3 = 3
    ACTION4 = 4
    ACTION5 = 5
    ACTION6 = 6
    ACTION7 = 7

    def is_complex(self):
        return self == Action.ACTION6


def test_action_mapper_simple_parse():
    available = [Action.ACTION1, Action.ACTION2, Action.ACTION3, Action.ACTION4]
    
    text = "Based on observation, the next move is:\nACTION=ACTION1"
    act, data = ARCActionMapper.parse(text, available)
    assert act == Action.ACTION1
    assert data == {}

    text2 = "ACTION: ACTION3"
    act2, data2 = ARCActionMapper.parse(text2, available)
    assert act2 == Action.ACTION3
    assert data2 == {}


def test_action_mapper_complex_coordinates():
    available = [Action.ACTION1, Action.ACTION6]
    grid_shape = (20, 20)

    text = "ACTION=ACTION6 X=12 Y=8"
    act, data = ARCActionMapper.parse(text, available, grid_shape=grid_shape)
    assert act == Action.ACTION6
    assert data == {"x": 12, "y": 8}


def test_action_mapper_out_of_bounds_rejection():
    available = [Action.ACTION1, Action.ACTION6]
    grid_shape = (10, 10)

    # (15, 20) is out of (10, 10) bounds
    text = "ACTION=ACTION6 X=15 Y=20"
    act, data = ARCActionMapper.parse(text, available, grid_shape=grid_shape)
    assert act is None


def test_action_mapper_prohibited_signature():
    available = [Action.ACTION1, Action.ACTION2]
    prohibited = {ActionSignature(name="ACTION1")}

    text = "ACTION=ACTION1"
    act, data = ARCActionMapper.parse(text, available, prohibited=prohibited)
    assert act is None


def test_parse_plan():
    available = [Action.ACTION1, Action.ACTION2, Action.ACTION4]
    plan_text = """
    1. ACTION=ACTION1
    2. ACTION=ACTION4
    3. ACTION=ACTION2
    """
    plan = ARCActionMapper.parse_plan(plan_text, available)
    assert len(plan) == 3
    assert plan[0][0] == Action.ACTION1
    assert plan[1][0] == Action.ACTION4
    assert plan[2][0] == Action.ACTION2


def test_action_mapper_coordinate_formats():
    available = [Action.ACTION6]
    grid_shape = (64, 64)

    cases = [
        ("ACTION=ACTION6 [X=10, Y=20]", {"x": 10, "y": 20}),
        ("ACTION=ACTION6 Y=20 X=10", {"x": 10, "y": 20}),
        ("ACTION=ACTION6 (5, 8)", {"x": 5, "y": 8}),
        ("ACTION=ACTION6 [5, 8]", {"x": 5, "y": 8}),
        ("ACTION=ACTION6 12 34", {"x": 12, "y": 34}),
        ("ACTION=ACTION6 12, 34", {"x": 12, "y": 34}),
    ]
    for text, expected in cases:
        act, data = ARCActionMapper.parse(text, available, grid_shape=grid_shape)
        assert act == Action.ACTION6
        assert data == expected


def test_agent_log_action_signature_formats(tmp_path):
    from unittest.mock import MagicMock
    from arc_agent.agent.arc_langchain_agent import ARCLangChainAgent

    mock_chain = MagicMock()
    agent = ARCLangChainAgent(
        eye_chain=mock_chain,
        brain_chain=mock_chain,
        resolver=MagicMock(),
        memory_root=str(tmp_path / "memory"),
        vision_cache_dir=str(tmp_path / "vision"),
    )

    # 1. ActionSignature created via from_action (data is tuple of pairs)
    sig_tuple = ActionSignature.from_action(Action.ACTION6, {"x": 14, "y": 18})
    assert isinstance(sig_tuple.data, tuple)
    agent.log_action("test_game", 1, 1, sig_tuple, "hash_before_123", "hash_after_456")

    # 2. ActionSignature without data
    sig_simple = ActionSignature(name="ACTION1")
    agent.log_action("test_game", 1, 2, sig_simple, "hash_before_123", "hash_after_456")

    # 3. None action_sig
    agent.log_action("test_game", 1, 3, None, "hash_before_123", "hash_after_456")

    # Verify log content
    log_file = tmp_path / "memory" / "test_game" / "level_1" / "actions.md"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "ACTION6(X=14 Y=18)" in content
    assert "ACTION1" in content
    assert "UNKNOWN" in content


def test_action_mapper_integer_actions():
    """Verify ARCActionMapper handles raw integer action spaces [1, 2, 3, 4, 6, 7]."""
    available = [1, 2, 3, 4, 5, 6, 7]

    # Test ACTION=ACTION1 maps to 1
    act, data = ARCActionMapper.parse("ACTION=ACTION1", available)
    assert act == 1
    assert data == {}

    # Test ACTION=1 maps to 1
    act, data = ARCActionMapper.parse("ACTION=1", available)
    assert act == 1

    # Test ACTION=UP maps to 1
    act, data = ARCActionMapper.parse("ACTION=UP", available)
    assert act == 1

    # Test ACTION=RIGHT maps to 4
    act, data = ARCActionMapper.parse("ACTION=RIGHT", available)
    assert act == 4

    # Test markdown bold **ACTION**: 3
    act, data = ARCActionMapper.parse("**ACTION**: 3", available)
    assert act == 3

    # Test complex click with integer action 6
    act, data = ARCActionMapper.parse("ACTION=6 X=15 Y=25", available, grid_shape=(30, 30))
    assert act == 6
    assert data == {"x": 15, "y": 25}

    # Test ACTION=CLICK X=10 Y=20 maps to 6
    act, data = ARCActionMapper.parse("ACTION=CLICK X=10 Y=20", available, grid_shape=(30, 30))
    assert act == 6
    assert data == {"x": 10, "y": 20}


def test_action_mapper_markdown_and_prefix_formats():
    """Verify various prompt reply formats like 'Next action: ACTION=1' or '* ACTION: UP'."""
    available = [Action.ACTION1, Action.ACTION2, Action.ACTION3, Action.ACTION4]

    cases = [
        ("Next action: ACTION=ACTION1", Action.ACTION1),
        ("Next action: ACTION=1", Action.ACTION1),
        ("Plan: Move up to explore.\nACTION=UP", Action.ACTION1),
        ("- **ACTION**: ACTION2", Action.ACTION2),
        ("* ACTION: DOWN", Action.ACTION2),
        ("Plan: Try going left.\nNext action: ACTION=LEFT", Action.ACTION3),
    ]
    for text, expected in cases:
        act, _ = ARCActionMapper.parse(text, available)
        assert act == expected, f"Failed parsing: {text!r}"


def test_prohibited_action_distinguishes_syntax_validity():
    """Verify that a prohibited action can be recognized as valid syntax even though rejected."""
    available = [Action.ACTION1, Action.ACTION2]
    prohibited = {ActionSignature(name="ACTION1")}

    # Rejected when prohibited is passed
    act, _ = ARCActionMapper.parse("ACTION=ACTION1", available, prohibited=prohibited)
    assert act is None

    # But accepted when prohibited=None, allowing the agent to confirm the model is responsive
    act_syntax, _ = ARCActionMapper.parse("ACTION=ACTION1", available, prohibited=None)
    assert act_syntax == Action.ACTION1

