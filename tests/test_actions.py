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


def test_numeric_movement_does_not_inherit_click_coordinates():
    action, data = ARCActionMapper.parse("Plan: Try a tile at (33,32).\nACTION=1 X=33 Y=32", [1, 2, 6], (64, 64))
    assert action == 1 and data == {}
    assert ActionSignature.from_action(1, {"x": 33, "y": 32}) == ActionSignature("ACTION1")
    action, data = ARCActionMapper.parse("ACTION=6 X=33 Y=32", [1, 2, 6], (64, 64))
    assert action == 6 and data == {"x": 33, "y": 32}


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


def test_action_mapper_standalone_action_and_multiaction_reasoning():
    """Verify that standalone action lines like 'ACTION4' are parsed correctly even when earlier lines mention other actions."""
    available = [Action.ACTION1, Action.ACTION2, Action.ACTION3, Action.ACTION4, Action.ACTION5, Action.ACTION6]

    # Exact log response from m0r0 run:
    response1 = """World model: 64x64 grid, split yellow (left) and orange (right) background, black central shape with two light blue squares at the bottom center.
Goal model: Unknown; likely involves manipulating the light blue squares or the black shape to achieve a target state.
Action model: ACTION2 (Down) has previously caused changes in the grid (moving an object or cursor), but the most recent ACTION2 resulted in a NO-OP.
Hypotheses: H1: The cursor/object has reached the boundary of the grid or the bottom of the shape and cannot move further down. H2: The object is stuck or the movement logic changed.
Open questions: Is the object at the bottom edge? Does a different direction (ACTION1, ACTION3, ACTION4) trigger a change?
Plan: Switch direction to ACTION4 (Right) to test if movement is possible horizontally, breaking the potential vertical lock.
Expected effect: A change in the grid state if horizontal movement is possible.

ACTION4"""
    act, data = ARCActionMapper.parse(response1, available)
    assert act == Action.ACTION4
    assert data == {}

    response2 = """World model: 64x64 grid, split background (Yellow/Orange), black central shape with two light blue squares at the bottom center.
Goal model: Unknown; likely involves interacting with the light blue squares or the black shape.
Action model: ACTION1 (Up) and ACTION2 (Down) appear to move a cursor or the central shape, as evidenced by the 100-pixel bounding box changes in the lower region.
Hypotheses: H1: The light blue squares are interactive objects that need to be selected or manipulated. H2: ACTION1/ACTION2 moves a cursor or the black shape vertically.
Open questions: Does moving the shape/cursor to the light blue squares trigger a change?
Plan: Use ACTION2 to move the object/cursor further down to interact with the light blue squares.
Expected effect: Further modification of the grid in the lower region or a change in the light blue squares.

ACTION2"""
    act, data = ARCActionMapper.parse(response2, available)
    assert act == Action.ACTION2
    assert data == {}

    # Cut off text where only Plan mentions action
    response3 = """World model: ...
Action model: ACTION1, ACTION3
Plan: Switch direction to ACTION4 (Right)
Expected effect: change in grid"""
    act, data = ARCActionMapper.parse(response3, available)
    assert act == Action.ACTION4


def test_parse_action_with_repeat():
    available = [Action.ACTION1, Action.ACTION2, Action.ACTION3, Action.ACTION4]
    resp = "Plan: Move right across corridor\nACTION=ACTION4 REPEAT=5"
    act, data = ARCActionMapper.parse(resp, available)
    assert act == Action.ACTION4
    assert data.get("repeat") == 5


def test_parse_action_with_x_multiplier():
    available = [Action.ACTION1, Action.ACTION2, Action.ACTION3, Action.ACTION4]
    resp = "Plan: Move down\nACTION=ACTION2 x 3 [END_ACTION]"
    act, data = ARCActionMapper.parse(resp, available)
    assert act == Action.ACTION2
    assert data.get("repeat") == 3


def test_parse_plan_expands_repeats():
    available = [Action.ACTION1, Action.ACTION2, Action.ACTION3, Action.ACTION4]
    plan_text = "ACTION=ACTION1 REPEAT=3\nACTION=ACTION4 REPEAT=2\nACTION=ACTION2"
    plan = ARCActionMapper.parse_plan(plan_text, available)
    assert len(plan) == 6
    assert [a for a, d in plan] == [
        Action.ACTION1, Action.ACTION1, Action.ACTION1,
        Action.ACTION4, Action.ACTION4,
        Action.ACTION2,
    ]


def test_complex_action_ignores_repeat():
    # ACTION6 (click) must not repeat blindly at same coordinates
    available = [Action.ACTION1, Action.ACTION6]
    resp = "ACTION=ACTION6 X=10 Y=20 REPEAT=5"
    act, data = ARCActionMapper.parse(resp, available)
    assert act == Action.ACTION6
    assert data == {"x": 10, "y": 20}
    assert "repeat" not in data
