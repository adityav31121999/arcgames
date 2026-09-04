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
