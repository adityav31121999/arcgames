"""Unit tests for ARCState and ARCTransition representations."""

import numpy as np
import pytest

from arc_agent.core.state import ARCState, compute_state_hash, compute_transition
from arc_agent.core.actions import ActionSignature


class DummyObservation:
    def __init__(self, grid, state="PLAYING", levels_completed=0):
        self.grid = grid
        self.state = state
        self.levels_completed = levels_completed


def test_arc_state_creation_and_hashing():
    grid = np.zeros((12, 12), dtype=int)
    grid[5, 5] = 9  # Blue pixel
    obs = DummyObservation(grid=grid, state="PLAYING", levels_completed=1)

    state = ARCState.create(game_id="game_1", level=1, step=0, obs=obs)

    assert state.game_id == "game_1"
    assert state.level == 1
    assert state.step == 0
    assert state.grid is not None
    assert state.grid.shape == (12, 12)
    assert state.levels_completed == 1
    assert len(state.state_hash) == 64
    assert "Shape: (12, 12)" in state.text_repr


def test_arc_state_json_metadata():
    grid = np.ones((10, 10), dtype=int)
    obs = DummyObservation(grid=grid, state="PLAYING", levels_completed=0)
    state = ARCState.create(game_id="game_1", level=1, step=0, obs=obs)

    data = state.json_data
    assert "frame" in data
    assert isinstance(data["frame"], list)
    assert len(data["frame"]) == 10

    compact_str = state.compact_json_repr
    assert "Grid List" in compact_str

    proper_str = state.proper_json_repr
    assert "1" in proper_str


def test_transition_computation():
    grid1 = np.zeros((12, 12), dtype=int)
    grid2 = np.zeros((12, 12), dtype=int)
    grid2[5, 5] = 14  # Green pixel inside gameplay area (offset > 4)

    obs1 = DummyObservation(grid=grid1)
    obs2 = DummyObservation(grid=grid2)

    s1 = ARCState.create("g", 1, 0, obs1)
    s2 = ARCState.create("g", 1, 1, obs2)

    sig = ActionSignature(name="ACTION1")
    transition = compute_transition(s1, s2, action_sig=sig)

    assert transition.changed is True
    assert transition.is_noop is False
    assert transition.action_sig == sig
