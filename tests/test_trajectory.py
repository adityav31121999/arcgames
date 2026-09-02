"""Unit tests for TrajectoryMemory, loop warnings, oscillation avoidance, and sprite tracking."""

from enum import Enum
import numpy as np
import pytest

from arc_agent.core.actions import ActionSignature
from arc_agent.memory.trajectory import TrajectoryMemory


class Action(Enum):
    ACTION1 = 1
    ACTION2 = 2
    ACTION3 = 3
    ACTION4 = 4

    def is_complex(self):
        return False


def test_trajectory_recording_and_loop_warning():
    mem = TrajectoryMemory()
    mem.reset("hash_s0")

    sig1 = ActionSignature.from_action(Action.ACTION1)
    is_repeat_state, is_repeat_trans = mem.record_transition("hash_s0", sig1, "hash_s1", changed=True)
    assert not is_repeat_state
    assert not is_repeat_trans
    assert mem.visits("hash_s1") == 1

    # Transition back to hash_s0
    sig2 = ActionSignature.from_action(Action.ACTION2)
    is_repeat_state, is_repeat_trans = mem.record_transition("hash_s1", sig2, "hash_s0", changed=True)
    assert is_repeat_state  # Visited hash_s0 again
    assert mem.visits("hash_s0") == 2

    warning = mem.loop_warning("hash_s0")
    assert "[LOOP WARNING]" in warning
    assert "visited 2x" in warning


def test_oscillation_avoidance():
    mem = TrajectoryMemory()
    mem.reset("hash_A")

    sig_up = ActionSignature.from_action(Action.ACTION1)
    sig_down = ActionSignature.from_action(Action.ACTION2)

    # A -> B via UP
    mem.record_transition("hash_A", sig_up, "hash_B", changed=True)
    # B -> A via DOWN (oscillation)
    mem.record_transition("hash_B", sig_down, "hash_A", changed=True)

    # Now at hash_A: oscillation target is hash_B (state_history[-3])
    # Action UP from A leads to B, so UP should be filtered if alternatives exist
    allowed = mem.get_allowed_actions("hash_A", [Action.ACTION1, Action.ACTION3, Action.ACTION4])
    assert Action.ACTION1 not in allowed
    assert Action.ACTION3 in allowed
    assert Action.ACTION4 in allowed


def test_sprite_region_tracking():
    mem = TrajectoryMemory()
    g1 = np.zeros((16, 16), dtype=int)
    g2 = np.zeros((16, 16), dtype=int)
    g2[6, 8] = 9  # Change at (Y=6, X=8)

    mem.update_sprite_region(g1, g2)
    assert mem.sprite_box is not None
    assert mem.sprite_box == (8, 8, 6, 6)

    guidance = mem.get_sprite_guidance()
    assert "[SPRITE NAVIGATION HIGHLIGHT]" in guidance
    assert "X=[8, 8], Y=[6, 6]" in guidance
