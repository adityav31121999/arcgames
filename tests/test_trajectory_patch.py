"""Tests for the TrajectoryMemory momentum vs noop patch and action guard."""

from enum import Enum
import pytest

from arc_agent.core.actions import ActionSignature
from arc_agent.memory.trajectory import TrajectoryMemory, Outcome


class Action(Enum):
    ACTION1 = 1
    ACTION2 = 2
    ACTION3 = 3
    ACTION4 = 4

    def is_complex(self):
        return False


def test_momentum_streak_allows_repeated_working_action():
    mem = TrajectoryMemory()
    mem.reset("hash_s0")

    # Step 1: ACTION1 moves player from s0 to s1 (changed=True)
    sig1 = ActionSignature.from_action(Action.ACTION1)
    mem.record_transition("hash_s0", sig1, "hash_s1", changed=True)
    assert mem.momentum_streak == 1
    assert mem.noop_streak == 0
    assert not mem.is_action_blocked("hash_s0", Action.ACTION1)

    # Step 2: ACTION1 moves player from s1 to s2 (changed=True) -> momentum!
    mem.record_transition("hash_s1", sig1, "hash_s2", changed=True)
    assert mem.momentum_streak == 2
    assert mem.noop_streak == 0
    assert not mem.is_action_blocked("hash_s1", Action.ACTION1)

    # Repeating ACTION1 is still allowed because it changed the board
    allowed = mem.get_allowed_actions("hash_s2", [Action.ACTION1, Action.ACTION2])
    assert Action.ACTION1 in allowed
    assert not mem.should_diversify()
    assert mem.suggested_temperature() == 0.0

    # Context notice distinguishes momentum
    notice = mem.context_notice("hash_s2")
    assert notice is not None
    assert "[MOMENTUM]" in notice
    assert "ACTION1" in notice


def test_noop_streak_blocks_dead_action_and_escalates_temperature():
    mem = TrajectoryMemory()
    mem.reset("hash_wall")

    sig_right = ActionSignature.from_action(Action.ACTION4)

    # ACTION4 hits a wall from hash_wall (changed=False)
    mem.record_transition("hash_wall", sig_right, "hash_wall", changed=False)
    assert mem.momentum_streak == 0
    assert mem.noop_streak == 1
    assert mem.is_action_blocked("hash_wall", Action.ACTION4)

    # ACTION4 must now be blocked from hash_wall!
    allowed = mem.get_allowed_actions("hash_wall", [Action.ACTION1, Action.ACTION4])
    assert Action.ACTION4 not in allowed
    assert Action.ACTION1 in allowed

    # Notice reports dead action
    blocked_notice = mem.context_notice("hash_wall", action="ACTION4")
    assert blocked_notice is not None
    assert "[TRAJECTORY WARNING]" in blocked_notice
    assert "NO visible change" in blocked_notice

    # 2 more NO-OPs trigger temperature escalation
    mem.record_transition("hash_wall", sig_right, "hash_wall", changed=False)
    mem.record_transition("hash_wall", sig_right, "hash_wall", changed=False)
    assert mem.noop_streak == 3
    assert mem.should_diversify()
    assert mem.suggested_temperature() == 1.0


def test_allowed_actions_falls_back_if_everything_blocked():
    mem = TrajectoryMemory()
    mem.reset("hash_deadlock")

    sig1 = ActionSignature.from_action(Action.ACTION1)
    mem.record_transition("hash_deadlock", sig1, "hash_deadlock", changed=False)

    # Candidate actions has only ACTION1, which produced NO_CHANGE
    # It falls back to returning the candidate rather than empty list
    allowed = mem.get_allowed_actions("hash_deadlock", [Action.ACTION1])
    assert allowed == [Action.ACTION1]
