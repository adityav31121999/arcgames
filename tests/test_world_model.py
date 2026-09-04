"""Unit tests for WorldModel goal+plan belief state tracking and agent integration."""

from enum import Enum
import numpy as np
import pytest

from arc_agent.core.actions import ARCActionMapper
from arc_agent.core.resolver import GameStateResolver
from arc_agent.core.state import ARCState
from arc_agent.chains.brain import BrainChain
from arc_agent.chains.debugger import DebuggerChain
from arc_agent.chains.eye import EyeChain
from arc_agent.chains.reviewer import ReviewerChain
from arc_agent.agent.arc_langchain_agent import ARCLangChainAgent
from arc_agent.agent.runner import ARCRunner
from arc_agent.memory.world_model import WorldModel, _extract_labeled_blocks
from arc_agent.models.gemma_transformers import MockChatModel


class Action(Enum):
    RESET = 0
    ACTION1 = 1
    ACTION2 = 2
    ACTION3 = 3
    ACTION6 = 6

    def is_complex(self):
        return self == Action.ACTION6


class DummyObservation:
    def __init__(self, grid):
        self.grid = grid
        self.state = "PLAYING"
        self.levels_completed = 0


class DummyEnv:
    def __init__(self, grid):
        self.grid = grid
        self.action_space = [Action.ACTION1, Action.ACTION2, Action.ACTION3]

    def reset(self):
        return DummyObservation(self.grid)

    def step(self, action, **kwargs):
        # Shift grid slightly on action
        new_grid = self.grid.copy()
        if action == Action.ACTION1:
            new_grid[0, 0] = 5
        return DummyObservation(new_grid)


def test_world_model_label_extraction():
    sample_text = """
    World model: The blue square is movable and the green region is the goal target.
    Goal model: Navigate blue into green while avoiding red walls.
    Action model: ACTION1 moves UP, ACTION2 moves DOWN.
    Plan: Move up two steps towards the opening.
    Recent findings:
    - Moving down collided with a wall.
    - Path to the right is blocked.
    """
    wm = WorldModel()
    wm.update_from_text(sample_text)

    assert "blue square is movable" in wm.world_model
    assert "Navigate blue into green" in wm.goal_model
    assert "ACTION1 moves UP" in wm.action_model
    assert "Move up two steps" in wm.current_plan
    assert "collided with a wall" in wm.recent_findings

    block = wm.to_prompt_block()
    assert "Working world model carried from earlier turns:" in block
    assert "World model: The blue square is movable" in block
    assert "Goal model: Navigate blue into green" in block
    assert "Plan: Move up two steps" in block


def test_world_model_level_reset():
    wm = WorldModel(
        world_model="Level 1 model",
        goal_model="Level 1 goal",
        current_plan="Step 1 plan",
        recent_findings="Found key",
        cross_level_notes="All levels have 64x64 grid with green goal",
    )

    wm.reset_level_fields()

    assert wm.world_model == ""
    assert wm.goal_model == ""
    assert wm.current_plan == ""
    assert wm.recent_findings == ""
    # Cross level notes must persist across levels
    assert wm.cross_level_notes == "All levels have 64x64 grid with green goal"


def test_brain_chain_with_world_model_injection(tmp_path):
    mock_model = MockChatModel(mock_responses=[
        "Plan: Move up to reach open passage.\nACTION=ACTION1",
    ])
    brain = BrainChain(mock_model)

    grid = np.zeros((10, 10), dtype=int)
    obs = DummyObservation(grid)
    s0 = ARCState.create("game_test", 1, 0, obs)
    from arc_agent.memory.knowledge import KnowledgeCache
    cache = KnowledgeCache(memory_root=tmp_path)

    wm = WorldModel(world_model="Blue is avatar", goal_model="Reach green")
    response = brain.decide_action(
        "game_test", 1, s0, s0, [Action.ACTION1, Action.ACTION2], "No warning", cache,
        world_model_block=wm.to_prompt_block(),
    )

    assert "ACTION=ACTION1" in response
    act, _ = ARCActionMapper.parse(response, [Action.ACTION1, Action.ACTION2])
    assert act == Action.ACTION1


def test_agent_world_model_closed_loop(tmp_path):
    mock_model = MockChatModel(mock_responses=[
        "World model: Player is cyan token.\nGoal model: Reach magenta portal.\nPlan: Advance towards corridor.",
        "Plan: Step forward.\nACTION=ACTION1",
        "EXPECTED: Move succeeded.\nRecent findings: Corridor is clear.",
    ])

    eye = EyeChain(mock_model)
    debugger = DebuggerChain(mock_model)
    brain = BrainChain(mock_model)
    reviewer = ReviewerChain(mock_model)
    resolver = GameStateResolver()

    agent = ARCLangChainAgent(
        eye_chain=eye,
        debugger_chain=debugger,
        brain_chain=brain,
        reviewer_chain=reviewer,
        resolver=resolver,
        memory_root=str(tmp_path),
    )

    grid = np.zeros((10, 10), dtype=int)
    obs = DummyObservation(grid)

    # 1. enter_level should seed world model from eye
    s0 = agent.enter_level("game_loop_test", 1, obs, is_first_level_of_game=True, valid_actions=[Action.ACTION1])
    assert "Player is cyan token" in agent.world_model.world_model
    assert "Reach magenta portal" in agent.world_model.goal_model

    # 2. decide_action should update world model with plan and return valid action
    action, data, note = agent.decide_action("game_loop_test", 1, s0, s0, [Action.ACTION1], "")
    assert action == Action.ACTION1
    assert "Step forward" in agent.world_model.current_plan

    # 3. runner loop integration check
    runner = ARCRunner(agent)
    env = DummyEnv(grid)
    final_obs, state, steps = runner.play_level(
        "game_loop_test", 1, env, obs, [Action.ACTION1, Action.ACTION2], max_steps=2
    )
    assert steps > 0
    assert not agent.world_model.is_empty()
