"""Regression coverage for observation and instruction loss in model context."""

from types import SimpleNamespace

import numpy as np

from arc_agent.chains.brain import BrainChain
from arc_agent.core.state import ARCState
from arc_agent.memory.knowledge import KnowledgeCache
from arc_agent.memory.world_model import WorldModel
from arc_agent.memory.knowledge import maybe_append_rule, apply_iteration_review
from arc_agent.models.gemma_transformers import GemmaTransformersChatModel
import pytest


def test_brain_receives_current_board_not_initial_board(tmp_path):
    captured = []

    class Recorder:
        def invoke(self, messages, **kwargs):
            captured.extend(messages)
            return SimpleNamespace(content="ACTION=ACTION1")

    def state(grid, step):
        return ARCState.create("g", 1, step, SimpleNamespace(
            grid=grid, state="PLAYING", levels_completed=0))

    initial = state(np.zeros((3, 3), dtype=int), 0)
    current = state(np.ones((3, 3), dtype=int), 1)
    BrainChain(Recorder()).decide_action(
        "g", 1, initial, current, ["ACTION1"], "", KnowledgeCache(tmp_path))
    parts = captured[-1].content
    assert parts[1]["image"] is current.get_pil_image()
    assert "X is column, Y is row" in parts[0]["text"]


def test_action_line_does_not_pollute_carried_plan():
    world = WorldModel()
    world.update_from_text("Plan: Test the blue object.\nACTION=ACTION6 X=4 Y=5")
    assert world.current_plan == "Test the blue object."


def test_append_preserves_persisted_memory_and_separates_entries(tmp_path):
    old = KnowledgeCache(tmp_path)
    old.append_scratch("g", "Earlier finding")
    old.append_ostate("g", "Earlier level")
    new = KnowledgeCache(tmp_path)
    new.append_scratch("g", "New finding")
    new.append_ostate("g", "New level")
    assert "Earlier finding\n\nNew finding" in new.scratch("g")
    assert "Earlier level\n\nNew level" in new.ostate("g")


def test_context_ceiling_reserves_output_tokens():
    model = GemmaTransformersChatModel()
    assert model.max_context_length == 81930
    model._validate_context_length(81000, 930)
    with pytest.raises(ValueError, match="No context was silently truncated"):
        model._validate_context_length(81000, 931)


def test_memory_and_world_fields_are_not_cut_off(tmp_path):
    cache = KnowledgeCache(tmp_path)
    long_text = "Beginning " + "observed fact " * 1000 + " End"
    cache.append_scratch("g", long_text)
    cache.append_ostate("g", long_text)
    cache.append_action_log("g", 1, long_text)
    assert long_text in cache.scratch("g")
    assert long_text in cache.ostate("g")
    assert cache.actions_log("g", 1) == long_text
    world = WorldModel()
    world.update_from_text("Recent findings: " + long_text)
    assert world.recent_findings == " ".join(long_text.split())


@pytest.mark.parametrize("verdict", [
    "EXPECTED: A wall blocked movement.",
    "DIVERGED: This must be a wall.",
    "There is no wall; movement was not blocked.",
])
def test_observations_do_not_promote_guesses_to_verified_rules(tmp_path, verdict):
    cache = KnowledgeCache(tmp_path)
    maybe_append_rule("g", verdict, False, False, cache)
    memory = cache.scratch("g")
    assert "VERIFIED MECHANICS" not in memory
    assert "cause unknown" in memory
    assert verdict in memory


def test_review_preserves_verified_rules_and_marks_its_claims_unverified(tmp_path):
    cache = KnowledgeCache(tmp_path)
    cache.write_scratch("g", "## VERIFIED MECHANICS AND RULES\n- Existing evidence-backed rule\n")
    apply_iteration_review(cache, "g", 1, 1, "FAILURE_REASON: Repeated a move\nRULES:\n- Maybe a wall")
    memory = cache.scratch("g")
    assert "Existing evidence-backed rule" in memory
    assert "## REVIEW HYPOTHESES (UNVERIFIED)\n- Maybe a wall" in memory


def test_eye_format_updates_world_model_and_includes_action_coordinates(tmp_path):
    from arc_agent.chains.eye import EyeChain
    from arc_agent.core.state import compute_transition
    from arc_agent.core.actions import ActionSignature
    captured = []

    class Recorder:
        def invoke(self, messages, **kwargs):
            captured.extend(messages)
            return SimpleNamespace(content=(
                "Recent findings: No visible change.\n"
                "Open questions: Does the target require activation?\n"
                "Plan: Test another target."))

    current = ARCState.create("g", 1, 0, SimpleNamespace(
        grid=np.zeros((3, 3), dtype=int), state="PLAYING", levels_completed=0))
    transition = compute_transition(current, current, ActionSignature("ACTION6", (("x", 1), ("y", 2))))
    result = EyeChain(Recorder()).analyse_visual("g", current, transition, "0 pixels")
    prompt = captured[-1].content[0]["text"]
    assert str(transition.action_sig) in prompt
    assert "Previous board" in prompt
    assert "Keep headers/markdown out" not in prompt
    world = WorldModel()
    world.update_from_text(result)
    assert world.current_plan == "Test another target."
    assert world.recent_findings == "No visible change."
