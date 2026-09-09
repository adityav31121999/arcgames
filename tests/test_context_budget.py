"""Token-budget and image-marker coverage using a deterministic processor double."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from arc_agent.chains.brain import BrainChain
from arc_agent.chains.inference import build_messages, invoke_stage
from arc_agent.core.context import ContextBudgetError, context_part
from arc_agent.core.state import ARCState
from arc_agent.memory.knowledge import KnowledgeCache, _write_scratch_section
from arc_agent.models.gemma_transformers import GemmaTransformersChatModel


class Processor:
    """Each text word is one token; each image expands to four slots and two markers."""
    image_processor = object()
    image_token_id = 2
    boi_token = "<|image>"
    eoi_token = "<image|>"

    def __init__(self, missing_end=False, missing_pixels=False):
        self.tokenizer = self
        self.missing_end, self.missing_pixels = missing_end, missing_pixels
        self.last_text = ""
        self.calls = 0

    def convert_tokens_to_ids(self, token):
        return {self.boi_token: 1, self.eoi_token: 3}[token]

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(
            part["text"] if part["type"] == "text" else "<|image|>"
            for message in messages for part in message["content"]
        )

    def __call__(self, text, images=None, **kwargs):
        self.calls += 1
        self.last_text = text[0]
        assert kwargs["add_special_tokens"] is False
        ids = []
        for word in text[0].split():
            ids.extend(([1] + [2] * 4 + ([] if self.missing_end else [3])) if word == "<|image|>" else [10])
        inputs = {"input_ids": np.array([ids])}
        if images is not None:
            assert len(images) == 1 and isinstance(images[0], list)
            assert len(images[0]) == text[0].count("<|image|>")
            if not self.missing_pixels:
                inputs["pixel_values"] = np.zeros((len(images[0]), 3, 2, 2))
        return inputs


def prepare(wrapper, prompt="Required action instructions", context=(), images=(), reserve=10):
    messages = build_messages("System instructions", prompt, images, context)
    formatted, pil_images = wrapper._extract_images_and_text(messages)
    return wrapper._prepare_inputs(formatted, pil_images, reserve)


def test_budget_counts_images_and_reserves_output():
    wrapper = GemmaTransformersChatModel(processor=Processor(), max_context_length=40)
    images = [("Before", Image.new("RGB", (4, 4))), ("After", Image.new("RGB", (4, 4)))]
    prepare(wrapper, images=images)
    usage = wrapper.last_context_usage
    assert usage["images"] == 2
    assert usage["image_tokens"] == 8
    assert usage["input_tokens"] + usage["output_reserved"] <= 40
    assert usage["text_and_control_tokens"] + 8 == usage["input_tokens"]


def test_optional_context_is_compacted_without_losing_task_images_or_fresh_memory():
    processor = Processor()
    wrapper = GemmaTransformersChatModel(processor=processor, max_context_length=55)
    context = [
        context_part("Verified mechanics", "NEW_FINDING door opens after switch", 0, "head"),
        context_part("Old history", "\n".join(f"old_{i} action result" for i in range(100)), 4),
    ]
    prepare(wrapper, context=context, images=[("Current", Image.new("RGB", (4, 4)))])
    assert processor.calls > 1
    assert "Required action instructions" in processor.last_text
    assert "System instructions" in processor.last_text
    assert "NEW_FINDING" in processor.last_text
    assert "old_99" in processor.last_text and "old_0 " not in processor.last_text
    assert processor.last_text.count("<|image|>") == 1
    assert wrapper.last_context_usage["compacted_sections"] == ["Old history"]


def test_required_context_overflow_fails_explicitly():
    wrapper = GemmaTransformersChatModel(processor=Processor(), max_context_length=5)
    with pytest.raises(ContextBudgetError, match="effective 5-token maximum"):
        prepare(wrapper)


@pytest.mark.parametrize("config", [
    SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=20)),
    {"text_config": {"max_position_embeddings": 20}},
    SimpleNamespace(max_position_embeddings=20),
])
def test_loaded_model_capacity_caps_configured_context(config):
    wrapper = GemmaTransformersChatModel(model=SimpleNamespace(config=config), max_context_length=100)
    assert wrapper._context_limit() == 20
    with pytest.raises(ContextBudgetError):
        wrapper._validate_context_length(15, 6)


@pytest.mark.parametrize("flag, message", [("missing_end", "start/end markers"), ("missing_pixels", "no image tensors")])
def test_missing_image_markers_or_tensors_fail_before_generation(flag, message):
    wrapper = GemmaTransformersChatModel(processor=Processor(**{flag: True}))
    with pytest.raises(RuntimeError, match=message):
        prepare(wrapper, images=[("Current", Image.new("RGB", (4, 4)))])


def test_fresh_section_entries_reach_brain_and_disk_stays_complete(tmp_path):
    cache = KnowledgeCache(tmp_path)
    original = "## HYPOTHESES & ASSUMPTIONS\n" + "".join(f"- Old finding {i}: " + "word " * 100 + "\n" for i in range(14))
    cache.write_scratch("g", original)
    marker = "NEW_EVIDENCE_RED_SWITCH_OPENS_DOOR"
    _write_scratch_section(cache, "g", "## HYPOTHESES & ASSUMPTIONS", marker)
    model = Mock()
    model.invoke.return_value = SimpleNamespace(content="ACTION=ACTION1")
    state = ARCState.create("g", 1, 0, SimpleNamespace(grid=np.zeros((3, 3), dtype=int), state="PLAYING", levels_completed=0))
    BrainChain(model).decide_action("g", 1, state, state, ["ACTION1"], "", cache)
    messages = model.invoke.call_args.args[0]
    assert marker in "\n".join(p.get("text", "") for p in messages[-1].content)
    wrapper = GemmaTransformersChatModel(processor=Processor(), max_context_length=470)
    formatted, images = wrapper._extract_images_and_text(messages)
    wrapper._prepare_inputs(formatted, images, 128)
    assert marker in wrapper.processor.last_text
    assert "Old finding 13" in cache.scratch("g")
    assert marker in (tmp_path / "g" / "scratchpad.md").read_text(encoding="utf-8")


def test_memory_keeps_multiline_findings_when_new_entry_is_added(tmp_path):
    cache = KnowledgeCache(tmp_path)
    first = "Recent findings: Switch changed.\nOpen questions: Was the door opened?\nPlan: Check the door."
    _write_scratch_section(cache, "g", "## HYPOTHESES & ASSUMPTIONS", first)
    _write_scratch_section(cache, "g", "## HYPOTHESES & ASSUMPTIONS", "Another observation")
    assert first in cache.scratch("g")


def test_consolidated_entries_remain_available_in_archive(tmp_path):
    cache = KnowledgeCache(tmp_path)
    for index in range(20):
        _write_scratch_section(cache, "g", "## OBSERVATIONS", f"Observation_{index:02d}")
    working = cache.scratch("g")
    archived = (tmp_path / "g" / "memory_history.md").read_text(encoding="utf-8")
    assert "Observation_19" in working and "Observation_00" not in working
    for index in range(20):
        assert f"Observation_{index:02d}" in archived


def test_consolidation_archives_entries_from_older_scratchpads(tmp_path):
    cache = KnowledgeCache(tmp_path)
    cache.write_scratch("g", "## OBSERVATIONS\n- Old observation from previous run\n")
    _write_scratch_section(cache, "g", "## OBSERVATIONS", "New finding", max_entries=1)
    assert "Old observation" not in cache.scratch("g")
    archived = (tmp_path / "g" / "memory_history.md").read_text(encoding="utf-8")
    assert "Old observation from previous run" in archived


def test_irreducible_context_error_is_not_retried_as_bad_model_output():
    model = Mock()
    model.invoke.side_effect = ContextBudgetError("Required images cannot fit")
    result = invoke_stage(model, "system", "task", stage="Vision", max_tokens=10)
    assert not result.ok and result.attempts == 1
    model.invoke.assert_called_once()
