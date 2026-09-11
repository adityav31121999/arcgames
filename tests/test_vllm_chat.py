import base64
import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image
from langchain_core.messages import HumanMessage, SystemMessage

from arc_agent.models.vllm_chat import VLLMChatModel, load_vllm
from arc_agent.config import ModelConfig
from arc_agent.core.context import ContextBudgetError


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: kw))
    monkeypatch.setattr(VLLMChatModel, "_prepare_inputs", lambda self, *args: {})
    engine = Mock()
    engine.chat.return_value = [SimpleNamespace(outputs=[SimpleNamespace(text="ACTION=ACTION1")])]
    return VLLMChatModel(engine=engine)


def test_vllm_uses_chat_and_preserves_images_in_order(backend):
    images = [Image.new("RGB", (2, 2), color) for color in ("red", "blue")]
    response = backend.invoke([SystemMessage(content="Observe"), HumanMessage(content=[
        {"type": "text", "text": "Compare"},
        *[{"type": "image", "image": im} for im in images],
    ])], max_tokens=64, temperature=0.0, stop=["[END_ACTION]"])
    assert response.content == "ACTION=ACTION1"
    args, kw = backend.engine.chat.call_args
    assert args[0][0]["role"] == "system"
    parts = args[0][1]["content"]
    for part, original in zip(parts[1:], images):
        decoded = Image.open(io.BytesIO(base64.b64decode(part["image_url"]["url"].split(",")[1])))
        assert decoded.getpixel((0, 0)) == original.getpixel((0, 0))
    assert kw["sampling_params"]["temperature"] == 0
    assert kw["sampling_params"]["max_tokens"] == 64
    assert kw["sampling_params"]["stop"] == ["[END_ACTION]"]
    assert kw["chat_template_kwargs"] == {"enable_thinking": False}
    backend.engine.generate.assert_not_called()


def test_vllm_raw_health_output_is_not_sanitized(backend):
    raw = "much own garbled"
    backend.engine.chat.return_value[0].outputs[0].text = raw
    assert backend.invoke([HumanMessage(content="READY")], raw_output=True).content == raw


def test_context_failure_prevents_gpu_call(backend, monkeypatch):
    def fail(*args):
        raise ContextBudgetError("Required context too large")
    monkeypatch.setattr(VLLMChatModel, "_prepare_inputs", fail)
    with pytest.raises(ContextBudgetError):
        backend.invoke([HumanMessage(content="Test")])
    backend.engine.chat.assert_not_called()


def test_thinking_keeps_only_final_decision_and_does_not_stop_inside_thoughts(backend):
    backend.engine.chat.return_value[0].outputs[0].text = (
        "<|channel>thought\nACTION=ACTION6 [END_ACTION]\n<channel|>"
        "Hypotheses: H1 movement; H2 toggle.\nACTION=ACTION1 [END_ACTION]<turn|>")
    response = backend.invoke([HumanMessage(content="Choose a test")], enable_thinking=True,
                              max_tokens=2048, stop=["[END_ACTION]"])
    assert "ACTION6" not in response.content
    assert response.content.endswith("ACTION=ACTION1")
    kwargs = backend.engine.chat.call_args.kwargs
    assert kwargs["chat_template_kwargs"]["enable_thinking"] is True
    assert kwargs["sampling_params"]["skip_special_tokens"] is False
    assert kwargs["sampling_params"]["stop"] is None


def test_incomplete_thought_is_not_rescued_as_action(backend):
    backend.engine.chat.return_value[0].outputs[0].text = "<|channel>thought\nACTION=ACTION1"
    with pytest.raises(RuntimeError, match="Thinking budget exhausted"):
        backend.invoke([HumanMessage(content="Choose")], enable_thinking=True)


def test_vllm_loader_uses_local_checkpoint_and_no_transformers_weights(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
    llm = Mock(return_value=Mock())
    processor = SimpleNamespace(image_processor=object(), chat_template="native")
    auto = SimpleNamespace(from_pretrained=Mock(return_value=processor))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=llm, SamplingParams=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, "vllm.config", SimpleNamespace(ReasoningConfig=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoProcessor=auto))
    wrapper = load_vllm(ModelConfig(backend="vllm", max_context_length=32768), str(tmp_path))
    assert wrapper.engine is llm.return_value
    assert llm.call_args.kwargs["tensor_parallel_size"] == 1
    assert llm.call_args.kwargs["limit_mm_per_prompt"] == {"image": 2, "video": 0}
    assert llm.call_args.kwargs["max_model_len"] == 32768
    assert auto.from_pretrained.call_args.kwargs["local_files_only"] is True
    assert llm.call_args.kwargs["reasoning_config"] == {
        "reasoning_start_str": "<|channel>thought\n", "reasoning_end_str": "<channel|>",
    }
    assert wrapper.thinking_token_budget == 1024


def test_native_thinking_budget_leaves_room_for_final_answer(backend):
    backend.thinking_token_budget = 1024
    backend.engine.chat.return_value[0].outputs[0].text = "<channel|>ACTION=ACTION1"
    backend.invoke([HumanMessage(content="Choose")], enable_thinking=True, max_tokens=4096)
    params = backend.engine.chat.call_args.kwargs["sampling_params"]
    assert params["thinking_token_budget"] == 1024 and params["max_tokens"] == 4096
    backend.invoke([HumanMessage(content="Initial assumption")], enable_thinking=True,
                   thinking_token_budget=6144, max_tokens=8192)
    assert backend.engine.chat.call_args.kwargs["sampling_params"]["thinking_token_budget"] == 6144
    backend.invoke([HumanMessage(content="Observe")], max_tokens=512)
    assert "thinking_token_budget" not in backend.engine.chat.call_args.kwargs["sampling_params"]


def test_native_budget_cannot_consume_entire_generation(backend):
    backend.thinking_token_budget = 1024
    with pytest.raises(ValueError, match="leave room"):
        backend.invoke([HumanMessage(content="Choose")], enable_thinking=True, max_tokens=1024)
    backend.engine.chat.assert_not_called()


def test_brain_chain_auto_clamps_thinking_budget_when_budget_equals_max_tokens(backend):
    from arc_agent.chains.brain import BrainChain
    backend.thinking_token_budget = 1024
    backend.engine.chat.return_value[0].outputs[0].text = "<channel|>ACTION=ACTION1"
    brain = BrainChain(backend, max_tokens=1024, enable_thinking=True)
    res = brain._invoke("Choose", max_tokens=1024)
    assert res == "ACTION=ACTION1"
    params = backend.engine.chat.call_args.kwargs["sampling_params"]
    assert params["thinking_token_budget"] == 768 and params["max_tokens"] == 1024


def test_brain_chain_preserves_full_thinking_budget_when_max_tokens_is_1536(backend):
    from arc_agent.chains.brain import BrainChain
    backend.thinking_token_budget = 1024
    backend.engine.chat.return_value[0].outputs[0].text = "<channel|>ACTION=ACTION1"
    brain = BrainChain(backend, max_tokens=1536, enable_thinking=True)
    res = brain._invoke("Choose", max_tokens=1536)
    assert res == "ACTION=ACTION1"
    params = backend.engine.chat.call_args.kwargs["sampling_params"]
    assert params["thinking_token_budget"] == 1024 and params["max_tokens"] == 1536



def test_unsupported_thinking_budget_fails_before_gpu_load(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"model_type": "gemma4"}')
    engine = Mock()
    def old_sampling_params(max_tokens):
        return {"max_tokens": max_tokens}
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=engine, SamplingParams=old_sampling_params))
    monkeypatch.setitem(sys.modules, "vllm.config", SimpleNamespace(ReasoningConfig=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoProcessor=SimpleNamespace(
        from_pretrained=lambda *a, **kw: SimpleNamespace(image_processor=object(), chat_template="native"))))
    with pytest.raises(RuntimeError, match="lacks native thinking-budget support"):
        load_vllm(ModelConfig(backend="vllm"), str(tmp_path))
    engine.assert_not_called()
