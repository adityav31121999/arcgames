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


def test_vllm_loader_uses_local_checkpoint_and_no_transformers_weights(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
    llm = Mock(return_value=Mock())
    processor = SimpleNamespace(image_processor=object(), chat_template="native")
    auto = SimpleNamespace(from_pretrained=Mock(return_value=processor))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=llm))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoProcessor=auto))
    wrapper = load_vllm(ModelConfig(backend="vllm", max_context_length=32768), str(tmp_path))
    assert wrapper.engine is llm.return_value
    assert llm.call_args.kwargs["tensor_parallel_size"] == 1
    assert llm.call_args.kwargs["limit_mm_per_prompt"] == {"image": 2}
    assert llm.call_args.kwargs["max_model_len"] == 32768
    assert auto.from_pretrained.call_args.kwargs["local_files_only"] is True
