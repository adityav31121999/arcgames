"""In-process vLLM chat backend; vLLM owns all model weights and GPU execution."""

import base64
import io
import re
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from .gemma_transformers import GemmaTransformersChatModel, _sanitize_llm_text


class VLLMChatModel(GemmaTransformersChatModel):
    engine: Any = Field(default=None)
    thinking_active: bool = False
    last_generation_info: dict = Field(default_factory=dict)
    thinking_token_budget: int | None = None

    @property
    def _llm_type(self):
        return "vllm_chat_model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from vllm import SamplingParams

        formatted, images = self._extract_images_and_text(messages)
        reserve = kwargs.get("max_new_tokens", kwargs.get("max_tokens", 128))
        self.thinking_active = kwargs.get("enable_thinking", False)
        thinking_budget = kwargs.get("thinking_token_budget", self.thinking_token_budget)
        sampling_options = {}
        if self.thinking_active and thinking_budget is not None:
            if not isinstance(thinking_budget, int) or not 0 < thinking_budget < reserve:
                raise ValueError("thinking_token_budget must be positive and smaller than max_tokens to leave room for the final answer.")
            sampling_options["thinking_token_budget"] = thinking_budget
        # CPU processor expansion preserves the existing context compaction rules.
        # No Transformers model is loaded and no input tensors are moved to CUDA.
        self._prepare_inputs(formatted, images, reserve)
        image_iter = iter(images)
        chat = []
        for message in formatted:
            content = []
            for part in message["content"]:
                if part["type"] == "image":
                    buffer = io.BytesIO()
                    next(image_iter).save(buffer, format="PNG")
                    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
                    content.append({"type": "image_url", "image_url": {"url": url}})
                else:
                    content.append({"type": "text", "text": part["text"]})
            chat.append({"role": message["role"], "content": content})
        params = SamplingParams(
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            repetition_penalty=kwargs.get("repetition_penalty", self.repeat_penalty),
            max_tokens=reserve, stop=None if self.thinking_active else stop or None,
            skip_special_tokens=not self.thinking_active,
            **sampling_options,
        )
        result = self.engine.chat(chat, sampling_params=params, use_tqdm=False,
                                  chat_template_kwargs={"enable_thinking": self.thinking_active})
        raw = result[0].outputs[0].text.strip()
        output = result[0].outputs[0]
        self.last_generation_info = {
            "finish_reason": getattr(output, "finish_reason", None),
            "generated_tokens": len(getattr(output, "token_ids", []) or []),
            "max_tokens": reserve,
            "thinking_enabled": self.thinking_active,
            "thinking_token_budget": sampling_options.get("thinking_token_budget"),
        }
        if self.thinking_active:
            # Gemma's thought channel is internal deliberation, not an action or belief update.
            # A budget exhausted inside that channel must never be parsed as a decision.
            if "<channel|>" not in raw and ("<|channel>" in raw or getattr(result[0].outputs[0], "finish_reason", None) == "length"):
                raise RuntimeError(f"Thinking budget exhausted before a final answer: {self.last_generation_info}")
            if "<channel|>" in raw:
                raw = raw.rsplit("<channel|>", 1)[1]
            raw = re.sub(r"<\|[^>]*>|<[^<\s]*\|>", "", raw).strip()
            for marker in stop or []:
                raw = raw.split(marker, 1)[0].strip()
        text = raw if kwargs.get("raw_output", False) else _sanitize_llm_text(
            raw, action_response=kwargs.get("action_response", False))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def load_vllm(config, model_id):
    from pathlib import Path
    import os
    import json

    if not Path(model_id).is_dir():
        raise RuntimeError("vLLM offline mode requires a local checkpoint directory.")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError(
            f"vLLM dependency import failed before model loading: {type(exc).__name__}: {exc}. "
            "After installing wheels, restart the Python kernel and continue at the runtime cell "
            "without rerunning installation. If a fresh Python process also fails, repair the "
            "dependency installation using the resolved wheelhouse."
        ) from exc
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True,
                                              trust_remote_code=config.trust_remote_code)
    if not getattr(processor, "image_processor", None) or not getattr(processor, "chat_template", None):
        raise RuntimeError("vLLM Eye/Brain requires the checkpoint's multimodal processor and chat template.")
    metadata = json.loads((Path(model_id) / "config.json").read_text(encoding="utf-8"))
    reasoning_options = {}
    thinking_budget = None
    if str(metadata.get("model_type", "")).startswith("gemma4"):
        # Configure native reasoning boundaries before creating the GPU engine.
        # max_tokens alone can be consumed entirely by thoughts, leaving no action.
        try:
            from vllm import SamplingParams
            from vllm.config import ReasoningConfig
            SamplingParams(max_tokens=2048, thinking_token_budget=1024)
            reasoning_options["reasoning_config"] = ReasoningConfig(
                reasoning_start_str="<|channel>thought\n", reasoning_end_str="<channel|>",
            )
        except (ImportError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "This vLLM wheelhouse lacks native thinking-budget support. "
                "Use a compatible vLLM build with SamplingParams.thinking_token_budget "
                "and ReasoningConfig before loading Gemma 4."
            ) from exc
        thinking_budget = 1024
    limit_mm = getattr(config, "limit_mm_per_prompt", None) or {"image": 2, "video": 0}
    engine = LLM(
        model=model_id, tokenizer=model_id, trust_remote_code=config.trust_remote_code,
        dtype="auto", tensor_parallel_size=1, max_model_len=config.max_context_length,
        gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        max_num_seqs=1, limit_mm_per_prompt=limit_mm,
        enforce_eager=config.vllm_enforce_eager,
        **reasoning_options,
    )
    # Record checkpoint metadata without loading a second model.
    from types import SimpleNamespace
    return VLLMChatModel(engine=engine, model=SimpleNamespace(config=metadata),
                         thinking_token_budget=thinking_budget,
                         processor=processor, max_context_length=config.max_context_length,
                         temperature=config.temperature, top_p=config.top_p,
                         repeat_penalty=config.repeat_penalty)
