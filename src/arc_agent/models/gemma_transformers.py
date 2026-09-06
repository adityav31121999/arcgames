"""LangChain BaseChatModel implementation for Hugging Face Transformers models."""

from typing import Any, Dict, Iterator, List, Optional
import base64
import io
import re
from PIL import Image

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field


class GemmaTransformersChatModel(BaseChatModel):
    """LangChain ChatModel wrapper for Hugging Face Transformers models.

    Optimized for Gemma-4-26B-A4B-NVFP4 and multimodal vision-language architectures.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any = Field(default=None, description="The loaded AutoModel instance")
    processor: Any = Field(default=None, description="The loaded AutoProcessor or AutoTokenizer")
    device: str = Field(default="cuda:0")
    torch_dtype: str = Field(default="bfloat16")
    max_context_length: int = Field(default=8192)
    temperature: float = Field(default=0.1)
    top_p: float = Field(default=0.95)
    repeat_penalty: float = Field(default=1.05)


    @property
    def _llm_type(self) -> str:
        return "gemma_transformers_chat_model"

    def _extract_images_and_text(self, messages: List[BaseMessage]) -> tuple[List[Dict[str, Any]], List[Image.Image]]:
        """Parses LangChain messages into HuggingFace chat template format and extracted PIL images."""
        formatted_messages = []
        pil_images = []
        system_texts = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                if msg.content:
                    system_texts.append(str(msg.content))
            elif isinstance(msg, HumanMessage):
                if isinstance(msg.content, str):
                    formatted_messages.append({"role": "user", "content": [{"type": "text", "text": msg.content}]})
                elif isinstance(msg.content, list):
                    content_list = []
                    for item in msg.content:
                        if isinstance(item, str):
                            content_list.append({"type": "text", "text": item})
                        elif isinstance(item, dict):
                            item_type = item.get("type", "")
                            if item_type == "text":
                                content_list.append({"type": "text", "text": item.get("text", "")})
                            elif item_type in ("image_url", "image"):
                                img_obj = None
                                if "image" in item and isinstance(item["image"], Image.Image):
                                    img_obj = item["image"]
                                elif "image_url" in item:
                                    url = item["image_url"]
                                    if isinstance(url, dict):
                                        url = url.get("url", "")
                                    if url.startswith("data:image"):
                                        # Base64 data URI
                                        b64_str = url.split(",", 1)[-1]
                                        img_bytes = base64.b64decode(b64_str)
                                        img_obj = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                                    elif url.startswith("/") or "\\" in url or ":" in url:
                                        # Local file path
                                        img_obj = Image.open(url).convert("RGB")
                                
                                if img_obj is not None:
                                    pil_images.append(img_obj)
                                    content_list.append({"type": "image"})
                    formatted_messages.append({"role": "user", "content": content_list})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": [{"type": "text", "text": str(msg.content)}]})
            elif isinstance(msg, ChatMessage):
                formatted_messages.append({"role": msg.role, "content": [{"type": "text", "text": str(msg.content)}]})

        # Gemma 4 MoE NVFP4 bug fix: system prompts > 400-500 characters trigger immediate empty responses (1-3 tokens).
        # We split or merge: keep system role <= 150 chars, and prepend full detailed guidelines to the first user turn.
        if system_texts:
            full_system = "\n\n".join(system_texts)
            if len(full_system) > 400:
                short_sys = "You are an expert agent solving ARC-AGI-3 grid reasoning puzzles."
                formatted_messages.insert(0, {"role": "system", "content": [{"type": "text", "text": short_sys}]})
                # Prepend full system prompt into the first user message
                user_found = False
                for m in formatted_messages:
                    if m["role"] == "user":
                        for c in m["content"]:
                            if c.get("type") == "text":
                                c["text"] = f"[System Instructions]\n{full_system}\n\n[Task]\n{c['text']}"
                                user_found = True
                                break
                        if user_found:
                            break
                if not user_found:
                    formatted_messages.append({"role": "user", "content": [{"type": "text", "text": full_system}]})
            else:
                formatted_messages.insert(0, {"role": "system", "content": [{"type": "text", "text": full_system}]})

        return formatted_messages, pil_images

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        import torch

        if self.model is None or self.processor is None:
            raise RuntimeError("GemmaTransformersChatModel model or processor is not loaded.")

        formatted_messages, pil_images = self._extract_images_and_text(messages)

        # Build prompt using chat template
        if hasattr(self.processor, "apply_chat_template"):
            prompt_text = self.processor.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback for text tokenizers
            prompt_parts = []
            for m in formatted_messages:
                role = m["role"]
                text_content = " ".join(c["text"] for c in m["content"] if c.get("type") == "text")
                prompt_parts.append(f"<|im_start|>{role}\n{text_content}<|im_end|>")
            prompt_parts.append("<|im_start|>assistant\n")
            prompt_text = "\n".join(prompt_parts)

        # Prepare inputs with processor
        if pil_images and hasattr(self.processor, "image_processor"):
            inputs = self.processor(
                text=[prompt_text],
                images=pil_images,
                return_tensors="pt",
                padding=True,
            )
        else:
            inputs = self.processor(
                text=[prompt_text],
                return_tensors="pt",
                padding=True,
            )

        # Move tensors to device
        target_device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        max_new_tokens = kwargs.get("max_new_tokens", kwargs.get("max_tokens", 48))
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        repetition_penalty = kwargs.get("repetition_penalty", self.repeat_penalty)

        do_sample = temperature > 0.01

        # Real-time stopping criteria so generate() stops immediately on stop token / newline
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        stopping_criteria_list = None
        if stop and tokenizer is not None:
            from transformers import StoppingCriteria, StoppingCriteriaList

            stop_token_ids = []
            for s in stop:
                try:
                    tok_ids = tokenizer.encode(s, add_special_tokens=False)
                    if tok_ids:
                        stop_token_ids.append(tok_ids)
                except Exception:
                    pass

            if stop_token_ids:
                class CustomStopCriteria(StoppingCriteria):
                    def __init__(self, stop_sequences):
                        self.stop_sequences = [torch.tensor(s, dtype=torch.long) for s in stop_sequences]
                        self._device_sequences = None

                    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **c_kwargs) -> bool:
                        if self._device_sequences is None:
                            self._device_sequences = [s.to(input_ids.device) for s in self.stop_sequences]
                        cur_len = input_ids.shape[-1]
                        for seq in self._device_sequences:
                            s_len = seq.shape[0]
                            if cur_len >= s_len and torch.equal(input_ids[0, -s_len:], seq):
                                return True
                        return False

                stopping_criteria_list = StoppingCriteriaList([CustomStopCriteria(stop_token_ids)])

        with torch.inference_mode():
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "use_cache": True,
                "pad_token_id": (
                    lambda p: p[0] if isinstance(p, (list, tuple)) else (p if p is not None else 0)
                )(getattr(tokenizer, "pad_token_id", 0)),
            }
            if repetition_penalty and repetition_penalty > 1.0:
                generate_kwargs["repetition_penalty"] = repetition_penalty

            # Try native stop_strings first to delegate stop detection to C++/CUDA
            use_fallback_stopping = True
            if stop and tokenizer is not None:
                try:
                    generate_kwargs["stop_strings"] = stop
                    generate_kwargs["tokenizer"] = tokenizer
                    use_fallback_stopping = False
                except Exception:
                    pass

            if use_fallback_stopping and stopping_criteria_list is not None:
                generate_kwargs["stopping_criteria"] = stopping_criteria_list

            if do_sample:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = temperature
                generate_kwargs["top_p"] = top_p
            else:
                generate_kwargs["do_sample"] = False

            # Attempt accelerated generation with speculative prompt lookup
            output_ids = None
            if not pil_images:
                try:
                    fast_kwargs = dict(generate_kwargs)
                    fast_kwargs["prompt_lookup_num_tokens"] = 3
                    output_ids = self.model.generate(**inputs, **fast_kwargs)
                except Exception:
                    output_ids = None

            if output_ids is None:
                output_ids = self.model.generate(**inputs, **generate_kwargs)

        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[-1]
        out = output_ids[0]

        # Determine generated tokens safely (handling multimodal prefixes, image tokens, and left-padding)
        if out.shape[-1] > input_len and (out[:input_len] == input_ids[0]).all():
            generated_ids = out[input_len:]
        elif out.shape[-1] <= input_len:
            # Model output only generated tokens
            generated_ids = out
        else:
            # Check prefix overlap between prompt input_ids and generated output
            prefix_match_len = 0
            while (
                prefix_match_len < input_len
                and prefix_match_len < out.shape[-1]
                and out[prefix_match_len] == input_ids[0][prefix_match_len]
            ):
                prefix_match_len += 1
            if prefix_match_len > 0:
                generated_ids = out[prefix_match_len:]
            else:
                generated_ids = out[input_len:] if out.shape[-1] > input_len else out

        decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Handle stop sequences cleanup if specified
        if stop:
            for s in stop:
                if s in decoded_text:
                    decoded_text = decoded_text.split(s)[0].strip()

        # Sanitize against degenerate phrase repetition (e.g. repeated token loops)
        words = decoded_text.split()
        if len(words) > 10:
            for phrase_len in (1, 2, 3):
                repeated_streak = 0
                max_streak = 0
                for i in range(phrase_len, len(words), phrase_len):
                    if words[i : i + phrase_len] == words[i - phrase_len : i]:
                        repeated_streak += 1
                        max_streak = max(max_streak, repeated_streak)
                    else:
                        repeated_streak = 0
                if max_streak >= 5:
                    print(f"⚠️ [LLM SANITIZER] Detected degenerate repetition streak (x{max_streak}), truncating.")
                    decoded_text = " ".join(words[:12])
                    break

        # Debug logging for raw model generation to diagnose tokenization / corruption issues
        import os
        if (
            os.getenv("DEBUG_LLM_OUTPUT", "false").lower() in ("true", "1")
            or os.getenv("DEBUG", "false").lower() in ("true", "1")
            or len(decoded_text) == 0
        ):
            preview = repr(decoded_text[:150]) if decoded_text else "<EMPTY>"
            print(f"🔍 [LLM RAW RESPONSE] tokens={len(generated_ids)} | text={preview}")

        message = AIMessage(content=decoded_text)
        return ChatResult(generations=[ChatGeneration(message=message)])


class MockChatModel(BaseChatModel):
    """Mock ChatModel for fast local unit testing and dry runs without GPU."""

    mock_responses: List[str] = Field(default_factory=list)
    call_count: int = Field(default=0)

    @property
    def _llm_type(self) -> str:
        return "mock_chat_model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.call_count += 1
        if self.mock_responses:
            idx = (self.call_count - 1) % len(self.mock_responses)
            resp = self.mock_responses[idx]
        else:
            # Check prompt content to generate reasonable mock responses
            last_msg = messages[-1].content if messages else ""
            last_text = str(last_msg)

            if "Synthesize a ONE-SHOT plan" in last_text:
                resp = "ACTION=ACTION1\nACTION=ACTION4\nACTION=ACTION1"
            elif "Legal actions" in last_text or "Next action:" in last_text:
                resp = "ACTION=ACTION1"
            elif "Analyse the initial PNG" in last_text or "PROMPT_ASSUME" in last_text:
                resp = "The blue shape is the player. The goal is to reach the green tile."
            elif "EXPECTED:" in last_text or "DIVERGED:" in last_text:
                resp = "EXPECTED: Wall collision resulted in NO-OP.\nRecommend turning right."
            else:
                resp = "ACTION=ACTION1"

        message = AIMessage(content=resp)
        return ChatResult(generations=[ChatGeneration(message=message)])
