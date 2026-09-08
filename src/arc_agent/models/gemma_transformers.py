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


def _sanitize_llm_text(text: str) -> str:
    """Sanitizes model output to prevent non-English script drift, repetition loops, and control chars."""
    if not text:
        return ""

    # 1. Remove non-printable control characters except newline and tab
    clean = "".join(ch for ch in text if ch in ("\n", "\r", "\t") or (ord(ch) >= 32 and ord(ch) != 127))

    # 2. Check for non-Latin / non-ASCII foreign script flooding (CJK, Cyrillic, Arabic, Hangul, etc.)
    # Count characters in foreign script unicode blocks
    foreign_chars = sum(
        1
        for ch in clean
        if (
            "\u4e00" <= ch <= "\u9fff"  # CJK Unified Ideographs
            or "\u3040" <= ch <= "\u30ff"  # Hiragana / Katakana
            or "\uac00" <= ch <= "\ud7af"  # Hangul
            or "\u0400" <= ch <= "\u04ff"  # Cyrillic
            or "\u0600" <= ch <= "\u06ff"  # Arabic
            or "\u0900" <= ch <= "\u097f"  # Devanagari
        )
    )
    total_letters = sum(1 for ch in clean if ch.isalpha())

    # If foreign script dominates or is mixed into output, filter or reject it
    if foreign_chars > 0:
        if total_letters > 0 and (foreign_chars / total_letters) > 0.15:
            print(f"⚠️ [LLM SANITIZER] High ratio of non-English/foreign script detected ({foreign_chars}/{total_letters}). Stripping foreign characters.")
            # Strip non-ASCII/foreign characters
            clean = re.sub(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff\u0900-\u097f]+", "", clean)
        else:
            clean = re.sub(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff\u0900-\u097f]+", "", clean)

    clean = clean.strip()

    # 3. Sanitize against degenerate phrase repetition (e.g. repeated token loops)
    words = clean.split()
    if len(words) > 8:
        for phrase_len in (1, 2, 3, 4):
            repeated_streak = 0
            max_streak = 0
            for i in range(phrase_len, len(words), phrase_len):
                if words[i : i + phrase_len] == words[i - phrase_len : i]:
                    repeated_streak += 1
                    max_streak = max(max_streak, repeated_streak)
                else:
                    repeated_streak = 0
            if max_streak >= 4:
                print(f"⚠️ [LLM SANITIZER] Detected degenerate repetition streak (x{max_streak}), truncating.")
                clean = " ".join(words[:12])
                break

    return clean.strip()


class GemmaTransformersChatModel(BaseChatModel):
    """LangChain ChatModel wrapper for Hugging Face Transformers models.

    Optimized for Gemma-4-26B-A4B-NVFP4 and multimodal vision-language architectures.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any = Field(default=None, description="The loaded AutoModel instance")
    processor: Any = Field(default=None, description="The loaded AutoProcessor or AutoTokenizer")
    device: str = Field(default="cuda:0")
    torch_dtype: str = Field(default="bfloat16")
    max_context_length: int = Field(default=81930)
    temperature: float = Field(default=0.1)
    top_p: float = Field(default=0.95)
    repeat_penalty: float = Field(default=1.05)

    @property
    def _llm_type(self) -> str:
        return "gemma_transformers_chat_model"

    def _validate_context_length(self, input_tokens: int, output_tokens: int) -> None:
        if input_tokens + output_tokens > self.max_context_length:
            raise ValueError(
                f"Context requires {input_tokens} input + {output_tokens} output tokens, "
                f"exceeding the configured {self.max_context_length}-token maximum. "
                "No context was silently truncated."
            )

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
                                        b64_str = url.split(",", 1)[-1]
                                        img_bytes = base64.b64decode(b64_str)
                                        img_obj = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                                    elif url.startswith("/") or "\\" in url or ":" in url:
                                        img_obj = Image.open(url).convert("RGB")

                                if img_obj is not None:
                                    pil_images.append(img_obj)
                                    content_list.append({"type": "image"})
                    formatted_messages.append({"role": "user", "content": content_list})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": [{"type": "text", "text": str(msg.content)}]})
            elif isinstance(msg, ChatMessage):
                formatted_messages.append({"role": msg.role, "content": [{"type": "text", "text": str(msg.content)}]})

        if system_texts:
            full_system = "\n\n".join(system_texts)
            if len(full_system) > 400:
                short_sys = "You are an expert agent solving ARC-AGI-3 grid reasoning puzzles in English."
                formatted_messages.insert(0, {"role": "system", "content": [{"type": "text", "text": short_sys}]})
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

        if hasattr(self.processor, "apply_chat_template"):
            prompt_text = self.processor.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            raise RuntimeError("The model processor must provide its checkpoint's chat template.")

        if pil_images and not hasattr(self.processor, "image_processor"):
            raise RuntimeError("Images were supplied but the loaded processor cannot process images.")

        if pil_images and hasattr(self.processor, "image_processor"):
            inputs = self.processor(
                text=[prompt_text],
                images=pil_images,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
        else:
            inputs = self.processor(
                text=[prompt_text],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )

        target_device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        max_new_tokens = kwargs.get("max_new_tokens", kwargs.get("max_tokens", 48))
        self._validate_context_length(inputs["input_ids"].shape[-1], max_new_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        repetition_penalty = kwargs.get("repetition_penalty", self.repeat_penalty)

        do_sample = temperature > 0.01

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

        input_ids = inputs.get("input_ids")
        input_len = input_ids.shape[-1] if input_ids is not None else 0
        out = output_ids[0]

        # Decoder-only generation returns the prompt followed by new tokens.
        generated_ids = out[input_len:]

        raw_decoded = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()

        # Handle stop sequences cleanup if specified
        if stop:
            for s in stop:
                if s in raw_decoded:
                    raw_decoded = raw_decoded.split(s)[0].strip()

        # Sanitize text
        decoded_text = _sanitize_llm_text(raw_decoded)

        import os
        if (
            os.getenv("DEBUG_LLM_OUTPUT", "false").lower() in ("true", "1")
            or os.getenv("DEBUG", "false").lower() in ("true", "1")
            or len(decoded_text) == 0
        ):
            preview = repr(raw_decoded[:500]) if raw_decoded else "<EMPTY>"
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
            last_msg = messages[-1].content if messages else ""
            last_text = str(last_msg)

            if "Synthesize a ONE-SHOT plan" in last_text:
                resp = "ACTION=ACTION1\nACTION=ACTION4\nACTION=ACTION1"
            elif "Legal actions" in last_text or "Next action:" in last_text or "best target coordinates" in last_text:
                resp = "Plan: Click interactive target.\nACTION=ACTION1"
            elif (
                "initial visual layout" in last_text
                or "PROMPT_ASSUME" in last_text
                or "World model:" in last_text
                or "Compare this level" in last_text
            ):
                resp = (
                    "World model: 2D grid puzzle with blue player and green target.\n"
                    "Goal model: Move player to reach the green goal block.\n"
                    "Action model: ACTION1 through ACTION4 provide directional movement.\n"
                    "Recent findings: Initial layout verified.\n"
                    "Plan: Move up toward target."
                )
            elif "EXPECTED:" in last_text or "DIVERGED:" in last_text:
                resp = "EXPECTED: Wall collision resulted in NO-OP.\nRecommend turning right."
            else:
                resp = "ACTION=ACTION1"

        message = AIMessage(content=resp)
        return ChatResult(generations=[ChatGeneration(message=message)])
