"""LangChain BaseChatModel implementation for Hugging Face Transformers models."""

from typing import Any, Dict, Iterator, List, Optional
import base64
import io
import re
import sys
import unicodedata
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
from ..core.context import ContextBudgetError, compact_context


# ---------------------------------------------------------------------------
# Token corruption / degeneration constants
# ---------------------------------------------------------------------------
# Minimum ASCII printable chars to consider a response non-empty after stripping
_MIN_RESPONSE_CHARS = 3
# Foreign script ratio above which we strip and attempt rescue
_FOREIGN_RATIO_THRESHOLD = 0.10  # lowered from 0.15 → catches 50/231 case sooner
# Repetition streak length that triggers truncation
_REPETITION_STREAK_THRESHOLD = 4
# Truncation word count when repetition detected
_REPETITION_TRUNCATE_WORDS = 12


def _log_llm(message: str) -> None:
    """Keep diagnostics printable even on legacy Windows output streams."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(message.encode(encoding, errors="backslashreplace").decode(encoding))


def _is_all_whitespace_or_special(text: str) -> bool:
    """Returns True if text is empty, all whitespace, or only special/punctuation chars."""
    stripped = text.strip()
    if not stripped:
        return True
    # Only punctuation / symbols left after stripping letters and digits
    if not any(ch.isalnum() for ch in stripped):
        return True
    return False


def _rescue_action_from_corrupted_text(text: str) -> str:
    """Last-resort extraction: try to recover a valid ACTION= line from corrupted/partial output.

    Accept explicit assignments or standalone choices, never incidental prose mentions.
    Conflicting choices and incomplete click coordinates must be retried.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)

    aliases = {
        "UP": "ACTION1", "DOWN": "ACTION2", "LEFT": "ACTION3",
        "RIGHT": "ACTION4", "CLICK": "ACTION6", "UNDO": "ACTION7",
        "1": "ACTION1", "2": "ACTION2", "3": "ACTION3",
        "4": "ACTION4", "5": "ACTION5", "6": "ACTION6", "7": "ACTION7",
        "0": "RESET", "RESET": "RESET", "INTERACT": "ACTION5",
        "SELECT": "ACTION5", "EXECUTE": "ACTION5", "MOUSE": "ACTION6",
    }
    for number in range(1, 8):
        aliases[f"ACTION{number}"] = f"ACTION{number}"
        aliases[f"ACTION_{number}"] = f"ACTION{number}"

    choices = set()
    for line in text.splitlines():
        action_match = re.search(
            r"\bACTION\*{0,2}\s*[:=]\s*([A-Za-z0-9_]+)\b(.*)$",
            line.strip(),
            re.IGNORECASE,
        )
        if action_match is None:
            action_match = re.fullmatch(
                r"\*{0,2}([A-Za-z0-9_]+)\*{0,2}([ \t]*(?:(?:X\s*[:=].*)|(?:[\[(].*))?)",
                line.strip(),
                re.IGNORECASE,
            )
        if action_match is None:
            continue
        name = aliases.get(action_match.group(1).upper())
        if name is None:
            continue
        rest = action_match.group(2)
        # Only use coordinates belonging to this choice, not another line's target.
        coord_m = (
            re.search(r"\bX\s*[:=]\s*(-?\d+)[ \t,;]+Y\s*[:=]\s*(-?\d+)\b", rest, re.IGNORECASE)
            or re.search(r"[(\[]\s*(-?\d+)\s*[, ]\s*(-?\d+)\s*[)\]]", rest)
        )
        if name == "ACTION6" and (coord_m is None or any(int(c) < 0 for c in coord_m.groups())):
            return ""
        choice = f"ACTION={name}"
        if coord_m:
            choice += f" X={int(coord_m.group(1))} Y={int(coord_m.group(2))}"
        choices.add(choice)
    return choices.pop() if len(choices) == 1 else ""


def _sanitize_llm_text(text: str, *, action_response: bool = False) -> str:
    """Sanitizes model output to prevent non-English script drift, repetition loops, and control chars.

    Enhanced to:
    - Apply a lower foreign script ratio threshold (0.10 vs old 0.15) to catch marginal cases.
    - Attempt last-resort ACTION= rescue from corrupted output before discarding.
    - Explicitly handle EOS / padding token artefacts that appear as unicode replacement chars.
    """
    if not text:
        return ""

    # Normalize fullwidth action syntax/digits before removing foreign scripts.
    normalized = unicodedata.normalize("NFKC", text)
    # 1. Remove non-printable control characters except newline and tab
    clean = "".join(ch for ch in normalized if ch in ("\n", "\r", "\t") or (ord(ch) >= 32 and ord(ch) != 127))

    # 1b. Remove unicode replacement characters (U+FFFD) — EOS/pad token artefacts
    clean = clean.replace("\ufffd", "")
    # Also remove null bytes that some decoders emit for pad tokens
    clean = clean.replace("\x00", "")

    # 2. Check for non-Latin / non-ASCII foreign script flooding (CJK, Cyrillic, Arabic, Hangul, etc.)
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
            or "\u3000" <= ch <= "\u303f"  # CJK Symbols and Punctuation
            or "\uff01" <= ch <= "\uffee"  # Halfwidth and Fullwidth Forms
        )
    )
    total_letters = sum(1 for ch in clean if ch.isalpha())

    if foreign_chars > 0:
        _log_llm(f"[LLM SANITIZER] Raw text preview before stripping: {text[:500]!r}")
        ratio = (foreign_chars / total_letters) if total_letters > 0 else 1.0
        if ratio > _FOREIGN_RATIO_THRESHOLD:
            _log_llm(
                f"⚠️ [LLM SANITIZER] High ratio of non-English/foreign script detected "
                f"({foreign_chars}/{total_letters}, ratio={ratio:.2f}). Stripping foreign characters."
            )
        # Always strip regardless of ratio to keep output clean
        clean = re.sub(
            r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff\u0900-\u097f\u3000-\u303f\uff01-\uffee]+",
            "",
            clean,
        )

    clean = clean.strip()

    # 3. Rescue corrupted actions even when readable plan fragments remain.
    # Match the mapper's action syntax, including aliases and markdown bolding.
    has_action = re.search(
        r"\bACTION\*{0,2}\s*[:=]\s*([A-Za-z0-9_]+)\b",
        clean,
        re.IGNORECASE,
    ) is not None
    if action_response and (len(clean) < _MIN_RESPONSE_CHARS or not has_action):
        rescued = _rescue_action_from_corrupted_text(text)
        if rescued:
            _log_llm(f"🔧 [LLM SANITIZER] Rescued ACTION token from corrupted output: {rescued!r}")
            return rescued
        return ""  # An ambiguous or missing choice must trigger the caller's retry.
    if len(clean) < _MIN_RESPONSE_CHARS:
        return ""

    # 4. Sanitize against degenerate phrase repetition (e.g. repeated token loops)
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
            if max_streak >= _REPETITION_STREAK_THRESHOLD:
                # Attempt to rescue the head of the text before repetition kicks in
                head_words = words[:_REPETITION_TRUNCATE_WORDS]
                truncated = " ".join(head_words)
                # If the head itself contains an ACTION= line, keep it; otherwise rescue
                if "ACTION" in truncated.upper():
                    _log_llm(
                        f"⚠️ [LLM SANITIZER] Detected degenerate repetition streak (x{max_streak}), "
                        f"truncating to first {_REPETITION_TRUNCATE_WORDS} words."
                    )
                    clean = truncated
                else:
                    rescued = _rescue_action_from_corrupted_text(clean) if action_response else ""
                    if rescued:
                        _log_llm(
                            f"⚠️ [LLM SANITIZER] Repetition loop detected (x{max_streak}); "
                            f"rescued ACTION token: {rescued!r}"
                        )
                        clean = rescued
                    else:
                        _log_llm(
                            f"⚠️ [LLM SANITIZER] Detected degenerate repetition streak (x{max_streak}), "
                            f"truncating."
                        )
                        clean = truncated
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
    repeat_penalty: float = Field(default=1.0)
    last_context_usage: Dict[str, Any] = Field(default_factory=dict)

    def _context_limit(self) -> int:
        limits = [self.max_context_length]
        config = getattr(self.model, "config", None)
        text_config = config.get("text_config") if isinstance(config, dict) else getattr(config, "text_config", None)
        for candidate in (text_config, config):
            value = candidate.get("max_position_embeddings") if isinstance(candidate, dict) else getattr(candidate, "max_position_embeddings", None)
            if isinstance(value, int) and value > 0:
                limits.append(value)
        return min(limits)

    @property
    def _llm_type(self) -> str:
        return "gemma_transformers_chat_model"

    def _validate_context_length(self, input_tokens: int, output_tokens: int) -> None:
        if input_tokens + output_tokens > self._context_limit():
            raise ContextBudgetError(
                f"Context requires {input_tokens} input + {output_tokens} output tokens, "
                f"exceeding the effective {self._context_limit()}-token maximum. "
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
                                content_list.append(dict(item))
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
            formatted_messages.insert(0, {
                "role": "system", "content": [{"type": "text", "text": "\n\n".join(system_texts)}],
            })

        return formatted_messages, pil_images

    def _prepare_inputs(self, formatted_messages, pil_images, max_new_tokens):
        if not hasattr(self.processor, "apply_chat_template"):
            raise RuntimeError("The model processor must provide its checkpoint's chat template.")
        if pil_images and getattr(self.processor, "image_processor", None) is None:
            raise RuntimeError("Images were supplied but the loaded processor cannot process images.")
        compacted = []
        while True:
            prompt_text = self.processor.apply_chat_template(
                formatted_messages, tokenize=False, add_generation_prompt=True,
            )
            processor_kwargs = dict(text=[prompt_text], return_tensors="pt", padding=True,
                                    add_special_tokens=False)
            if pil_images:
                # All images belong to the single conversation in this batch.
                processor_kwargs["images"] = [pil_images]
            inputs = self.processor(**processor_kwargs)
            input_tokens = inputs["input_ids"].shape[-1]
            if input_tokens + max_new_tokens <= self._context_limit():
                break
            removed = compact_context(formatted_messages)
            if removed is None:
                self._validate_context_length(input_tokens, max_new_tokens)
            compacted.append(removed)

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        ids = inputs["input_ids"][0].tolist()
        image_id = getattr(tokenizer, "image_token_id", None)
        image_tokens = ids.count(image_id) if isinstance(image_id, int) else None
        if pil_images:
            if not any(key in inputs for key in ("pixel_values", "pixel_values_images", "image_embeds")):
                raise RuntimeError("Processor produced no image tensors for the attached images.")
            if isinstance(image_id, int) and not image_tokens:
                raise RuntimeError("Processor produced no image token slots for attached images.")
            counts = []
            for name in ("boi_token", "eoi_token"):
                token = getattr(tokenizer, name, None)
                if isinstance(token, str):
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    counts.append(ids.count(token_id))
            if counts and (min(counts) < len(pil_images) or len(set(counts)) != 1):
                raise RuntimeError("Image start/end markers do not match the attached images.")
        self.last_context_usage = {
            "input_tokens": input_tokens, "image_tokens": image_tokens,
            "text_and_control_tokens": input_tokens - (image_tokens or 0),
            "output_reserved": max_new_tokens, "context_limit": self._context_limit(),
            "images": len(pil_images), "compacted_sections": sorted(set(compacted)),
        }
        import os
        if compacted or os.getenv("DEBUG_LLM_CONTEXT", "false").lower() in ("true", "1"):
            _log_llm(f"[LLM CONTEXT] {self.last_context_usage}")
        return inputs

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

        max_new_tokens = kwargs.get("max_new_tokens", kwargs.get("max_tokens", 48))
        inputs = self._prepare_inputs(formatted_messages, pil_images, max_new_tokens)
        target_device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

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
            import os
            enable_lookup = os.getenv("ENABLE_PROMPT_LOOKUP", "false").lower() in ("true", "1")
            if not pil_images and enable_lookup:
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

        # Sanitize text (enhanced: detects corruption, rescues ACTION tokens, strips replacement chars)
        decoded_text = (raw_decoded if kwargs.get("raw_output", False) else
                        _sanitize_llm_text(raw_decoded, action_response=kwargs.get("action_response", False)))

        import os
        if (
            os.getenv("DEBUG_LLM_OUTPUT", "false").lower() in ("true", "1")
            or os.getenv("DEBUG", "false").lower() in ("true", "1")
            or len(decoded_text) == 0
        ):
            preview = repr(raw_decoded[:500]) if raw_decoded else "<EMPTY>"
            _log_llm(f"🔍 [LLM RAW RESPONSE] tokens={len(generated_ids)} | text={preview}")

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
                resp = "Plan: Test upward movement.\nExpected effect: Player moves upward.\nACTION=ACTION1"
            elif "Evaluate the latest action result" in last_text:
                resp = "Recent findings: Observed the action result.\nOpen questions: Is the goal hypothesis correct?\nPlan: Test an alternative."
            elif "Analyze the visual change" in last_text:
                resp = "Recent findings: Compared the before and after boards.\nOpen questions: Does the change advance the goal?"
            elif "An attempt at this level just ended" in last_text:
                resp = "FAILURE_REASON: Exploration did not establish the goal.\nRULES:\n- Test an alternative and compare the result."
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
