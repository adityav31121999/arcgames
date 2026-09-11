"""Shared multimodal message assembly and bounded, validated stage invocation."""

from dataclasses import dataclass
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from ..core.context import ContextBudgetError


@dataclass
class StageResult:
    ok: bool
    text: str = ""
    error: str = ""
    attempts: int = 0


def normalize_labels(text):
    # Accept presentation-only Markdown without relaxing field requirements.
    text = re.sub(r"(?m)^[ \t]*(?:#{1,6}[ \t]+)?(?:[-*][ \t]+)?\*\*([^*\n:]+):?\*\*[ \t]*:?[ \t]*", r"\1: ", text)
    return text


def build_messages(system: str, prompt: str, images=(), context=()):
    parts = [{"type": "text", "text": prompt}]
    for label, image in images:
        if image is not None:
            if label:
                parts.append({"type": "text", "text": f"\n{label}\n"})
            parts.append({"type": "image", "image": image})
    parts.extend(context)
    return [SystemMessage(content=system), HumanMessage(content=parts)]


def invoke_stage(model, system: str, prompt: str, *, stage: str, max_tokens: int,
                 required_labels=(), images=(), context=(), **kwargs: Any) -> StageResult:
    error = ""
    kwargs.setdefault("raw_output", True)
    for attempt in range(1, 3):
        reminder = ""
        if attempt > 1:
            reminder = "\nPrevious inference failed validation. Return non-empty labeled fields: " + ", ".join(required_labels)
        try:
            actual_max_tokens = max_tokens * (2 if attempt > 1 and "Thinking budget exhausted" in error else 1)
            call_kwargs = dict(kwargs)
            if call_kwargs.get("enable_thinking"):
                model_budget = getattr(model, "thinking_token_budget", None)
                if model_budget is not None and "thinking_token_budget" not in call_kwargs and model_budget >= actual_max_tokens:
                    call_kwargs["thinking_token_budget"] = max(64, actual_max_tokens - 256)
            response = model.invoke(
                build_messages(system, prompt + reminder, images, context),
                max_tokens=actual_max_tokens, **call_kwargs,
            )
            text = normalize_labels(str(response.content).strip())
            valid = bool(text) and "INFERENCE FAILED" not in text
            for label in required_labels:
                # RULES is the only block field; other fields need an inline value.
                pattern = (r"(?mi)^\s*RULES:\s*\n\s*[-*]\s+\S" if label == "RULES"
                           else r"(?mi)^\s*(?:[-*]\s*)?" + re.escape(label) + r":[ \t]*\S")
                valid = valid and bool(re.search(pattern, text))
            if valid:
                return StageResult(True, text=text, attempts=attempt)
            error = "Empty or malformed labeled response; final answer preview=" + repr(text[:500])
        except ContextBudgetError as exc:
            # Optional context was already compacted by the wrapper; retrying cannot help.
            return StageResult(False, error=f"{stage}: {exc}", attempts=attempt)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    print(f"[{stage}] inference unavailable after 2 attempts: {ascii(error)}")
    return StageResult(False, error=f"{stage}: {error}", attempts=2)
