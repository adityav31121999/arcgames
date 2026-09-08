"""Optional prompt sections that can be compacted without touching task inputs."""

from typing import Any


class ContextBudgetError(ValueError):
    """Required instructions, images, and output reservation cannot fit."""


def context_part(label: str, body: str, priority: int = 3, keep: str = "tail") -> dict[str, Any]:
    return {
        "type": "text", "text": f"\n\n{label}:\n{body}\n",
        "context_label": label, "context_body": body,
        "context_priority": priority, "context_keep": keep,
    }


def compact_context(messages: list[dict]) -> str | None:
    """Reduce the lowest-priority optional section by whole lines. Never edit images."""
    candidates = [
        (message, part) for message in messages for part in message["content"]
        if "context_priority" in part
    ]
    if not candidates:
        return None
    message, part = max(candidates, key=lambda pair: (
        pair[1]["context_priority"], len(pair[1]["context_body"]),
    ))
    label = part["context_label"]
    lines = part["context_body"].splitlines(keepends=True)
    if len(lines) <= 1:
        message["content"].remove(part)
    else:
        count = max(1, len(lines) // 2)
        kept = lines[:count] if part["context_keep"] == "head" else lines[-count:]
        part["context_body"] = "".join(kept)
        part["text"] = f"\n\n{label} (compacted):\n{part['context_body']}\n"
    return label
