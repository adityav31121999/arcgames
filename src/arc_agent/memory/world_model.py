"""Turn-persistent structured world model — goal, plan, action model, and findings."""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List

_LABELS = [
    "World model",
    "Goal model",
    "Action model",
    "Recent findings",
    "Open questions",
    "Plan",
    "Cross-level notes",
]
_MAX_FIELD_CHARS = 300


def _extract_labeled_blocks(content: str, labels: list[str]) -> Dict[str, str]:
    normalized = {label.lower(): label for label in labels}
    targets = tuple(f"{label.lower()}:" for label in labels)
    extracted: Dict[str, List[str]] = {label: [] for label in labels}
    current_label: str | None = None

    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        candidate = stripped
        while candidate.startswith(("-", "*")):
            candidate = candidate[1:].lstrip()
        lowered = candidate.lower()

        matched_label: str | None = None
        inline_value = ""
        for target in targets:
            if lowered.startswith(target):
                matched_label = normalized[target[:-1]]
                inline_value = candidate[len(target):].strip()
                break

        if matched_label is not None:
            current_label = matched_label
            if inline_value:
                extracted[current_label].append(inline_value)
            continue

        if current_label is not None and stripped:
            extracted[current_label].append(stripped)

    result: Dict[str, str] = {}
    for label, lines in extracted.items():
        joined = " ".join(" ".join(lines).split()).strip()
        if joined:
            key = label.lower().replace(" ", "_")
            result[key] = joined
    return result


_FIELD_KEY_MAP = {
    "world_model": "world_model",
    "goal_model": "goal_model",
    "action_model": "action_model",
    "recent_findings": "recent_findings",
    "open_questions": "open_questions",
    "plan": "current_plan",
    "current_plan": "current_plan",
    "cross_level_notes": "cross_level_notes",
    "cross-level_notes": "cross_level_notes",
}


@dataclass
class WorldModel:
    """Structured carry-forward belief state extracted from LLM responses and environment feedback."""

    world_model: str = ""
    goal_model: str = ""
    action_model: str = ""
    recent_findings: str = ""
    open_questions: str = ""
    current_plan: str = ""
    cross_level_notes: str = ""

    @property
    def plan(self) -> str:
        return self.current_plan

    @plan.setter
    def plan(self, value: str) -> None:
        self.current_plan = value

    def update_from_text(self, text: str) -> None:
        """Parse labeled sections from any LLM or feedback text and merge non-empty values."""
        if not text or not text.strip():
            return
        extracted = _extract_labeled_blocks(text, _LABELS)
        for key, value in extracted.items():
            if value:
                target_attr = _FIELD_KEY_MAP.get(key, key)
                clean_val = value[:_MAX_FIELD_CHARS].strip()
                if clean_val and hasattr(self, target_attr):
                    setattr(self, target_attr, clean_val)

    def reset_level_fields(self) -> None:
        """Clear per-level fields on level transition; keep cross-level notes."""
        self.world_model = ""
        self.goal_model = ""
        self.action_model = ""
        self.recent_findings = ""
        self.open_questions = ""
        self.current_plan = ""

    def is_empty(self) -> bool:
        return not any([
            self.world_model,
            self.goal_model,
            self.action_model,
            self.recent_findings,
            self.open_questions,
            self.current_plan,
            self.cross_level_notes,
        ])

    def to_prompt_lines(self) -> list[str]:
        """Render as injection block for Brain and Planner prompts."""
        entries = [
            ("World model", self.world_model),
            ("Goal model", self.goal_model),
            ("Action model", self.action_model),
            ("Recent findings", self.recent_findings),
            ("Open questions", self.open_questions),
            ("Plan", self.current_plan),
            ("Cross-level notes", self.cross_level_notes),
        ]
        lines = [f"- {label}: {value}" for label, value in entries if value]
        if not lines:
            return []
        return [
            "Working world model carried from earlier turns:",
            *lines,
            "- Revise any item above immediately if current state contradicts it.",
        ]

    def to_prompt_block(self) -> str:
        lines = self.to_prompt_lines()
        return "\n".join(lines) if lines else ""
