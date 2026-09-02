"""LangChain chains and prompt definitions for ARC-AGI-3 Agent."""

from .prompts import (
    SYSTEM_PROMPT,
    PROMPT_ASSUME,
    PROMPT_COMP_ASSUME,
    PROMPT_ANALYSE_VISUAL,
    PROMPT_STATE_DEBUG,
    PROMPT_ACTION,
    PROMPT_ITERATION_REVIEW,
)
from .eye import EyeChain
from .debugger import DebuggerChain
from .brain import BrainChain
from .reviewer import ReviewerChain

__all__ = [
    "SYSTEM_PROMPT",
    "PROMPT_ASSUME",
    "PROMPT_COMP_ASSUME",
    "PROMPT_ANALYSE_VISUAL",
    "PROMPT_STATE_DEBUG",
    "PROMPT_ACTION",
    "PROMPT_ITERATION_REVIEW",
    "EyeChain",
    "DebuggerChain",
    "BrainChain",
    "ReviewerChain",
]
