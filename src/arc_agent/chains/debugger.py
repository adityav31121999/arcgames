"""State transition verification and rule divergence debugging chain."""

from typing import Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel

from ..core.state import ARCState, ARCTransition
from ..memory.knowledge import KnowledgeCache
from ..core.context import context_part
from .inference import StageResult, invoke_stage
from .prompts import PROMPT_STATE_DEBUG, SYSTEM_PROMPT


class DebuggerChain:
    """Validates whether state transitions matched expectations or collided with obstacles."""

    def __init__(self, model: BaseChatModel, max_tokens: int = 512, system_prompt: str = SYSTEM_PROMPT):
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.last_result = StageResult(False, error="Not run")

    def set_system_prompt(self, system_prompt: str) -> None:
        """Updates the system prompt for dynamic action spaces."""
        self.system_prompt = system_prompt

    def _invoke(self, prompt: str, image_obj: Optional[Any] = None, *, images=None, context=()) -> str:
        self.last_result = invoke_stage(
            self.model, self.system_prompt, prompt, stage="Debugger", max_tokens=self.max_tokens,
            images=images if images is not None else [("Current board", image_obj)],
            context=context, required_labels=("Recent findings", "Open questions", "Plan"),
        )
        return self.last_result.text

    def validate(
        self,
        game_id: str,
        level: int,
        s0_state: ARCState,
        transition: ARCTransition,
        diff_text: str,
        visual_analysis: str = "",
        cache: Optional[KnowledgeCache] = None,
        budget_context: str = "",
        intended_plan: str = "",
        expected_effect: str = "",
        world_model_block: str = "",
    ) -> str:
        """Evaluates whether the last move was expected, blocked, or altered the target."""
        context = cache.context_sections(game_id, level) if cache else []
        if world_model_block:
            context.append(context_part("Working world model (hypotheses)", world_model_block, 1, "head"))
        action_line = str(transition.action_sig) if transition.action_sig else "Initial step"
        budget_line = f"\nMove Budget Status: {budget_context}" if budget_context else ""

        prompt = f"""{PROMPT_STATE_DEBUG}{budget_line}

Action causing transition: {action_line}
Intended plan: {intended_plan or 'No model plan available (fallback/unknown).'}
Expected observable effect: {expected_effect or 'Unknown; do not invent a prediction.'}
Compare the expected effect against observed evidence; state whether it was supported, contradicted, or uncertain.
Ground-truth pixel changes: {diff_text}
Visual context: {visual_analysis}

State Metadata of current step:
{transition.current.compact_json_repr}

Previous board (before this action):
Before image attached when available. Coordinates: X=column, Y=row, origin top left.
Detected gameplay change: {transition.changed}
Use only the three labeled lines requested above."""

        return self._invoke(prompt, images=[
            ("Before action", transition.previous.get_pil_image() if transition.previous else None),
            ("After action", transition.current.get_pil_image()),
        ], context=context)

    def check_divergence(
        self,
        game_id: str,
        level: int,
        s0_state: ARCState,
        attempted_action: str,
        transition: ARCTransition,
        diff_text: str,
        cache: Optional[KnowledgeCache] = None,
    ) -> str:
        """Classifies zero-diff / NO-OP results to update verified rules or debunked assumptions."""
        context = cache.context_sections(game_id, level) if cache else []

        prompt = f"""{PROMPT_STATE_DEBUG}

Attempted action: {attempted_action}
Action signature including coordinates: {transition.action_sig}
Detected gameplay change: {transition.changed}
Ground-truth pixel changes: {diff_text}

State Metadata of current step:
{transition.current.compact_json_repr}

Previous board (before this action):
Before image attached when available. Coordinates: X=column, Y=row, origin top left.
Use only the three labeled lines requested above. If the cause is unknown, say so."""

        return self._invoke(prompt, images=[
            ("Before action", transition.previous.get_pil_image() if transition.previous else None),
            ("After action", transition.current.get_pil_image()),
        ], context=context)
