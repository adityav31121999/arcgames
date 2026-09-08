"""State transition verification and rule divergence debugging chain."""

from typing import Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.state import ARCState, ARCTransition
from ..memory.knowledge import KnowledgeCache
from .prompts import PROMPT_STATE_DEBUG, SYSTEM_PROMPT


class DebuggerChain:
    """Validates whether state transitions matched expectations or collided with obstacles."""

    def __init__(self, model: BaseChatModel, max_tokens: int = 512, system_prompt: str = SYSTEM_PROMPT):
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    def set_system_prompt(self, system_prompt: str) -> None:
        """Updates the system prompt for dynamic action spaces."""
        self.system_prompt = system_prompt

    def _invoke(self, prompt: str, image_obj: Optional[Any] = None) -> str:
        messages = [SystemMessage(content=self.system_prompt)]
        if image_obj is not None:
            messages.append(
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image_obj},
                    ]
                )
            )
        else:
            messages.append(HumanMessage(content=prompt))

        try:
            response = self.model.invoke(messages, max_tokens=self.max_tokens)
            return str(response.content).strip()
        except Exception as e:
            return f"[DEBUGGER INFERENCE FAILED: {type(e).__name__}: {e}]"

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
    ) -> str:
        """Evaluates whether the last move was expected, blocked, or altered the target."""
        actions_log = cache.actions_log(game_id, level) if cache else ""
        scratch = cache.scratch(game_id) if cache else ""
        action_line = str(transition.action_sig) if transition.action_sig else "Initial step"
        budget_line = f"\nMove Budget Status: {budget_context}" if budget_context else ""

        prompt = f"""{PROMPT_STATE_DEBUG}{budget_line}

Action causing transition: {action_line}
Ground-truth pixel changes: {diff_text}
Visual context: {visual_analysis}
Recent Actions: {actions_log}
Memory (includes unverified hypotheses): {scratch}

State Metadata of current step:
{transition.current.compact_json_repr}

Previous board (before this action):
{transition.previous.text_repr if transition.previous else "Unavailable"}
Detected gameplay change: {transition.changed}
Use only the three labeled lines requested above."""

        pil_img = transition.current.get_pil_image()
        return self._invoke(prompt, image_obj=pil_img)

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
        actions_log = cache.actions_log(game_id, level) if cache else ""
        scratch = cache.scratch(game_id) if cache else ""

        prompt = f"""{PROMPT_STATE_DEBUG}

Attempted action: {attempted_action}
Action signature including coordinates: {transition.action_sig}
Detected gameplay change: {transition.changed}
Ground-truth pixel changes: {diff_text}
Recent Actions: {actions_log}
Scratchpad: {scratch}

State Metadata of current step:
{transition.current.compact_json_repr}

Previous board (before this action):
{transition.previous.text_repr if transition.previous else "Unavailable"}
Use only the three labeled lines requested above. If the cause is unknown, say so."""

        pil_img = transition.current.get_pil_image()
        return self._invoke(prompt, image_obj=pil_img)
