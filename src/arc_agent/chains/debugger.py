"""State transition verification and rule divergence debugging chain."""

from typing import Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.state import ARCState, ARCTransition
from ..memory.knowledge import KnowledgeCache
from .prompts import PROMPT_STATE_DEBUG, SYSTEM_PROMPT


class DebuggerChain:
    """Validates whether state transitions matched expectations or collided with obstacles."""

    def __init__(self, model: BaseChatModel, max_tokens: int = 512):
        self.model = model
        self.max_tokens = max_tokens

    def _invoke(self, prompt: str, image_obj: Optional[Any] = None) -> str:
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
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
    ) -> str:
        """Evaluates whether the last move was expected, blocked, or altered the target."""
        actions_log = cache.actions_log(game_id, level) if cache else ""
        scratch = cache.scratch(game_id) if cache else ""
        action_line = str(transition.action_sig.name) if transition.action_sig else "Initial step"

        prompt = f"""{PROMPT_STATE_DEBUG}

Action causing transition: {action_line}
Ground-truth pixel changes: {diff_text}
Visual context: {visual_analysis}
Recent Actions: {actions_log}
Known Rules: {scratch}

JSON State Metadata of current step:
{transition.current.proper_json_repr}

Reply in at most 3 short lines: (1) was this transition expected or blocked/wall collision, (2) recommendation for next step. Keep headers/markdown out."""

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

Action '{attempted_action}' produced NO state change.
Ground-truth pixel changes: {diff_text}
Recent Actions: {actions_log}
Scratchpad: {scratch}

JSON State Metadata of current step:
{transition.current.proper_json_repr}

Classify result. Begin response strictly with:
EXPECTED: <reason this is a normal rule or wall collision>
DIVERGED: <reason logic is incorrect>"""

        pil_img = transition.current.get_pil_image()
        return self._invoke(prompt, image_obj=pil_img)
