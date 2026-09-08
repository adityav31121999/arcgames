"""Multimodal perception chain for ARC-AGI-3 Agent."""

from datetime import datetime, timezone
from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.state import ARCState, ARCTransition
from ..memory.knowledge import KnowledgeCache
from .prompts import PROMPT_ANALYSE_VISUAL, PROMPT_ASSUME, PROMPT_COMP_ASSUME, SYSTEM_PROMPT


class EyeChain:
    """Multimodal vision perception chain processing visual grid layouts and pixel changes."""

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
            return f"[EYE INFERENCE FAILED: {type(e).__name__}: {e}]"

    def assume(
        self,
        game_id: str,
        level: int,
        s0_state: ARCState,
        cache: KnowledgeCache,
        object_list: str = "",
        action_names: Optional[List[str]] = None,
    ) -> str:
        """Analyzes initial S0 state to form hypotheses regarding game rules and objective."""
        object_section = f"\nDetected Foreground Objects:\n{object_list}\n" if object_list else ""
        actions_section = f"Allowed Actions: {action_names}\n" if action_names else ""

        prompt = f"""{PROMPT_ASSUME}
{object_section}{actions_section}
State Metadata:
{s0_state.compact_json_repr}"""

        pil_img = s0_state.get_pil_image()
        text = self._invoke(prompt, image_obj=pil_img)

        # Robust English fallback if model inference failed or was empty
        if not text or "[EYE INFERENCE FAILED" in text:
            if object_list:
                text = (
                    f"World model: Grid environment containing detected foreground objects.\n"
                    "Goal model: Discover puzzle mechanics by interacting with unclicked shapes.\n"
                    "Action model: Execute allowable actions on target coordinates.\n"
                    "Recent findings: Initial visual segmentation completed.\n"
                    "Plan: Test primary interactive target object."
                )
            else:
                text = (
                    "World model: 2D grid puzzle state initialized.\n"
                    "Goal model: Explore grid interactions to reach goal state.\n"
                    "Action model: Execute allowable actions.\n"
                    "Recent findings: Initial state captured.\n"
                    "Plan: Begin systematic exploration."
                )

        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        cache.append_scratch(game_id, f"### Level {level} Hypothesis ({ts})\n{text}")
        cache.append_ostate(game_id, f"### Level {level} S0 Analysis ({ts})\n{text}")
        return text

    def compare_assume(
        self,
        game_id: str,
        level: int,
        s0_state: ARCState,
        cache: KnowledgeCache,
        object_list: str = "",
        action_names: Optional[List[str]] = None,
    ) -> str:
        """Compares prior level analyses with new level S0 state."""
        prior_ostate = cache.ostate(game_id)
        prior_scratch = cache.scratch(game_id)
        object_section = f"\nDetected Foreground Objects:\n{object_list}\n" if object_list else ""
        actions_section = f"Allowed Actions: {action_names}\n" if action_names else ""

        prompt = f"""{PROMPT_COMP_ASSUME}

Prior Level Analyses:
{prior_ostate}

Knowledge Store:
{prior_scratch}
{object_section}{actions_section}
State Metadata of S0:
{s0_state.compact_json_repr}"""

        pil_img = s0_state.get_pil_image()
        text = self._invoke(prompt, image_obj=pil_img)

        if not text or "[EYE INFERENCE FAILED" in text:
            text = (
                "World model: Shifted level layout with updated coordinate targets.\n"
                "Goal model: Apply established rules to new level configuration.\n"
                "Action model: Utilize valid action space.\n"
                "Recent findings: Level transition observed.\n"
                "Plan: Apply mechanics learned from previous levels."
            )

        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        cache.append_scratch(game_id, f"### Level {level} Objective Delta ({ts})\n{text}")
        cache.append_ostate(game_id, f"### Level {level} S0 Analysis ({ts})\n{text}")
        return text

    def analyse_visual(
        self, game_id: str, s0_state: ARCState, transition: ARCTransition, diff_text: str
    ) -> str:
        """Analyzes specific visual changes between steps using ground-truth diff bounding box."""
        prompt = f"""{PROMPT_ANALYSE_VISUAL}

GROUND-TRUTH PIXEL DIFF:
{diff_text}

State Metadata:
{transition.current.compact_json_repr}"""

        pil_img = transition.current.get_pil_image()
        return self._invoke(prompt, image_obj=pil_img)

