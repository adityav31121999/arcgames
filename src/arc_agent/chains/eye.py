"""Multimodal perception chain for ARC-AGI-3 Agent."""

from datetime import datetime, timezone
from typing import Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.state import ARCState, ARCTransition
from ..memory.knowledge import KnowledgeCache
from .prompts import PROMPT_ANALYSE_VISUAL, PROMPT_ASSUME, PROMPT_COMP_ASSUME, SYSTEM_PROMPT


class EyeChain:
    """Multimodal vision perception chain processing visual grid layouts and pixel changes."""

    def __init__(self, model: BaseChatModel, max_tokens: int = 1024):
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
            return f"[EYE INFERENCE FAILED: {type(e).__name__}: {e}]"

    def assume(self, game_id: str, level: int, s0_state: ARCState, cache: KnowledgeCache) -> str:
        """Analyzes initial S0 state to form hypotheses regarding game rules and objective."""
        prompt = f"""{PROMPT_ASSUME}

Analyse the initial PNG configuration and JSON state metadata to infer overall game rules and level objectives.

JSON State Metadata:
{s0_state.proper_json_repr}"""

        pil_img = s0_state.get_pil_image()
        text = self._invoke(prompt, image_obj=pil_img)

        ts = datetime.now(timezone.utc).isoformat()
        cache.append_scratch(game_id, f"### Level {level} Hypothesis ({ts})\n{text}")
        cache.append_ostate(game_id, f"### Level {level} S0 Analysis ({ts})\nHash: {s0_state.state_hash}\n{text}")
        return text

    def compare_assume(self, game_id: str, level: int, s0_state: ARCState, cache: KnowledgeCache) -> str:
        """Compares prior level analyses with new level S0 state."""
        prior_ostate = cache.ostate(game_id)
        prior_scratch = cache.scratch(game_id)

        prompt = f"""{PROMPT_COMP_ASSUME}

Cross-level S0 analyses:
{prior_ostate}

Knowledge Store:
{prior_scratch}

JSON State Metadata of S0:
{s0_state.proper_json_repr}

Infer specific targets and coordinate focal fields from the visual layout and state data."""

        pil_img = s0_state.get_pil_image()
        text = self._invoke(prompt, image_obj=pil_img)

        ts = datetime.now(timezone.utc).isoformat()
        cache.append_scratch(game_id, f"### Level {level} Objective Delta ({ts})\n{text}")
        cache.append_ostate(game_id, f"### Level {level} S0 Analysis ({ts})\nHash: {s0_state.state_hash}\n{text}")
        return text

    def analyse_visual(
        self, game_id: str, s0_state: ARCState, transition: ARCTransition, diff_text: str
    ) -> str:
        """Analyzes specific visual changes between steps using ground-truth diff bounding box."""
        prompt = f"""{PROMPT_ANALYSE_VISUAL}

GROUND-TRUTH PIXEL DIFF (Exactly what changed - do not describe anything beyond this specific coordinate region):
{diff_text}

JSON State Metadata of the resulting state:
{transition.current.proper_json_repr}

Respond in at most 2 short sentences. No markdown, no speculation about unevidenced entities."""

        pil_img = transition.current.get_pil_image()
        return self._invoke(prompt, image_obj=pil_img)
