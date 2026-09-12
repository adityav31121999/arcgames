"""Multimodal perception chain for ARC-AGI-3 Agent."""

from datetime import datetime, timezone
from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel

from ..core.state import ARCState, ARCTransition
from ..memory.knowledge import KnowledgeCache
from ..core.object_detection import render_detected_objects
from ..core.context import context_part
from .inference import StageResult, invoke_stage
from .prompts import PROMPT_ANALYSE_VISUAL, PROMPT_ASSUME, PROMPT_COMP_ASSUME, SYSTEM_PROMPT


class EyeChain:
    """Multimodal vision perception chain processing visual grid layouts and pixel changes."""

    def __init__(self, model: BaseChatModel, max_tokens: int = 1024, system_prompt: str = SYSTEM_PROMPT):
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.last_result = StageResult(False, error="Not run")

    def set_system_prompt(self, system_prompt: str) -> None:
        """Updates the system prompt for dynamic action spaces."""
        self.system_prompt = system_prompt

    def _invoke(self, prompt: str, image_obj: Optional[Any] = None, *, images=None,
                context=(), required_labels=("World model", "Goal model", "Action model", "Plan")) -> str:
        self.last_result = invoke_stage(
            self.model, self.system_prompt, prompt, stage="Vision", max_tokens=self.max_tokens,
            images=images if images is not None else [("Current board", image_obj)],
            context=context, required_labels=required_labels,
        )
        return self.last_result.text

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

        if not self.last_result.ok:
            # Report capture status, not invented objectives or mechanics.
            return (
                f"Recent findings: Initial board captured; visual interpretation unavailable.\n"
                "Open questions: Identify the goal and action effects through observation."
            )

        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        cache.append_scratch(game_id, f"## LEVEL ANALYSES (UNVERIFIED)\n### Level {level} Hypothesis ({ts})\n{text}")
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
        prior_s0_image: Optional[Any] = None,
        prior_world_model: Optional[str] = None,
    ) -> str:
        """Compares prior level analyses with new level S0 state."""
        object_section = f"\nDetected Foreground Objects:\n{object_list}\n" if object_list else ""
        actions_section = f"Allowed Actions: {action_names}\n" if action_names else ""
        prior_wm_section = f"\nPrior Level World Model / Grounded Notes:\n{prior_world_model}\n" if prior_world_model else ""

        prompt = f"""{PROMPT_COMP_ASSUME}
{prior_wm_section}
{object_section}{actions_section}
State Metadata of S0:
{s0_state.compact_json_repr}"""

        pil_img = s0_state.get_pil_image()
        if prior_s0_image is not None:
            images = [("Prior level S0", prior_s0_image), ("Current level S0", pil_img)]
            text = self._invoke(prompt, images=images, context=cache.context_sections(game_id, level, include_prior=True))
        else:
            text = self._invoke(prompt, image_obj=pil_img, context=cache.context_sections(game_id, level, include_prior=True))

        if not self.last_result.ok:
            return (
                "Recent findings: New board captured; visual comparison unavailable.\n"
                "Open questions: Recheck prior hypotheses against the current board."
            )

        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        cache.append_scratch(game_id, f"## LEVEL ANALYSES (UNVERIFIED)\n### Level {level} Objective Delta ({ts})\n{text}")
        cache.append_ostate(game_id, f"### Level {level} S0 Analysis ({ts})\n{text}")
        return text

    def analyse_visual(
        self, game_id: str, s0_state: ARCState, transition: ARCTransition, diff_text: str,
        *, intended_plan: str = "", expected_effect: str = "", world_model_block: str = "",
    ) -> str:
        """Analyzes specific visual changes between steps using ground-truth diff bounding box."""
        prompt = f"""{PROMPT_ANALYSE_VISUAL}

Action causing transition: {transition.action_sig}
Intended plan: {intended_plan or 'Unknown (fallback or speculative action).'}
Expected observable effect: {expected_effect or 'Unknown; do not invent a prediction.'}
Compare this prediction with the observed changes; report support, contradiction, or uncertainty.
Working world model (hypotheses): {world_model_block}
Previous board shape: {transition.previous.grid.shape if transition.previous and transition.previous.grid is not None else 'unavailable'}
Current board shape: {transition.current.grid.shape if transition.current.grid is not None else 'unavailable'}
Coordinates use original grid cells: X=column, Y=row, origin top left.
GROUND-TRUTH PIXEL DIFF:
{diff_text}

State Metadata:
{transition.current.compact_json_repr}"""

        return self._invoke(
            prompt,
            images=[
                ("Before action", transition.previous.get_pil_image() if transition.previous else None),
                ("After action", transition.current.get_pil_image()),
            ],
            required_labels=("Recent findings", "Open questions"),
            context=[
                context_part("Measured objects before (roles unknown)", render_detected_objects(transition.previous.grid) if transition.previous else "Unavailable", 1, "head"),
                context_part("Measured objects after (roles unknown)", render_detected_objects(transition.current.grid), 1, "head"),
            ],
        )

