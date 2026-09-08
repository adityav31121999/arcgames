"""Brain decision-making and macro-planning chain for ARC-AGI-3 Agent."""

from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.state import ARCState
from ..core.object_detection import is_click_only
from ..memory.knowledge import KnowledgeCache
from .prompts import PROMPT_ACTION, PROMPT_CLICK_ONLY_TARGET, SYSTEM_PROMPT


class BrainChain:
    """Core reasoning and action-selection engine."""

    def __init__(self, model: BaseChatModel, max_tokens: int = 256, system_prompt: str = SYSTEM_PROMPT):
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    def set_system_prompt(self, system_prompt: str) -> None:
        """Updates the system prompt for dynamic action spaces."""
        self.system_prompt = system_prompt

    def _invoke(self, prompt: str, temperature: float = 0.0, max_tokens: int = 32, stop: Optional[List[str]] = None, image_obj: Optional[Any] = None) -> str:
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt if image_obj is None else [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image_obj},
            ]),
        ]
        try:
            invoke_kwargs = {
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if stop:
                invoke_kwargs["stop"] = stop
            response = self.model.invoke(messages, **invoke_kwargs)
            return str(response.content).strip()
        except Exception as e:
            return f"[BRAIN INFERENCE FAILED: {type(e).__name__}: {e}]"

    def decide_action(
        self,
        game_id: str,
        level: int,
        s0_state: ARCState,
        current_state: ARCState,
        valid_actions: List[Any],
        context_note: str,
        cache: KnowledgeCache,
        world_model_block: str = "",
        budget_context: str = "",
        object_list: str = "",
        click_history: str = "",
    ) -> str:
        """Determines next discrete or complex coordinate action."""
        actions_log = cache.actions_log(game_id, level)
        scratch = cache.scratch(game_id)

        grid_repr_context = (
            f"Current board image attached. Grid shape (height, width): {current_state.grid.shape}. "
            "Coordinates use original grid cells: X is column, Y is row, origin at top left."
            if current_state.grid is not None else "No current board available."
        )

        action_names = [getattr(a, "name", str(a)) for a in valid_actions]
        world_model_section = f"\n{world_model_block}\n" if world_model_block else ""
        budget_section = f"Move Budget Status: {budget_context}\n" if budget_context else ""
        context_section = f"Navigation Context: {context_note}\n" if context_note else ""

        if is_click_only(valid_actions):
            base_prompt = PROMPT_CLICK_ONLY_TARGET.format(
                object_list=object_list or "No distinct foreground objects detected.",
                click_history=click_history or "No coordinates clicked yet in this attempt.",
            )
        else:
            base_prompt = PROMPT_ACTION

        prompt = f"""{base_prompt}
{world_model_section}{budget_section}{context_section}
State Metadata:
{current_state.compact_json_repr}
{grid_repr_context}

Recent Actions Log:
{actions_log}

Knowledge Store:
{scratch}

Legal actions: {action_names}

Reply format:
Plan: <one sentence goal and rationale>
ACTION=<NAME> [X=<int> Y=<int>]
Next action:"""

        return self._invoke(
            prompt,
            temperature=0.0,
            max_tokens=min(128, self.max_tokens),
            image_obj=current_state.get_pil_image(),
        )

    def one_shot_plan(
        self,
        game_id: str,
        level: int,
        s0_state: ARCState,
        valid_actions: List[Any],
        cache: KnowledgeCache,
        world_model_block: str = "",
    ) -> str:
        """Synthesizes speculative macro-plan sequence for rapid execution."""
        ostate = cache.ostate(game_id)
        scratch = cache.scratch(game_id)
        actions_log = cache.actions_log(game_id, level)
        valid_names = [getattr(a, "name", str(a)) for a in valid_actions]
        world_model_section = f"\n{world_model_block}\n" if world_model_block else ""

        prompt = f"""{PROMPT_ACTION}
{world_model_section}
Synthesize a ONE-SHOT plan for Level {level}. Format EACH line strictly as:
ACTION=<NAME> [X=<int> Y=<int>]

Prior Level Analyses: {ostate}
Knowledge Store: {scratch}
Actions Log: {actions_log}
S0 Grid Matrix: {s0_state.text_repr}
Valid Actions: {valid_names}

Ordered action sequence:"""

        return self._invoke(prompt, temperature=0.0, max_tokens=256)
