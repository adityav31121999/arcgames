"""Brain decision-making and macro-planning chain for ARC-AGI-3 Agent."""

from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel

from ..core.state import ARCState
from ..core.object_detection import is_click_only
from ..memory.knowledge import KnowledgeCache
from .prompts import PROMPT_ACTION, PROMPT_CLICK_ONLY_TARGET, PROMPT_ITERATION_REVIEW, SYSTEM_PROMPT
from .inference import StageResult, build_messages, invoke_stage
from ..core.context import context_part

class BrainChain:
    """Core reasoning and action-selection engine."""

    def __init__(self, model: BaseChatModel, max_tokens: int = 256, system_prompt: str = SYSTEM_PROMPT):
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.last_review_result = StageResult(False, error="Not run")

    def set_system_prompt(self, system_prompt: str) -> None:
        """Updates the system prompt for dynamic action spaces."""
        self.system_prompt = system_prompt

    def _invoke(self, prompt: str, temperature: float = 0.0, max_tokens: int = 32, stop: Optional[List[str]] = None, image_obj: Optional[Any] = None, action_response: bool = False, context=()) -> str:
        messages = build_messages(self.system_prompt, prompt, [("", image_obj)], context)
        try:
            invoke_kwargs = {
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if stop:
                invoke_kwargs["stop"] = stop
            if action_response:
                invoke_kwargs["action_response"] = True
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
        context = cache.context_sections(game_id, level)
        if world_model_block:
            context.append(context_part("Working world model", world_model_block, 0, "head"))

        grid_repr_context = (
            f"Current board image attached. Grid shape (height, width): {current_state.grid.shape}. "
            "Coordinates use original grid cells: X is column, Y is row, origin at top left."
            if current_state.grid is not None else "No current board available."
        )

        action_names = [getattr(a, "name", str(a)) for a in valid_actions]
        budget_section = f"Move Budget Status: {budget_context}\n" if budget_context else ""
        context_section = f"Navigation Context: {context_note}\n" if context_note else ""

        if is_click_only(valid_actions):
            base_prompt = PROMPT_CLICK_ONLY_TARGET.format(
                object_list="See detected objects context below.",
                click_history="See click history context below.",
            )
            context.append(context_part("Detected objects", object_list or "No distinct objects detected.", 1, "head"))
            context.append(context_part("Click history", click_history or "No clicks yet.", 3))
        else:
            base_prompt = PROMPT_ACTION

        prompt = f"""{base_prompt}
{budget_section}{context_section}
State Metadata:
{current_state.compact_json_repr}
{grid_repr_context}

Legal actions: {action_names}

Reply format: three labeled lines. End the action line with [END_ACTION].
Plan: <one sentence goal and rationale>
Expected effect: <specific observable change to test, or unknown>
ACTION=<NAME> [X=<int> Y=<int>] [END_ACTION]
Next action:"""

        return self._invoke(
            prompt,
            temperature=0.0,
            max_tokens=min(128, self.max_tokens),
            stop=["[END_ACTION]"],
            image_obj=current_state.get_pil_image(),
            action_response=True,
            context=context,
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
        context = cache.context_sections(game_id, level, include_prior=True)
        valid_names = [getattr(a, "name", str(a)) for a in valid_actions]
        if world_model_block:
            context.append(context_part("Working world model", world_model_block, 0, "head"))

        prompt = f"""{PROMPT_ACTION}
Synthesize a ONE-SHOT plan for Level {level}. Format EACH line strictly as:
ACTION=<NAME> [X=<int> Y=<int>]

Initial board image attached. Grid shape: {s0_state.grid.shape if s0_state.grid is not None else 'unavailable'}.
Coordinates use original grid cells, X=column, Y=row, origin top left.
Valid Actions: {valid_names}

Ordered action sequence:"""

        return self._invoke(prompt, temperature=0.0, max_tokens=256,
                            image_obj=s0_state.get_pil_image(), context=context)

    def review(
        self,
        game_id: str,
        level: int,
        iteration: int,
        s0_state: ARCState,
        final_state: ARCState,
        cache: KnowledgeCache,
    ) -> str:
        final_state_name = getattr(final_state.game_state, "name", str(final_state.game_state))

        prompt = f"""{PROMPT_ITERATION_REVIEW}

Iteration: {iteration}
Final game state reached: {final_state_name}

Review the attached initial/final boards and memory context. Distinguish hypotheses from evidence."""

        self.last_review_result = invoke_stage(
            self.model, self.system_prompt, prompt, stage="Brain review", max_tokens=self.max_tokens,
            images=[("Initial board", s0_state.get_pil_image()), ("Final board", final_state.get_pil_image())],
            context=cache.context_sections(game_id, level),
            required_labels=("FAILURE_REASON", "RULES"), temperature=0.35,
        )
        return self.last_review_result.text
