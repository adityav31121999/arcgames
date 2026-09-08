"""Post-iteration meta-reflection and failure analysis chain."""

from langchain_core.language_models.chat_models import BaseChatModel

from ..core.state import ARCState
from ..memory.knowledge import KnowledgeCache
from .inference import StageResult, invoke_stage
from .prompts import PROMPT_ITERATION_REVIEW, SYSTEM_PROMPT


class ReviewerChain:
    """Reflects on failed attempts to consolidate verified mechanics and navigation rules."""

    def __init__(self, model: BaseChatModel, max_tokens: int = 1024, system_prompt: str = SYSTEM_PROMPT):
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.last_result = StageResult(False, error="Not run")

    def set_system_prompt(self, system_prompt: str) -> None:
        """Updates the system prompt for dynamic action spaces."""
        self.system_prompt = system_prompt

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

        self.last_result = invoke_stage(
            self.model, self.system_prompt, prompt, stage="Reviewer", max_tokens=self.max_tokens,
            images=[("Initial board", s0_state.get_pil_image()), ("Final board", final_state.get_pil_image())],
            context=cache.context_sections(game_id, level),
            required_labels=("FAILURE_REASON", "RULES"), temperature=0.35,
        )
        return self.last_result.text
