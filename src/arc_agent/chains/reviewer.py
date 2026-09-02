"""Post-iteration meta-reflection and failure analysis chain."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.state import ARCState
from ..memory.knowledge import KnowledgeCache
from .prompts import PROMPT_ITERATION_REVIEW, SYSTEM_PROMPT


class ReviewerChain:
    """Reflects on failed attempts to consolidate verified mechanics and navigation rules."""

    def __init__(self, model: BaseChatModel, max_tokens: int = 768):
        self.model = model
        self.max_tokens = max_tokens

    def review(
        self,
        game_id: str,
        level: int,
        iteration: int,
        s0_state: ARCState,
        final_state: ARCState,
        cache: KnowledgeCache,
    ) -> str:
        full_actions = cache.actions_log(game_id, level, max_chars=6000)
        full_scratch = cache.scratch(game_id, max_chars=4000)
        final_state_name = getattr(final_state.game_state, "name", str(final_state.game_state))

        prompt = f"""{PROMPT_ITERATION_REVIEW}

Iteration: {iteration}
Final game state reached: {final_state_name}

Full Actions Log for this level (all iterations so far):
{full_actions}

Current Knowledge Store (scratchpad, untruncated):
{full_scratch}"""

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.model.invoke(
                messages,
                temperature=0.35,
                max_tokens=self.max_tokens,
            )
            return str(response.content).strip()
        except Exception as e:
            return f"[REVIEW INFERENCE FAILED: {type(e).__name__}: {e}]"
