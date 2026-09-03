"""Brain decision-making and macro-planning chain for ARC-AGI-3 Agent."""

from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.state import ARCState
from ..memory.knowledge import KnowledgeCache
from .prompts import PROMPT_ACTION, SYSTEM_PROMPT


class BrainChain:
    """Core reasoning and action-selection engine."""

    def __init__(self, model: BaseChatModel, max_tokens: int = 256):
        self.model = model
        self.max_tokens = max_tokens

    def _invoke(self, prompt: str, temperature: float = 0.0, max_tokens: int = 32, stop: Optional[List[str]] = None) -> str:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
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
    ) -> str:
        """Determines next discrete or complex coordinate action."""
        actions_log = cache.actions_log(game_id, level)
        scratch = cache.scratch(game_id)

        if current_state.step == 0:
            grid_repr_context = f"S0 Matrix (Hash: {s0_state.state_hash[:12]}):\n{s0_state.text_repr}"
        else:
            grid_repr_context = (
                f"Initial S0 Hash: {s0_state.state_hash[:12]} | Current St Hash: {current_state.state_hash[:12]}\n"
                f"Grid Shape: {current_state.grid.shape if current_state.grid is not None else 'Unknown'}"
            )

        action_names = [getattr(a, "name", str(a)) for a in valid_actions]

        prompt = f"""{PROMPT_ACTION}

{grid_repr_context}

JSON State Metadata:
{current_state.proper_json_repr}

Knowledge Store: {scratch}
Actions Log: {actions_log}
Tracker/Loop/Sprite Context: {context_note}
Legal actions (Prohibited/redundant ones are already filtered - choose from this list): {action_names}

Reply strictly in format: ACTION=<NAME> [X=<int> Y=<int>]
Next action:"""

        return self._invoke(prompt, temperature=0.0, max_tokens=min(32, self.max_tokens), stop=["\n"])

    def one_shot_plan(
        self,
        game_id: str,
        level: int,
        s0_state: ARCState,
        valid_actions: List[Any],
        cache: KnowledgeCache,
    ) -> str:
        """Synthesizes speculative macro-plan sequence for rapid execution."""
        ostate = cache.ostate(game_id)
        scratch = cache.scratch(game_id)
        actions_log = cache.actions_log(game_id, level)
        valid_names = [getattr(a, "name", str(a)) for a in valid_actions]

        prompt = f"""{PROMPT_ACTION}

Synthesize a ONE-SHOT plan for Level {level}. Format EACH line strictly as:
ACTION=<NAME> [X=<int> Y=<int>]

Cross-Level S0 Analysis: {ostate}
Knowledge Store: {scratch}
Actions Log: {actions_log}
S0 Matrix (Hash: {s0_state.state_hash[:12]}): {s0_state.text_repr}
Valid Actions: {valid_names}

Ordered action sequence:"""

        return self._invoke(prompt, temperature=0.0, max_tokens=256)
