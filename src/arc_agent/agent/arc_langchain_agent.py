"""High-level LangChain-powered ARCAgent orchestrating chains, spatial memory, and heuristics."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..chains.brain import BrainChain
from ..chains.debugger import DebuggerChain
from ..chains.eye import EyeChain
from ..chains.reviewer import ReviewerChain
from ..core.actions import ARCActionMapper, ActionSignature, is_complex_action
from ..core.resolver import GameStateResolver
from ..core.state import ARCState, ARCTransition, compute_transition, save_step_state_json
from ..memory.knowledge import (
    KnowledgeCache,
    apply_iteration_review,
    init_knowledge_files,
    maybe_append_rule,
)
from ..memory.trajectory import TrajectoryMemory
from ..memory.world_model import WorldModel
from ..utils.display import render_live


class ARCLangChainAgent:
    """Core Agent coordinating LangChain perceptual, debugger, planner, and review chains."""

    def __init__(
        self,
        eye_chain: EyeChain,
        debugger_chain: DebuggerChain,
        brain_chain: BrainChain,
        reviewer_chain: ReviewerChain,
        resolver: GameStateResolver,
        stuck_threshold: int = 3,
        memory_root: str = "./agent_memory",
        vision_cache_dir: str = "/tmp/agent_vision",
    ):
        self.eye = eye_chain
        self.debugger = debugger_chain
        self.brain = brain_chain
        self.reviewer = reviewer_chain
        self.resolver = resolver
        self.stuck_threshold = stuck_threshold
        self.memory_root = memory_root
        self.vision_cache_dir = vision_cache_dir

        self.memory = TrajectoryMemory()
        self.cache = KnowledgeCache(memory_root=memory_root)
        self.world_model = WorldModel()

    def enter_level(
        self,
        game_id: str,
        level: int,
        obs: Any,
        is_first_level_of_game: bool,
        valid_actions: Optional[List[Any]] = None,
    ) -> ARCState:
        """Initializes state, memory, and performs initial visual analysis of S0."""
        init_knowledge_files(game_id, level, valid_actions, memory_root=self.memory_root)
        self.cache.refresh_level(game_id, level)

        s0_state = ARCState.create(game_id, level, 0, obs, tag="S0")
        self.memory.reset(s0_state.state_hash)

        render_live(s0_state, status="S0 Initial State Setup")

        self.world_model.reset_level_fields()

        if is_first_level_of_game:
            eye_out = self.eye.assume(game_id, level, s0_state, self.cache)
        else:
            eye_out = self.eye.compare_assume(game_id, level, s0_state, self.cache)

        if eye_out:
            self.world_model.update_from_text(eye_out)

        return s0_state

    def execute_action(
        self,
        game_id: str,
        level: int,
        env: Any,
        prior_state: ARCState,
        action: Any,
        action_data: Dict[str, Any],
        step_index: int,
        tag: str,
    ) -> Tuple[ARCState, ARCTransition, bool, bool]:
        """Executes action in environment, captures new state and updates spatial trajectory."""
        action_sig = ActionSignature.from_action(action, action_data)
        if action_data:
            try:
                raw_obs = env.step(action, data=action_data)
            except TypeError:
                try:
                    raw_obs = env.step(action, **action_data)
                except TypeError:
                    raw_obs = env.step(action)
        else:
            raw_obs = env.step(action)

        new_state = ARCState.create(game_id, level, step_index, raw_obs, tag=tag)
        transition = compute_transition(prior_state, new_state, action_sig=action_sig)

        if step_index == 1:
            save_step_state_json(game_id, level, 1, new_state, memory_root=self.memory_root)

        if transition.changed:
            self.memory.update_sprite_region(prior_state.grid, new_state.grid)

        is_repeat_state, is_repeat_transition = self.memory.record_transition(
            prior_state.state_hash, action_sig, new_state.state_hash, transition.changed
        )
        return new_state, transition, is_repeat_state, is_repeat_transition

    def decide_action(
        self,
        game_id: str,
        level: int,
        s0_state: ARCState,
        current_state: ARCState,
        valid_actions: List[Any],
        debug_note: str,
    ) -> Tuple[Any, Dict[str, Any], str]:
        """Decides next action using Brain chain with formatting retries and heuristics fallbacks."""
        grid_shape = current_state.grid.shape if current_state.grid is not None else None
        state_hash = current_state.state_hash

        allowed_actions = self.memory.get_allowed_actions(state_hash, valid_actions)
        warning = self.memory.loop_warning(state_hash)
        behavior_warning = self.memory.consecutive_action_warning(threshold=5)
        trajectory_ctx = self.memory.recent_trajectory_text()
        sprite_highlight = self.memory.get_sprite_guidance()

        context_note = "\n".join(
            x
            for x in [debug_note, warning, behavior_warning, sprite_highlight, f"[TRAJECTORY] {trajectory_ctx}"]
            if x
        )
        prohibited = self.memory.tried_signatures(state_hash)

        world_model_block = self.world_model.to_prompt_block()

        raw = self.brain.decide_action(
            game_id,
            level,
            s0_state,
            current_state,
            allowed_actions,
            context_note,
            self.cache,
            world_model_block=world_model_block,
        )
        if raw:
            self.world_model.update_from_text(raw)
        action, action_data = ARCActionMapper.parse(raw, allowed_actions, grid_shape, prohibited=prohibited)
        if action is not None:
            return action, action_data, context_note

        # Retry once with explicit format reminder
        valid_names = ", ".join(sorted({getattr(a, "name", str(a)) for a in allowed_actions}))
        retry_note = (
            context_note
            + f"\n[FORMAT NOTICE] Last response was rejected: '{raw[:100]}'\n"
            + f"Required format: ACTION=<NAME> [X=<int> Y=<int>]. Choose from: {valid_names}"
        )
        raw_retry = self.brain.decide_action(
            game_id,
            level,
            s0_state,
            current_state,
            allowed_actions,
            retry_note,
            self.cache,
            world_model_block=world_model_block,
        )
        if raw_retry:
            self.world_model.update_from_text(raw_retry)
        action, action_data = ARCActionMapper.parse(raw_retry, allowed_actions, grid_shape, prohibited=prohibited)
        if action is not None:
            return action, action_data, context_note

        return self._safe_fallback(allowed_actions, state_hash, grid_shape, context_note)

    def _safe_fallback(
        self,
        allowed_actions: List[Any],
        state_hash: str,
        grid_shape: Optional[Tuple[int, int]],
        context_note: str,
    ) -> Tuple[Any, Dict[str, Any], str]:
        """Provides deterministic fallback when LLM output cannot be parsed."""
        tried = self.memory.tried_signatures(state_hash)

        # 1. Try untried simple action
        for a in allowed_actions:
            if not is_complex_action(a):
                sig = ActionSignature.from_action(a, {})
                if sig not in tried:
                    return a, {}, context_note + "\n[PARSER NOTICE] Fallback: untried simple action."

        # 2. Try untried coordinate for complex action
        for a in allowed_actions:
            if is_complex_action(a) and grid_shape:
                coord = self._untried_coordinate(a, state_hash, grid_shape)
                if coord is not None:
                    return (
                        a,
                        {"x": coord[0], "y": coord[1]},
                        context_note + f"\n[PARSER NOTICE] Fallback: untried coordinate {coord}.",
                    )

        # 3. Forced move fallback
        fallback_act = allowed_actions[0]
        fallback_data = {}
        if grid_shape and is_complex_action(fallback_act):
            coord = self._untried_coordinate(fallback_act, state_hash, grid_shape, force=True)
            if coord:
                fallback_data = {"x": coord[0], "y": coord[1]}
        return fallback_act, fallback_data, context_note + "\n[PARSER NOTICE] Fallback: forced move."

    def _untried_coordinate(
        self,
        action: Any,
        state_hash: str,
        grid_shape: Tuple[int, int],
        force: bool = False,
    ) -> Optional[Tuple[int, int]]:
        height, width = grid_shape
        name = getattr(action, "name", str(action)).upper()
        tried_coords = set(self.memory.tried_coords_for_action(state_hash, name))

        cx, cy = width // 2, height // 2
        if (cx, cy) not in tried_coords:
            return (cx, cy)

        max_radius = max(width, height)
        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    x, y = cx + dx, cy + dy
                    if 0 <= x < width and 0 <= y < height and (x, y) not in tried_coords:
                        return (x, y)

        return (cx, cy) if force else None

    def log_action(
        self,
        game_id: str,
        level: int,
        step_index: int,
        action_sig: Optional[ActionSignature],
        hash_before: str,
        hash_after: str,
    ) -> None:
        """Appends step entry to markdown actions log."""
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        action_name = action_sig.name if action_sig else "UNKNOWN"
        line = f"| {step_index} | {ts} | {action_name} | {hash_before[:12]} -> {hash_after[:12]} |\n"
        self.cache.append_action_log(game_id, level, line)

    def review_failed_iteration(
        self,
        game_id: str,
        level: int,
        iteration: int,
        s0_state: ARCState,
        final_state: ARCState,
    ) -> None:
        """Runs post-failure reflection chain and updates scratchpad rules."""
        render_live(
            final_state,
            status=f"🧠 Iteration {iteration} failed — reviewing full log & consolidating scratchpad...",
        )
        review_text = self.reviewer.review(
            game_id, level, iteration, s0_state, final_state, self.cache
        )
        apply_iteration_review(self.cache, game_id, level, iteration, review_text)
