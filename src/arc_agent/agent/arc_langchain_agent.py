"""High-level LangChain-powered ARCAgent orchestrating chains, spatial memory, and heuristics."""

from datetime import datetime, timezone
from pathlib import Path
from dataclasses import asdict
import json
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..chains.brain import BrainChain
from ..chains.eye import EyeChain
from ..chains.prompts import build_system_prompt
from ..core.actions import canonical_action_name, ARCActionMapper, ActionSignature, is_complex_action
from ..core.object_detection import (
    detect_grid_objects,
    is_click_only,
    render_click_history,
    render_detected_objects,
)
from ..core.resolver import GameStateResolver
from ..core.state import ARCState, ARCTransition, compute_transition, save_step_state_json
from ..core.diff import clear_hud_pixels
from ..memory.knowledge import (
    KnowledgeCache,
    apply_iteration_review,
    init_knowledge_files,
    maybe_append_rule,
)
from ..memory.trajectory import TrajectoryMemory
from ..memory.world_model import WorldModel
from ..memory.visual_index import VisualWorldModelIndex, build_compare_assume_payload
from ..utils.display import render_live


class ARCLangChainAgent:
    """Core Agent coordinating LangChain Eye and Brain chains."""

    def __init__(
        self,
        eye_chain: EyeChain,
        brain_chain: BrainChain,
        resolver: GameStateResolver,
        stuck_threshold: int = 3,
        memory_root: str = "./agent_memory",
        vision_cache_dir: str = "/tmp/agent_vision",
        debugger_chain=None,
        halt_on_invalid_decision: bool = False,
    ):
        self.eye = eye_chain
        self.brain = brain_chain
        if debugger_chain is not None:
            self.debugger = debugger_chain
        self.halt_on_invalid_decision = halt_on_invalid_decision
        self.decision_failed = False
        self.resolver = resolver
        self.stuck_threshold = stuck_threshold
        self.memory_root = memory_root
        self.vision_cache_dir = vision_cache_dir

        self.memory = TrajectoryMemory()
        self.cache = KnowledgeCache(memory_root=memory_root)
        self.world_model = WorldModel()
        self.visual_index = VisualWorldModelIndex(agent_memory_dir=memory_root)
        self.consecutive_parse_failures: int = 0
        self.last_action_plan = ""
        self.last_expected_effect = ""
        self._level_key = None

    def set_action_space(self, action_space: Optional[Any]) -> None:
        """Dynamically builds and sets system prompts across all chains matching the actual action space."""
        sys_prompt = build_system_prompt(action_space)
        self.set_system_prompt(sys_prompt)

    def set_system_prompt(self, system_prompt: str) -> None:
        """Sets the system prompt across all LangChain chains."""
        if hasattr(self, "debugger"):
            self.debugger.set_system_prompt(system_prompt)
        if hasattr(self.eye, "set_system_prompt"):
            self.eye.set_system_prompt(system_prompt)
        if hasattr(self.brain, "set_system_prompt"):
            self.brain.set_system_prompt(system_prompt)

    def enter_level(
        self,
        game_id: str,
        level: int,
        obs: Any,
        is_first_level_of_game: bool,
        valid_actions: Optional[List[Any]] = None,
    ) -> ARCState:
        """Initializes state, memory, and performs initial visual analysis of S0."""
        clear_hud_pixels()
        if valid_actions:
            self.set_action_space(valid_actions)
        init_knowledge_files(game_id, level, valid_actions, memory_root=self.memory_root)
        self.cache.refresh_level(game_id, level)

        s0_state = ARCState.create(game_id, level, 0, obs, tag="S0")
        self.memory.reset(s0_state.state_hash)

        render_live(s0_state, status="S0 Initial State Setup")

        same_level_retry = self._level_key == (game_id, level)
        if not same_level_retry:
            self.world_model.reset_level_fields()
        self._level_key = (game_id, level)

        object_list = ""
        if s0_state.grid is not None:
            object_list = render_detected_objects(s0_state.grid)
        action_names = [canonical_action_name(a) for a in (valid_actions or [])]

        pil_s0 = s0_state.get_pil_image()
        if pil_s0 is not None:
            self.visual_index.save_level_s0(level, pil_s0)

        if is_first_level_of_game:
            eye_out = self.eye.assume(game_id, level, s0_state, self.cache, object_list=object_list, action_names=action_names)
        else:
            payload = build_compare_assume_payload(self.visual_index, level, pil_s0)
            prior_img = None
            for role, img in payload.get("images", []):
                if role == "Prior level S0":
                    prior_img = img
                    break
            eye_out = self.eye.compare_assume(
                game_id, level, s0_state, self.cache, object_list=object_list, action_names=action_names,
                prior_s0_image=prior_img, prior_world_model=payload.get("prior_world_model_text"),
            )

        if eye_out and not same_level_retry:
            self.world_model.update_from_text(eye_out)
            self.visual_index.save_world_model_text(level, self.world_model.to_prompt_block())
        elif eye_out:
            # The board reset; learned mechanisms and the review's next experiment persist.
            self.world_model.recent_findings = "Board reset for a new attempt. " + eye_out
            self.visual_index.save_world_model_text(level, self.world_model.to_prompt_block())
        self.persist_world_model(game_id)

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
        if not is_complex_action(action):
            action_data = {}
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
        observation_note: str,
        budget_context: str = "",
        is_stuck: bool = False,
    ) -> Tuple[Any, Dict[str, Any], str]:
        """Decides next action using Brain chain with formatting retries and heuristics fallbacks."""
        grid_shape = current_state.grid.shape if current_state.grid is not None else None
        self.last_action_plan = ""
        self.last_expected_effect = ""
        state_hash = current_state.state_hash

        allowed_actions = self.memory.get_allowed_actions(state_hash, valid_actions)
        warning = self.memory.loop_warning(state_hash)
        behavior_warning = self.memory.consecutive_action_warning(threshold=5)
        trajectory_ctx = self.memory.recent_trajectory_text()
        sprite_highlight = self.memory.get_sprite_guidance()

        # Dynamic temperature: escalate off noop_streak (board stuck) or loop warnings, NOT off momentum
        stuck_or_looping = bool(is_stuck) or bool(warning) or self.memory.should_diversify()
        temperature = 1.0 if stuck_or_looping else 0.0

        context_notice = self.memory.context_notice(state_hash)

        context_note = "\n".join(
            x
            for x in [observation_note, warning, context_notice, sprite_highlight, f"[TRAJECTORY] {trajectory_ctx}"]
            if x
        )
        # Only prohibit actions that already produced NO_CHANGE in this exact state
        prohibited = {
            sig for sig in self.memory.tried_signatures(state_hash)
            if self.memory.is_action_blocked(state_hash, sig.name)
        }

        world_model_block = self.world_model.to_prompt_block()

        object_list = ""
        click_history = ""
        if is_click_only(allowed_actions):
            object_list = render_detected_objects(current_state.grid)
            actions_log = self.cache.actions_log(game_id, level)
            click_history = render_click_history(self.memory.trajectory, actions_log_text=actions_log)

        raw = self.brain.decide_action(
            game_id,
            level,
            s0_state,
            current_state,
            allowed_actions,
            context_note,
            self.cache,
            world_model_block=world_model_block,
            budget_context=budget_context,
            object_list=object_list,
            click_history=click_history,
            temperature=temperature,
        )
        action, action_data = ARCActionMapper.parse(raw, allowed_actions, grid_shape, prohibited=prohibited)
        if action is not None:
            self._record_action_intent(raw)
            self.persist_world_model(game_id)
            self.decision_failed = False
            self.consecutive_parse_failures = 0
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
            budget_context=budget_context,
            object_list=object_list,
            click_history=click_history,
            temperature=temperature,
        )
        action, action_data = ARCActionMapper.parse(raw_retry, allowed_actions, grid_shape, prohibited=prohibited)
        if action is not None:
            self._record_action_intent(raw_retry)
            self.persist_world_model(game_id)
            self.decision_failed = False
            self.consecutive_parse_failures = 0
            return action, action_data, context_note

        # Distinguish syntactically valid output (e.g. valid action that was prohibited)
        # from genuine LLM corruption / empty response
        raw_valid_syntax, _ = ARCActionMapper.parse(raw_retry or raw, allowed_actions, grid_shape, prohibited=None)
        if raw_valid_syntax is not None:
            self.decision_failed = False
            self.consecutive_parse_failures = 0
        else:
            self.consecutive_parse_failures += 1
            self.decision_failed = self.halt_on_invalid_decision
            diagnostic = f"Brain decision rejected after retry. Final response: {raw_retry or raw}"
            self.cache.archive(game_id, "Decision inference failure", diagnostic)
            print(diagnostic[:1000])

        if self.consecutive_parse_failures >= 6:
            print(
                f"\n🚨 [CRITICAL LLM FAILURE] Model produced {self.consecutive_parse_failures} consecutive empty or unparseable responses! "
                "No valid action was parsed. Inspect memory_history.md for the actual inference/format error."
            )

        return self._safe_fallback(allowed_actions, state_hash, grid_shape, context_note, current_grid=current_state.grid)

    def _record_action_intent(self, text: str) -> None:
        intent = WorldModel()
        intent.update_from_text(text)
        self.last_action_plan = intent.current_plan
        self.last_expected_effect = intent.expected_effect
        self.world_model.update_from_text(text)

    def persist_world_model(self, game_id: str) -> None:
        directory = Path(self.memory_root) / game_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "world_model.json").write_text(json.dumps(asdict(self.world_model), indent=2), encoding="utf-8")
        (directory / "world_model.md").write_text(self.world_model.to_prompt_block(), encoding="utf-8")
        scratch = self.cache.scratch(game_id)
        goal = self.world_model.goal_model or "Unknown; choose a discriminating experiment."
        scratch = re.sub(r"(## OBJECTIVE\n).*?(?=\n## |\Z)",
                         lambda match: match.group(1) + "Hypothesis (unverified): " + goal + "\n",
                         scratch, flags=re.S)
        self.cache.write_scratch(game_id, scratch)

    def _safe_fallback(
        self,
        allowed_actions: List[Any],
        state_hash: str,
        grid_shape: Optional[Tuple[int, int]],
        context_note: str,
        current_grid: Optional[np.ndarray] = None,
    ) -> Tuple[Any, Dict[str, Any], str]:
        """Provides deterministic fallback when LLM output cannot be parsed."""
        # 1. If click-only game, target detected unclicked objects before blind coordinate search
        if is_click_only(allowed_actions) and current_grid is not None:
            detected = detect_grid_objects(current_grid)
            candidates = sorted((o for o in detected if not o["is_hud"]),
                                key=lambda o: (o["possible_hud"], o["area"]))
            tried_coords = set(self.memory.tried_coords_for_action(state_hash, "ACTION6"))
            for obj in candidates:
                cx, cy = obj["solid_click_point"]
                if (cx, cy) not in tried_coords:
                    return (
                        allowed_actions[0],
                        {"x": int(cx), "y": int(cy)},
                        context_note + f"\n[PARSER NOTICE] Fallback: targeted untried object #{obj['id']} at ({cx}, {cy}).",
                    )
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
        name = canonical_action_name(action)
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
        if action_sig:
            name = getattr(action_sig, "name", str(action_sig))
            data = getattr(action_sig, "data", None)
            if data:
                items = data.items() if hasattr(data, "items") else data
                try:
                    data_str = " ".join(f"{str(k).upper()}={v}" for k, v in items)
                    action_name = f"{name}({data_str})"
                except Exception:
                    action_name = str(action_sig)
            else:
                action_name = str(name)
        else:
            action_name = "UNKNOWN"
        result = "Changed" if hash_before != hash_after else "NO-OP (Unchanged)"
        line = f"| {step_index} | {ts} | {action_name} | {result} |\n"
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
        reviewer = getattr(self, "debugger", self.brain)
        review_text = reviewer.review(
            game_id, level, iteration, s0_state, final_state, self.cache,
            world_model_block=self.world_model.to_prompt_block(),
        )
        if not review_text:
            result = getattr(reviewer, "last_review_result", None)
            self.cache.archive(game_id, f"Review unavailable, try {iteration}",
                               getattr(result, "error", "No validated review returned"))
            return
        apply_iteration_review(self.cache, game_id, level, iteration, review_text)
        self.world_model.update_from_text(review_text)
        self.persist_world_model(game_id)
