"""Game and level execution runner with timeout monitoring and closed-loop step loops."""

from typing import Any, List, Optional, Tuple
import time

from ..core.actions import ARCActionMapper
from ..core.diff import extract_grid_array, get_grid_difference_text
from ..core.state import ARCState
from ..memory.knowledge import maybe_append_rule, update_verified_mechanics
from ..utils.display import render_live


from .arc_langchain_agent import ARCLangChainAgent

GLOBAL_START_TIME = time.time()


def is_time_budget_exhausted(budget_hours: float = 8.5) -> bool:
    """Checks if the process is approaching overall tournament runtime limit."""
    elapsed = time.time() - GLOBAL_START_TIME
    return elapsed > (budget_hours * 3600)


def get_dynamic_max_steps(
    obs: Any, base_multiplier: float = 1.5, min_limit: int = 25, max_limit: int = 80
) -> int:
    grid = extract_grid_array(obs)
    if grid is None:
        return 40
    height, width = grid.shape
    calculated_steps = int((height + width) * base_multiplier)
    return max(min_limit, min(max_limit, calculated_steps))


def get_max_steps_for_level(env: Any, level: int, fallback_obs: Any = None) -> int:
    try:
        if hasattr(env, "environment_info") and env.environment_info is not None:
            baseline_actions = getattr(env.environment_info, "baseline_actions", None)
            if baseline_actions and isinstance(baseline_actions, list):
                if 0 <= level - 1 < len(baseline_actions):
                    baseline = baseline_actions[level - 1]
                    return int(baseline * 3)
    except Exception:
        pass
    return get_dynamic_max_steps(fallback_obs)


class ARCRunner:
    """Executes single levels and multi-level sequential games."""

    def __init__(
        self,
        agent: ARCLangChainAgent,
        time_budget_hours: float = 8.5,
        max_iterations_per_level: int = 3,
    ):
        self.agent = agent
        self.time_budget_hours = time_budget_hours
        self.max_iterations_per_level = max_iterations_per_level

    def attempt_one_shot(
        self,
        game_id: str,
        level: int,
        env: Any,
        s0_state: ARCState,
        valid_actions: List[Any],
        max_steps: int,
    ) -> Tuple[ARCState, Optional[ARCState], Any, int, bool]:
        """Attempts speculative one-shot macro plan before falling back to step loop."""
        grid_shape = s0_state.grid.shape if s0_state.grid is not None else None
        world_model_block = self.agent.world_model.to_prompt_block()
        plan_text = self.agent.brain.one_shot_plan(
            game_id, level, s0_state, valid_actions, self.agent.cache, world_model_block=world_model_block
        )
        plan = ARCActionMapper.parse_plan(plan_text, valid_actions, grid_shape)

        if not plan:
            return s0_state, None, getattr(s0_state.raw_obs, "state", self.agent.resolver.NOT_FINISHED), 0, False

        steps_used = 0
        zero_diff_streak = 0
        prior_state = s0_state
        predecessor_of_prior: Optional[ARCState] = None
        initial_completed = s0_state.levels_completed

        max_plan_execution = min(25, len(plan))
        for action, action_data in plan[:max_plan_execution]:
            if is_time_budget_exhausted(self.time_budget_hours):
                print("⚠️ [TIMEOUT MONITOR] Time budget exhausted during speculative one-shot execution.")
                break

            if steps_used >= max_steps:
                break

            if action not in self.agent.memory.get_allowed_actions(prior_state.state_hash, valid_actions):
                break

            steps_used += 1
            action_name = getattr(action, "name", str(action)).upper()
            status_text = f"⚡ [Speculative Step {steps_used}] Executing: {action_name}"
            render_live(prior_state, status=status_text, label="One-Shot Plan Speculation")

            curr_state, transition, _, _ = self.agent.execute_action(
                game_id,
                level,
                env,
                prior_state,
                action,
                action_data,
                steps_used,
                tag=f"oneshot_{steps_used}",
            )

            diff = get_grid_difference_text(prior_state.grid, curr_state.grid)
            reasoning_summary = f"[ACTION]: {action_name} | {diff}"
            render_live(
                curr_state,
                status=status_text,
                label="One-Shot Plan Speculation",
                reasoning=reasoning_summary,
            )

            self.agent.log_action(
                game_id, level, steps_used, transition.action_sig, prior_state.state_hash, curr_state.state_hash
            )

            if curr_state.levels_completed > initial_completed or self.agent.resolver.is_win(curr_state.game_state):
                return curr_state, prior_state, curr_state.game_state, steps_used, True

            if self.agent.resolver.is_game_over(curr_state.game_state):
                return curr_state, prior_state, curr_state.game_state, steps_used, False

            if transition.changed is True:
                zero_diff_streak = 0
            elif transition.changed is False:
                zero_diff_streak += 1

            if zero_diff_streak >= self.agent.stuck_threshold:
                print(f"⚠️ Speculative plan hit stuck threshold ({self.agent.stuck_threshold} NOOPs). Falling back to closed loop.")
                break

            predecessor_of_prior = prior_state
            prior_state = curr_state

        return prior_state, predecessor_of_prior, prior_state.game_state, steps_used, False

    def run_step_loop(
        self,
        game_id: str,
        level: int,
        env: Any,
        s0_state: ARCState,
        predecessor_state: Optional[ARCState],
        curr_state: ARCState,
        valid_actions: List[Any],
        start_step: int,
        max_steps: int,
    ) -> Tuple[ARCState, Any, int]:
        """Closed-loop perception-action-reflection step execution loop."""
        current_state = curr_state
        debug_note = ""
        zero_diff_streak = 0
        step_count = start_step
        initial_completed = current_state.levels_completed
        visited_hashes = {s0_state.state_hash, current_state.state_hash}

        while step_count < max_steps:
            if is_time_budget_exhausted(self.time_budget_hours):
                print(f"⚠️ [TIMEOUT MONITOR] Exceeded budget during Step {step_count}. Returning.")
                return current_state, current_state.game_state, step_count

            # Dynamically refresh permitted actions from current frame metadata or env.action_space
            raw_obs = current_state.raw_obs
            current_valid_actions = (
                getattr(raw_obs, "available_actions", None)
                or getattr(raw_obs, "action_space", None)
                or (
                    raw_obs.metadata.get("available_actions")
                    if hasattr(raw_obs, "metadata") and isinstance(raw_obs.metadata, dict)
                    else None
                )
                or getattr(env, "action_space", None)
                or valid_actions
            )

            render_live(current_state, status=f"🔄 Step {step_count + 1}/{max_steps} — Brain deciding next action...")

            action, action_data, debug_note = self.agent.decide_action(
                game_id, level, s0_state, current_state, current_valid_actions, debug_note
            )

            step_count += 1
            action_name = getattr(action, "name", str(action)).upper()
            render_live(current_state, status=f"🚀 Step {step_count}/{max_steps} — Executing: {action_name}")

            next_state, next_transition, is_repeat_state, is_repeat_transition = self.agent.execute_action(
                game_id,
                level,
                env,
                current_state,
                action,
                action_data,
                step_count,
                tag=f"step_{step_count}",
            )

            diff = get_grid_difference_text(current_state.grid, next_state.grid)
            visual_analysis = ""

            if next_transition.changed is True:
                zero_diff_streak = 0
                if not is_repeat_transition:
                    render_live(next_state, status=f"👁️ Step {step_count}/{max_steps} — Running visual analysis...")
                    visual_analysis = self.agent.eye.analyse_visual(game_id, s0_state, next_transition, diff)
                else:
                    visual_analysis = "[CACHED] Matches previously verified transition pattern."
            elif next_transition.changed is False:
                zero_diff_streak += 1

            transition_key = (current_state.state_hash, next_transition.action_sig)
            if is_repeat_transition and transition_key in self.agent.memory.debugger_cache:
                debug_note = self.agent.memory.debugger_cache[transition_key] + "\n[CACHED] Reused previous validation."
            else:
                debug_note = self.agent.debugger.validate(
                    game_id,
                    level,
                    s0_state,
                    next_transition,
                    diff,
                    visual_analysis,
                    self.agent.cache,
                )
                self.agent.memory.debugger_cache[transition_key] = debug_note

            if visual_analysis:
                self.agent.world_model.update_from_text(visual_analysis)
            if debug_note:
                self.agent.world_model.update_from_text(debug_note)

            is_visited_loop = next_state.state_hash in visited_hashes
            visited_hashes.add(next_state.state_hash)
            if is_visited_loop:
                debug_note += "\n[WARNING] Action led back to a visited state. Try a different direction."

            maybe_append_rule(game_id, debug_note, is_repeat_state, next_transition.changed, self.agent.cache)

            status_line = (
                f"🎮 {game_id} | Lvl {level} | Step {step_count}/{max_steps} | "
                f"{'CHANGED' if next_transition.changed else 'NOOP'} | Hash: {next_state.state_hash[:8]}"
            )
            if zero_diff_streak >= self.agent.stuck_threshold:
                status_line += f" | ⚠️ stuck x{zero_diff_streak}"
            if is_repeat_state:
                status_line += " | 🔁 seen before"

            reasoning_summary = f"[{action_name}] {diff}\n[EYE] {visual_analysis}\n[DEBUG] {debug_note}"
            render_live(next_state, status=status_line, reasoning=reasoning_summary)

            self.agent.log_action(
                game_id,
                level,
                step_count,
                next_transition.action_sig,
                current_state.state_hash,
                next_state.state_hash,
            )

            if next_state.levels_completed > initial_completed or self.agent.resolver.is_terminal(next_state.game_state):
                return next_state, next_state.game_state, step_count

            current_state = next_state

        return current_state, current_state.game_state, step_count

    def play_level(
        self,
        game_id: str,
        level: int,
        env: Any,
        obs: Any,
        valid_actions: List[Any],
        max_steps: int = 50,
        is_first_level_of_game: bool = False,
        max_iterations: Optional[int] = None,
    ) -> Tuple[Any, Any, int]:
        """Plays a level with automatic retry iterations (lives) and failure meta-reviews."""
        iterations_limit = max_iterations if max_iterations is not None else self.max_iterations_per_level
        curr_obs = obs
        final_state, total_steps = None, 0
        state = None

        for iteration in range(1, iterations_limit + 1):
            if is_time_budget_exhausted(self.time_budget_hours):
                print(f"⚠️ [TIMEOUT MONITOR] Skipping remaining retries for Level {level}.")
                break

            if iteration > 1:
                curr_obs = env.reset() if hasattr(env, "reset") else env.step(None)
                self.agent.world_model.reset_level_fields()
                self.agent.cache.append_action_log(game_id, level, f"\n### --- RETRY ITERATION {iteration} (Life {iteration}/{iterations_limit}) ---\n")

            s0_state = self.agent.enter_level(
                game_id, level, curr_obs, is_first_level_of_game, valid_actions
            )

            curr_state, predecessor_state, state, steps_used, solved = self.attempt_one_shot(
                game_id, level, env, s0_state, valid_actions, max_steps
            )
            if solved:
                return curr_state.raw_obs, state, steps_used

            if self.agent.resolver.is_game_over(state):
                if iteration < iterations_limit:
                    self.agent.review_failed_iteration(game_id, level, iteration, s0_state, curr_state)
                final_state = curr_state
                continue

            final_state, state, total_steps = self.run_step_loop(
                game_id,
                level,
                env,
                s0_state,
                predecessor_state,
                curr_state,
                valid_actions,
                steps_used,
                max_steps,
            )

            if self.agent.resolver.is_win(state) or self.agent.resolver.is_level_up(state):
                return final_state.raw_obs, state, total_steps

            if iteration < iterations_limit:
                self.agent.review_failed_iteration(game_id, level, iteration, s0_state, final_state)

        return (final_state.raw_obs if final_state else curr_obs), state, total_steps

    def play_game(
        self,
        game_id: str,
        env: Any,
        obs: Any = None,
        max_levels: int = 10,
        max_steps_per_level: Optional[int] = None,
        max_iterations_per_level: Optional[int] = None,
    ) -> Any:
        """Executes full multi-level game progression."""
        if obs is None:
            obs = env.reset() if hasattr(env, "reset") else env.step(None)

        level = 1
        while level <= max_levels:
            if is_time_budget_exhausted(self.time_budget_hours):
                print(f"⚠️ [TIMEOUT MONITOR] Aborting Game {game_id} at Level {level} to preserve time budget.")
                break

            valid_actions = (
                getattr(obs, "available_actions", None)
                or getattr(obs, "action_space", None)
                or (
                    obs.metadata.get("available_actions")
                    if hasattr(obs, "metadata") and isinstance(obs.metadata, dict)
                    else None
                )
                or getattr(env, "action_space", [])
            )
            dynamic_max_steps = get_max_steps_for_level(env, level, fallback_obs=obs)

            if max_steps_per_level is not None:
                dynamic_max_steps = min(dynamic_max_steps, max_steps_per_level)

            obs, state, steps_used = self.play_level(
                game_id,
                level,
                env,
                obs,
                valid_actions,
                max_steps=dynamic_max_steps,
                is_first_level_of_game=(level == 1),
                max_iterations=max_iterations_per_level or self.max_iterations_per_level,
            )

            if self.agent.resolver.is_game_over(state) or self.agent.resolver.is_win(state):
                break
            if self.agent.resolver.is_level_up(state):
                level += 1
                continue

            obs_completed = getattr(obs, "levels_completed", level - 1)
            if obs_completed >= level:
                level = obs_completed + 1
                continue

            break

        return obs
