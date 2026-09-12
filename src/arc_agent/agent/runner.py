"""Game and level execution runner with timeout monitoring and closed-loop step loops."""

import os
from typing import Any, List, Optional, Tuple
import time

from ..core.actions import ARCActionMapper, is_complex_action
from ..core.diff import clear_hud_pixels, extract_grid_array, get_grid_difference_text
from ..core.state import ARCState, compute_transition
from ..memory.knowledge import maybe_append_rule
from ..utils.display import render_live, reset_live_display


from .arc_langchain_agent import ARCLangChainAgent

# ARC-AGI engine flag: ensure RESET only resets the active level, preserving completed levels
os.environ["ONLY_RESET_LEVELS"] = "true"

GLOBAL_START_TIME = time.time()


def is_time_budget_exhausted(budget_hours: float = 8.5) -> bool:
    """Checks if the process is approaching overall tournament runtime limit."""
    elapsed = time.time() - GLOBAL_START_TIME
    return elapsed > (budget_hours * 3600)


def get_dynamic_max_steps(
    obs: Any, base_multiplier: float = 2.0, min_limit: int = 35, max_limit: int = 120
) -> int:
    grid = extract_grid_array(obs)
    if grid is None:
        return 50
    height, width = grid.shape
    calculated_steps = int((height + width) * base_multiplier)
    return max(min_limit, min(max_limit, calculated_steps))


def get_max_steps_for_level(
    env: Any,
    level: int,
    fallback_obs: Any = None,
    baseline_multiplier: float = 3.0,
    remaining_budget: Optional[int] = None,
) -> int:
    """Calculates maximum allowed steps for a level attempt.
    
    In ARC-AGI-3, baseline_actions is the human/optimal baseline for scoring
    ((baseline / actions)^2 * 100), NOT a hard game termination limit.
    This function scales the baseline by baseline_multiplier (default 3.0x)
    so the agent has sufficient exploration budget to solve the level.
    """
    steps = None
    try:
        if hasattr(env, "environment_info") and env.environment_info is not None:
            baseline_actions = getattr(env.environment_info, "baseline_actions", None)
            if baseline_actions and isinstance(baseline_actions, list):
                if 0 <= level - 1 < len(baseline_actions):
                    baseline = int(baseline_actions[level - 1])
                    steps = max(1, int(baseline * baseline_multiplier))
    except Exception:
        pass
    if steps is None:
        steps = get_dynamic_max_steps(fallback_obs)

    if remaining_budget is not None and remaining_budget > 0:
        steps = min(steps, remaining_budget)
    return max(1, steps)


class ARCRunner:
    """Executes single levels and multi-level sequential games with budget and retry management."""

    def __init__(
        self,
        agent: ARCLangChainAgent,
        time_budget_hours: float = 8.5,
        max_iterations_per_level: int = 3,
        max_total_actions: Optional[int] = None,
        baseline_multiplier: float = 3.0,
        fast_step_eval: bool = False,
        full_eval_interval: int = 8,
        speculative_plan_max_steps: int = 25,
    ):
        self.agent = agent
        self.time_budget_hours = time_budget_hours
        self.max_iterations_per_level = max_iterations_per_level
        self.max_total_actions = max_total_actions
        self.baseline_multiplier = baseline_multiplier
        self.fast_step_eval = fast_step_eval
        self.last_stage_status = {}
        self.full_eval_interval = max(1, full_eval_interval)
        self.speculative_plan_max_steps = max(0, speculative_plan_max_steps)
        self._total_actions_taken: int = 0
        self._current_game_budget: Optional[int] = max_total_actions
        os.environ["ONLY_RESET_LEVELS"] = "true"

    @property
    def total_actions_taken(self) -> int:
        return self._total_actions_taken

    def should_stop_game(self) -> bool:
        """Returns True if total action budget for the game is exhausted."""
        if self._current_game_budget is not None and self._total_actions_taken >= self._current_game_budget:
            return True
        return False

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
        if self.speculative_plan_max_steps == 0:
            return s0_state, None, s0_state.game_state, 0, False
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

        max_plan_execution = min(self.speculative_plan_max_steps, self.full_eval_interval, len(plan))
        for action, action_data in plan[:max_plan_execution]:
            if is_time_budget_exhausted(self.time_budget_hours):
                print("⚠️ [TIMEOUT MONITOR] Time budget exhausted during speculative one-shot execution.")
                break

            if self.should_stop_game():
                print(f"⛔ [BUDGET] Total action budget exhausted ({self._total_actions_taken}/{self._current_game_budget}) during one-shot.")
                break

            if steps_used >= max_steps:
                break

            current_actions = (getattr(prior_state.raw_obs, "available_actions", None)
                               or getattr(env, "action_space", None) or valid_actions)
            if action not in self.agent.memory.get_allowed_actions(prior_state.state_hash, current_actions):
                break

            steps_used += 1
            self._total_actions_taken += 1
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
                game_id, level, steps_used, transition.action_sig, prior_state.state_hash, curr_state.state_hash,
                changed=transition.changed,
            )

            if (curr_state.levels_completed > initial_completed
                    or self.agent.resolver.is_win(curr_state.game_state)
                    or self.agent.resolver.is_level_up(curr_state.game_state)):
                return curr_state, prior_state, curr_state.game_state, steps_used, True

            if self.agent.resolver.is_game_over(curr_state.game_state):
                return curr_state, prior_state, curr_state.game_state, steps_used, False

            if transition.changed is True:
                zero_diff_streak = 0
            elif transition.changed is False:
                zero_diff_streak += 1

            predecessor_of_prior = prior_state
            prior_state = curr_state
            if transition.changed is None:
                # Missing visual evidence requires assessment before another speculative move.
                break
            if zero_diff_streak >= self.agent.stuck_threshold:
                print(f"⚠️ Speculative plan hit stuck threshold ({self.agent.stuck_threshold} NOOPs). Falling back to closed loop.")
                break

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
        iteration: int = 1,
        max_iterations: int = 3,
    ) -> Tuple[ARCState, Any, int]:
        """Closed-loop perception-action-reflection step execution loop."""
        current_state = curr_state
        observation_note = ""
        zero_diff_streak = 0
        step_count = start_step
        initial_completed = current_state.levels_completed
        visited_hashes = {s0_state.state_hash, current_state.state_hash}
        evaluation_failed = not getattr(getattr(self.agent.eye, "last_result", None), "ok", True)

        # Evaluate the final speculative transition before Brain selects another move.
        if (predecessor_state is not None and start_step < max_steps
                and not self.should_stop_game() and not is_time_budget_exhausted(self.time_budget_hours)):
            action_sig = self.agent.memory.trajectory[-1].action_sig
            transition = compute_transition(predecessor_state, current_state, action_sig)
            diff = get_grid_difference_text(predecessor_state.grid, current_state.grid)
            observation_note = self.agent.eye.analyse_visual(
                game_id, s0_state, transition, diff,
                intended_plan="Evaluate the final speculative action before replanning.",
                world_model_block=self.agent.world_model.to_prompt_block(),
            )
            evaluation_failed = not observation_note
            self.last_stage_status = {"vision": getattr(self.agent.eye, "last_result", None)}
            if not observation_note:
                observation_note = f"Recent findings: {diff}; goal progress unknown."
            self.agent.world_model.update_from_text(observation_note)
            maybe_append_rule(game_id, observation_note, False, transition.changed, self.agent.cache)
            if evaluation_failed:
                observation_note += "\n[REASSESS] Speculative transition evaluation unavailable; mechanics remain uncertain."

        while step_count < max_steps:
            if is_time_budget_exhausted(self.time_budget_hours):
                print(f"⚠️ [TIMEOUT MONITOR] Exceeded budget during Step {step_count}. Returning.")
                return current_state, current_state.game_state, step_count

            if self.should_stop_game():
                print(f"⛔ [BUDGET] Total action budget exhausted ({self._total_actions_taken}/{self._current_game_budget}) at Step {step_count}.")
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
            self.agent.set_action_space(current_valid_actions)

            budget_str = f" | Total {self._total_actions_taken}/{self._current_game_budget}" if self._current_game_budget else f" | Total {self._total_actions_taken}"
            if self._current_game_budget:
                remaining_moves = max(0, self._current_game_budget - self._total_actions_taken)
                budget_context = (
                    f"You have used {self._total_actions_taken} of {self._current_game_budget} total allowed moves across the game "
                    f"({remaining_moves} moves remaining). "
                    f"Current Level {level} attempt: Step {step_count + 1}/{max_steps} (Try {iteration}/{max_iterations})."
                )
            else:
                budget_context = (
                    f"You have used {self._total_actions_taken} moves so far. "
                    f"Current Level {level} attempt: Step {step_count + 1}/{max_steps} (Try {iteration}/{max_iterations})."
                )

            render_live(current_state, status=f"🔄 Step {step_count + 1}/{max_steps} (Try {iteration}/{max_iterations}){budget_str} — Brain deciding next action...")

            is_stuck = zero_diff_streak > 0
            action, action_data, observation_note = self.agent.decide_action(
                game_id, level, s0_state, current_state, current_valid_actions, observation_note, budget_context=budget_context, is_stuck=is_stuck
            )

            if getattr(self.agent, "decision_failed", False) or getattr(self.agent, "consecutive_parse_failures", 0) >= 6:
                print(
                    f"\n⛔ [HALT ON MODEL FAILURE] Aborting Level {level} step loop at Step {step_count + 1}: "
                    f"LLM produced {self.agent.consecutive_parse_failures} consecutive empty or unparseable outputs. "
                    "Halting immediately to protect the move budget from being burned on an unresponsive model!"
                )
                return current_state, current_state.game_state, step_count

            intended_plan = getattr(self.agent, "last_action_plan", "")
            expected_effect = getattr(self.agent, "last_expected_effect", "")
            world_model_before = self.agent.world_model.to_prompt_block()
            step_count += 1
            self._total_actions_taken += 1
            action_name = getattr(action, "name", str(action)).upper()
            render_live(current_state, status=f"🚀 Step {step_count}/{max_steps} (Try {iteration}/{max_iterations}){budget_str} — Executing: {action_name}")

            repeat_count = max(1, min(10, int(action_data.pop("repeat", 1)))) if not is_complex_action(action) else 1

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

            # Macro repeat execution for non-complex directional moves
            if (repeat_count > 1 and next_transition.changed is True
                    and next_state.levels_completed <= initial_completed
                    and not self.agent.resolver.is_terminal(next_state.game_state)
                    and not self.should_stop_game()
                    and not is_repeat_state and not is_repeat_transition):
                self.agent.log_action(
                    game_id,
                    level,
                    step_count,
                    next_transition.action_sig,
                    current_state.state_hash,
                    next_state.state_hash,
                    changed=next_transition.changed,
                )
                for _ in range(2, repeat_count + 1):
                    if step_count >= max_steps or self.should_stop_game():
                        break
                    sub_prev = next_state
                    step_count += 1
                    self._total_actions_taken += 1
                    next_state, next_transition, is_repeat_state, is_repeat_transition = self.agent.execute_action(
                        game_id,
                        level,
                        env,
                        sub_prev,
                        action,
                        action_data,
                        step_count,
                        tag=f"step_{step_count}",
                    )
                    self.agent.log_action(
                        game_id,
                        level,
                        step_count,
                        next_transition.action_sig,
                        sub_prev.state_hash,
                        next_state.state_hash,
                        changed=next_transition.changed,
                    )
                    current_state = sub_prev
                    if (next_transition.changed is False
                            or next_state.levels_completed > initial_completed
                            or self.agent.resolver.is_terminal(next_state.game_state)
                            or is_repeat_state or is_repeat_transition):
                        break

            diff = get_grid_difference_text(current_state.grid, next_state.grid, action_name=action_name)
            visual_analysis = ""
            observation_note = ""

            # Terminal/progress checks precede expensive model evaluation.
            if next_state.levels_completed > initial_completed or self.agent.resolver.is_terminal(next_state.game_state):
                if repeat_count <= 1:
                    self.agent.log_action(game_id, level, step_count, next_transition.action_sig,
                                          current_state.state_hash, next_state.state_hash,
                                          changed=next_transition.changed)
                return next_state, next_state.game_state, step_count

            fast_mode = self.fast_step_eval
            zero_diff_streak = zero_diff_streak + 1 if next_transition.changed is False else 0
            full_evaluation = (
                not fast_mode or next_transition.changed is None or evaluation_failed
                or (not fast_mode and next_transition.changed is True and (is_repeat_state or is_repeat_transition))
                or zero_diff_streak >= self.agent.stuck_threshold
                or step_count % self.full_eval_interval == 0
            )
            observation = (
                "No visible board change detected; cause unknown." if next_transition.changed is False
                else f"Board changed: {diff}; goal progress unknown." if next_transition.changed is True
                else "Before/after comparison unavailable; effect unknown."
            )
            if full_evaluation:
                visual_analysis = self.agent.eye.analyse_visual(
                    game_id, s0_state, next_transition, diff,
                    intended_plan=intended_plan, expected_effect=expected_effect,
                    world_model_block=world_model_before,
                )
                observation_note = visual_analysis
                evaluation_failed = not visual_analysis
                self.last_stage_status = {"vision": getattr(self.agent.eye, "last_result", None)}
            else:
                observation_note = f"Recent findings: {observation}"

            debugger = getattr(self.agent, "debugger", None)
            if debugger is not None:
                audit = debugger.analyse(current_state, next_state, next_transition.action_sig,
                    expected_effect, f"{diff}\n{observation_note}", world_model_before,
                    repeated=is_repeat_state or is_repeat_transition)
                self.last_stage_status["debugger"] = debugger.last_result
                if audit:
                    observation_note += "\n" + audit
                else:
                    self.agent.cache.archive(game_id, f"Debugger failure step {step_count}", debugger.last_result.error)

            # Only validated stage text enters beliefs. On failure retain measured facts.
            if observation_note:
                self.agent.world_model.update_from_text(observation_note)
            else:
                observation_note = f"Recent findings: {observation}"
                self.agent.world_model.update_from_text(observation_note)
            if evaluation_failed:
                observation_note += "\n[REASSESS] Visual evaluation unavailable; treat mechanics as uncertain and re-evaluate next step."

            is_visited_loop = next_state.state_hash in visited_hashes
            visited_hashes.add(next_state.state_hash)
            if is_visited_loop:
                observation_note += "\n[WARNING] Action led back to a visited state. Try a different direction."

            maybe_append_rule(game_id, observation_note, is_repeat_state, next_transition.changed, self.agent.cache)

            status_line = (
                f"🎮 {game_id} | Lvl {level} | Step {step_count}/{max_steps} (Try {iteration}/{max_iterations}){budget_str} | "
                f"{'CHANGED' if next_transition.changed else 'NOOP'} | Hash: {next_state.state_hash[:8]}"
            )
            if zero_diff_streak >= self.agent.stuck_threshold:
                status_line += f" | ⚠️ stuck x{zero_diff_streak}"
            if is_repeat_state:
                status_line += " | 🔁 seen before"

            reasoning_summary = f"[{action_name}] {diff}\n[EYE] {visual_analysis}\n[OBSERVATION] {observation_note}"
            render_live(next_state, status=status_line, reasoning=reasoning_summary)

            if repeat_count <= 1:
                self.agent.log_action(
                    game_id,
                    level,
                    step_count,
                    next_transition.action_sig,
                    current_state.state_hash,
                    next_state.state_hash,
                    changed=next_transition.changed,
                )

            if next_state.levels_completed > initial_completed or self.agent.resolver.is_terminal(next_state.game_state):
                return next_state, next_state.game_state, step_count

            self.agent.cache.append_experiment(game_id, level,
                f"\n### Try {iteration}, step {step_count}\n\n"
                f"Action: {next_transition.action_sig}\n\n"
                f"Prediction: {expected_effect or 'unknown'}\n\n"
                f"Measured change: {diff}\n\n{observation_note}\n")
            self.agent.persist_world_model(game_id)
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

            if self.should_stop_game():
                print(f"⛔ [BUDGET] Stopping Level {level} retries: action budget ({self._total_actions_taken}/{self._current_game_budget}) reached.")
                break

            # Reset parse failure counter at the start of every retry iteration
            self.agent.consecutive_parse_failures = 0

            if iteration > 1:
                curr_obs = env.reset() if hasattr(env, "reset") else env.step(None)
                self.agent.cache.append_action_log(game_id, level, f"\n### --- RETRY ITERATION {iteration} (Try {iteration}/{iterations_limit}) ---\n")

            s0_state = self.agent.enter_level(
                game_id, level, curr_obs, is_first_level_of_game and iteration == 1, valid_actions
            )

            curr_state, predecessor_state, state, steps_used, solved = self.attempt_one_shot(
                game_id, level, env, s0_state, valid_actions, max_steps
            )
            if solved:
                return curr_state.raw_obs, state, total_steps + steps_used

            if self.should_stop_game():
                print(f"⛔ [BUDGET] Stopping Level {level} after one-shot: action budget ({self._total_actions_taken}/{self._current_game_budget}) reached.")
                final_state = curr_state
                total_steps += steps_used
                break

            if self.agent.resolver.is_game_over(state):
                self.agent.review_failed_iteration(game_id, level, iteration, s0_state, curr_state)
                final_state = curr_state
                total_steps += steps_used
                continue

            final_state, state, attempt_steps = self.run_step_loop(
                game_id,
                level,
                env,
                s0_state,
                predecessor_state,
                curr_state,
                valid_actions,
                steps_used,
                max_steps,
                iteration=iteration,
                max_iterations=iterations_limit,
            )
            total_steps += attempt_steps
            if getattr(self.agent, "decision_failed", False) or self.agent.consecutive_parse_failures >= 6:
                print("Stopping remaining retries: decision inference failed; see memory_history.md.")
                break

            if (final_state.levels_completed > s0_state.levels_completed
                    or self.agent.resolver.is_win(state) or self.agent.resolver.is_level_up(state)):
                return final_state.raw_obs, state, total_steps

            if self.should_stop_game():
                print(f"⛔ [BUDGET] Stopping Level {level}: action budget reached after iteration {iteration}.")
                break

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
        max_total_actions: Optional[int] = None,
    ) -> Any:
        """Executes full multi-level game progression with total move budget allocation."""
        self._total_actions_taken = 0
        self.agent.consecutive_parse_failures = 0
        clear_hud_pixels()  # Reset HUD counter registry from any previous game
        reset_live_display()

        effective_iterations = max_iterations_per_level or self.max_iterations_per_level
        if max_total_actions is not None:
            self._current_game_budget = max_total_actions
        elif self.max_total_actions is not None:
            self._current_game_budget = self.max_total_actions
        else:
            self._current_game_budget = None
            try:
                if hasattr(env, "environment_info") and env.environment_info is not None:
                    baselines = getattr(env.environment_info, "baseline_actions", None)
                    if baselines and isinstance(baselines, list):
                        self._current_game_budget = sum(baselines) * effective_iterations
            except Exception:
                pass

        if self._current_game_budget is None:
            # Default generous per-game move budget if baselines are hidden (e.g. competition mode)
            self._current_game_budget = 1500

        print(f"🎯 [GAME BUDGET] Total allowed moves for {game_id}: {self._current_game_budget}")

        # Configure dynamic action space on agent if available from environment
        act_space = getattr(env, "action_space", None)
        if act_space and hasattr(self.agent, "set_action_space"):
            self.agent.set_action_space(act_space)
            print(f"🕹️ Configured dynamic action prompt for: {[getattr(a, 'name', str(a)) for a in act_space]}")

        if obs is None:
            obs = env.reset() if hasattr(env, "reset") else env.step(None)

        level = 1
        while level <= max_levels:
            if is_time_budget_exhausted(self.time_budget_hours):
                print(f"⚠️ [TIMEOUT MONITOR] Aborting Game {game_id} at Level {level} to preserve time budget.")
                break

            if self.should_stop_game():
                print(f"⛔ [BUDGET] Aborting Game {game_id} before Level {level}: action budget exhausted ({self._total_actions_taken}/{self._current_game_budget}).")
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
            remaining_budget = (self._current_game_budget - self._total_actions_taken) if self._current_game_budget else None
            dynamic_max_steps = get_max_steps_for_level(
                env,
                level,
                fallback_obs=obs,
                baseline_multiplier=self.baseline_multiplier,
                remaining_budget=remaining_budget,
            )

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
                max_iterations=effective_iterations,
            )

            if self.agent.resolver.is_game_over(state) or self.agent.resolver.is_win(state):
                break
            if self.should_stop_game():
                print(f"⛔ [BUDGET] Stopping Game {game_id} after Level {level}: action budget exhausted ({self._total_actions_taken}/{self._current_game_budget}).")
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
