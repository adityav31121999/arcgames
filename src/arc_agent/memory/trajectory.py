"""Trajectory tracking, loop detection, oscillation prevention, and sprite region tracking."""

from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from ..core.actions import canonical_action_name, ActionSignature, is_complex_action
from ..core.diff import get_gameplay_grid


class Outcome(Enum):
    CHANGED = "changed"
    NO_CHANGE = "no_change"


@dataclass
class TrajectoryStep:
    state_hash: str
    action_sig: Optional[ActionSignature]
    changed: Optional[bool]


class TrajectoryMemory:
    """Maintains trajectory history, repeat state detection, and spatial heuristics."""

    def __init__(self):
        self.trajectory: List[TrajectoryStep] = []
        self.actions_tried_from_state: Dict[str, Set[ActionSignature]] = {}
        self.state_visit_count: Dict[str, int] = {}
        self.transition_model: Dict[Tuple[str, ActionSignature], Tuple[str, Optional[bool]]] = {}
        self.sprite_box: Optional[Tuple[int, int, int, int]] = None
        self.state_history: List[str] = []

        # (state_hash, action_name) -> last observed outcome for that exact pair
        self.tried_pairs: Dict[Tuple[str, str], Outcome] = {}
        self.same_action_streak: int = 0
        self._last_action: Optional[str] = None
        self.noop_streak: int = 0
        self.momentum_streak: int = 0
        self.NOOP_STREAK_THRESHOLD: int = 3
        self.NOOP_HARD_BLOCK_REPEATS: int = 1

    def reset(self, s0_hash: str) -> None:
        """Resets trajectory for a new level or retry iteration."""
        self.trajectory = [TrajectoryStep(s0_hash, None, None)]
        self.actions_tried_from_state = {}
        self.state_visit_count = {s0_hash: 1}
        self.transition_model = {}
        self.sprite_box = None
        self.state_history = [s0_hash]
        self.tried_pairs = {}
        self.same_action_streak = 0
        self._last_action = None
        self.noop_streak = 0
        self.momentum_streak = 0

    def update_sprite_region(
        self, grid1: Optional[np.ndarray], grid2: Optional[np.ndarray]
    ) -> None:
        """Determines active operational zones across all state shifts, ignoring HUD and status bars."""
        if grid1 is None or grid2 is None or grid1.shape != grid2.shape:
            return
        gp_grid1 = get_gameplay_grid(grid1)
        gp_grid2 = get_gameplay_grid(grid2)
        if gp_grid1 is None or gp_grid2 is None or gp_grid1.shape != gp_grid2.shape:
            return
        diff = (gp_grid1 != gp_grid2)
        if not np.any(diff):
            return

        y_indices, x_indices = np.where(diff)

        # Exclude registered HUD pixels so the step counter never poisons sprite tracking
        from ..core.diff import get_hud_pixels
        hud_coords = get_hud_pixels()
        if hud_coords:
            keep_mask = np.array(
                [(int(x), int(y)) not in hud_coords for x, y in zip(x_indices, y_indices)],
                dtype=bool,
            )
            if not np.any(keep_mask):
                return  # All changed pixels are HUD — ignore
            x_indices = x_indices[keep_mask]
            y_indices = y_indices[keep_mask]

        min_x, max_x = int(np.min(x_indices)), int(np.max(x_indices))
        min_y, max_y = int(np.min(y_indices)), int(np.max(y_indices))


        if self.sprite_box is None:
            self.sprite_box = (min_x, max_x, min_y, max_y)
        else:
            prev_min_x, prev_max_x, prev_min_y, prev_max_y = self.sprite_box
            self.sprite_box = (
                min(prev_min_x, min_x),
                max(prev_max_x, max_x),
                min(prev_min_y, min_y),
                max(prev_max_y, max_y),
            )

    def get_sprite_guidance(self) -> str:
        """Generates dynamic spatial clues for model prompts."""
        if self.sprite_box is None:
            return ""
        min_x, max_x, min_y, max_y = self.sprite_box
        return (
            f"\n[SPRITE NAVIGATION HIGHLIGHT] Active shifts have historically occurred near "
            f"bounding box X=[{min_x}, {max_x}], Y=[{min_y}, {max_y}]. "
            f"This union can include multiple objects and status indicators; it does not identify a player. "
            f"Static regions have unknown roles. Only click actions take coordinates."
        )

    def record_transition(
        self,
        prev_hash: str,
        action_sig: ActionSignature,
        new_hash: str,
        changed: Optional[bool],
    ) -> Tuple[bool, bool]:
        """Registers step changes. Returns (is_repeat_state, is_repeat_transition)."""
        is_repeat_transition = (prev_hash, action_sig) in self.transition_model

        self.actions_tried_from_state.setdefault(prev_hash, set()).add(action_sig)
        self.transition_model[(prev_hash, action_sig)] = (new_hash, changed)

        action_name = getattr(action_sig, "name", str(action_sig)).upper()
        self.record_step(prev_hash, action_name, bool(changed))

        is_repeat_state = new_hash in self.state_visit_count
        self.state_visit_count[new_hash] = self.state_visit_count.get(new_hash, 0) + 1
        self.trajectory.append(TrajectoryStep(new_hash, action_sig, changed))
        self.state_history.append(new_hash)
        return is_repeat_state, is_repeat_transition

    def record_step(self, state_hash: str, action: str, changed: bool) -> None:
        """Call this once per step, right after computing the grid diff."""
        action_name = str(action).upper()
        outcome = Outcome.CHANGED if changed else Outcome.NO_CHANGE
        self.tried_pairs[(state_hash, action_name)] = outcome

        if changed:
            self.noop_streak = 0
            self.momentum_streak += 1
        else:
            self.noop_streak += 1
            self.momentum_streak = 0

        if action_name == self._last_action:
            self.same_action_streak += 1
        else:
            self.same_action_streak = 1
        self._last_action = action_name

    def is_action_blocked(self, state_hash: str, action: Any) -> bool:
        """
        The ONLY thing that blocks an action: this exact (state, action) pair
        has already been tried in this exact state and produced no change.
        Repeating an action that keeps producing change is never blocked.
        """
        name = canonical_action_name(action)
        outcome = self.tried_pairs.get((state_hash, name))
        return outcome == Outcome.NO_CHANGE

    def should_diversify(self) -> bool:
        """True when the board itself has been stuck (not the action)."""
        return self.noop_streak >= self.NOOP_STREAK_THRESHOLD

    def suggested_temperature(self, base_temp: float = 0.0) -> float:
        return 1.0 if self.should_diversify() else base_temp

    def context_notice(self, state_hash: str, action: Optional[str] = None) -> Optional[str]:
        """Text to inject into the prompt, distinguishing momentum from real loops."""
        if action and self.is_action_blocked(state_hash, action):
            return (
                f"[TRAJECTORY WARNING] {action} was already tried in this exact "
                f"board state and produced NO visible change. Do not repeat it "
                f"here unless a prerequisite changes first."
            )
        if self.momentum_streak >= 2 and self._last_action:
            return (
                f"[MOMENTUM] {self._last_action} has produced visible change {self.momentum_streak} "
                f"times in a row. Continuing to repeat it is expected if progress is ongoing."
            )
        if self.should_diversify():
            return (
                f"[TRAJECTORY WARNING] The board has not changed for {self.noop_streak} "
                f"consecutive steps regardless of action taken. Try a different action "
                f"or target than recent attempts."
            )
        return None

    def tried_signatures(self, state_hash: str) -> Set[ActionSignature]:
        return self.actions_tried_from_state.get(state_hash, set())

    def visits(self, state_hash: str) -> int:
        return self.state_visit_count.get(state_hash, 0)

    def recent_trajectory_text(self, n: int = 6) -> str:
        parts = []
        recent_steps = self.trajectory[-n:]
        for step in recent_steps:
            if step.action_sig is None:
                parts.append("Start")
            else:
                tag = {True: "Changed", False: "NO-OP", None: "?"}.get(step.changed, "?")
                parts.append(f"{step.action_sig} [{tag}]")
        return " -> ".join(parts) if parts else "No moves yet"

    def oscillation_target(self) -> Optional[str]:
        if len(self.state_history) < 3:
            return None
        if self.state_history[-1] == self.state_history[-3]:
            return self.state_history[-2]
        return self.state_history[-3]

    def tried_coords_for_action(self, state_hash: str, action_name: str) -> List[Tuple[int, int]]:
        coords = []
        for sig in self.tried_signatures(state_hash):
            if sig.name != action_name.upper():
                continue
            d = dict(sig.data)
            if "x" in d and "y" in d:
                coords.append((d["x"], d["y"]))
        return coords

    def loop_warning(self, state_hash: str) -> str:
        visits = self.visits(state_hash)
        tried = self.tried_signatures(state_hash)
        if visits <= 1 or not tried:
            return ""

        counts: Dict[str, int] = {}
        coord_notes: Dict[str, List[str]] = {}
        for sig in tried:
            counts[sig.name] = counts.get(sig.name, 0) + 1
            if sig.data:
                d = dict(sig.data)
                if "x" in d and "y" in d:
                    coord_notes.setdefault(sig.name, []).append(f"({d['x']},{d['y']})")

        parts = []
        for name, c in counts.items():
            if name in coord_notes:
                parts.append(f"{name} x{c} at {', '.join(coord_notes[name])}")
            else:
                parts.append(f"{name} x{c}")
        tried_desc = ", ".join(parts)

        return (
            f"[LOOP WARNING] Current board state visited {visits} times. "
            f"Already tried from here: {tried_desc}. "
            f"Pick an alternate untried direction or coordinate."
        )

    def consecutive_action_warning(self, threshold: int = 5) -> str:
        if len(self.trajectory) < threshold:
            return ""

        recent = self.trajectory[-threshold:]
        recent_actions = [step.action_sig for step in recent if step.action_sig is not None]
        if len(recent_actions) < threshold:
            return ""

        first_action_name = recent_actions[0].name
        same_name = all(act.name == first_action_name for act in recent_actions)

        if same_name:
            all_noop = all(step.changed is False for step in recent[1:])
            if all_noop:
                return (
                    f"\n[BEHAVIORAL WARNING] '{first_action_name}' has produced NO state change "
                    f"for {threshold} consecutive steps. You are stuck — choose a different "
                    f"action or different coordinates."
                )
            else:
                return (
                    f"\n[EXPLORATION WARNING] You have moved in the same direction '{first_action_name}' "
                    f"for {threshold} consecutive steps. If traversing a corridor, check for intersecting "
                    f"branching paths or turn at junctions rather than exhausting your step budget."
                )
        return ""

    def get_allowed_actions(self, state_hash: str, valid_actions: List[Any]) -> List[Any]:
        """
        Filter candidate action list down to ones not yet proven dead in this exact state.
        An action is ONLY blocked if that exact (state, action) pair already produced NO_CHANGE.
        Repeating an action that keeps producing change (momentum) is never blocked.
        """
        allowed = [a for a in valid_actions if not self.is_action_blocked(state_hash, a)]
        if not allowed:
            allowed = list(valid_actions)

        osc_target = self.oscillation_target()
        if osc_target is not None:

            def _leads_to_oscillation(a: Any) -> bool:
                name = canonical_action_name(a)
                for sig in self.tried_signatures(state_hash):
                    if sig.name != name:
                        continue
                    result = self.transition_model.get((state_hash, sig))
                    if result and result[0] == osc_target:
                        return True
                return False

            non_oscillating = [a for a in allowed if not _leads_to_oscillation(a)]
            if non_oscillating:
                allowed = non_oscillating

        if self.noop_streak >= self.NOOP_STREAK_THRESHOLD and self._last_action:
            trimmed = [
                a
                for a in allowed
                if canonical_action_name(a) != self._last_action
            ]
            if trimmed:
                allowed = trimmed

        return allowed or list(valid_actions)

    def allowed_actions(self, state_hash: str, candidate_actions: List[Any]) -> List[Any]:
        """Filter candidate action list down to ones not yet proven dead in this exact state."""
        return self.get_allowed_actions(state_hash, candidate_actions)
