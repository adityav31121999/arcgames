"""Trajectory tracking, loop detection, oscillation prevention, and sprite region tracking."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from ..core.actions import ActionSignature, is_complex_action
from ..core.diff import get_gameplay_grid


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
        self.debugger_cache: Dict[Tuple[str, ActionSignature], str] = {}

    def reset(self, s0_hash: str) -> None:
        """Resets trajectory for a new level or retry iteration."""
        self.trajectory = [TrajectoryStep(s0_hash, None, None)]
        self.actions_tried_from_state = {}
        self.state_visit_count = {s0_hash: 1}
        self.transition_model = {}
        self.sprite_box = None
        self.state_history = [s0_hash]
        self.debugger_cache = {}

    def update_sprite_region(
        self, grid1: Optional[np.ndarray], grid2: Optional[np.ndarray]
    ) -> None:
        """Determines active operational zones across all state shifts, ignoring status bars."""
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
        min_x, max_x = int(np.min(x_indices)), int(np.max(x_indices))
        min_y, max_y = int(np.min(y_indices)), int(np.max(y_indices))

        offset = 4 if grid1.shape[0] >= 10 and grid1.shape[1] >= 10 else 0
        min_x += offset
        max_x += offset
        min_y += offset
        max_y += offset

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
            f"bounding box X=[{min_x}, {max_x}], Y=[{min_y}, {max_y}]. This is likely your controllable "
            f"cursor/sprite. Focus spatial references and coordinate moves on this active window! "
            f"Large, static background colored regions are terrain obstacles and will not shift."
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

        is_repeat_state = new_hash in self.state_visit_count
        self.state_visit_count[new_hash] = self.state_visit_count.get(new_hash, 0) + 1
        self.trajectory.append(TrajectoryStep(new_hash, action_sig, changed))
        self.state_history.append(new_hash)
        return is_repeat_state, is_repeat_transition

    def tried_signatures(self, state_hash: str) -> Set[ActionSignature]:
        return self.actions_tried_from_state.get(state_hash, set())

    def visits(self, state_hash: str) -> int:
        return self.state_visit_count.get(state_hash, 0)

    def recent_trajectory_text(self, n: int = 6) -> str:
        parts = []
        for step in self.trajectory[-n:]:
            if step.action_sig is None:
                parts.append(f"S0[{step.state_hash[:8]}]")
            else:
                tag = {True: "OK", False: "NOOP", None: "?"}.get(step.changed, "?")
                parts.append(f"--{step.action_sig.name}[{tag}]--> S[{step.state_hash[:8]}]")
        return " ".join(parts)

    def oscillation_target(self) -> Optional[str]:
        if len(self.state_history) < 3:
            return None
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
            f"[LOOP WARNING] State {state_hash[:8]} visited {visits}x. "
            f"Already tried from here: {tried_desc}. Simple (parameter-free) "
            f"repeats are pre-filtered from your action list. For coordinate "
            f"actions, you MUST pick a DIFFERENT X/Y than any listed above."
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
        tried = self.tried_signatures(state_hash)
        tried_simple_names = {sig.name for sig in tried if not sig.data}

        allowed = [
            a
            for a in valid_actions
            if is_complex_action(a) or getattr(a, "name", str(a)).upper() not in tried_simple_names
        ]

        osc_target = self.oscillation_target()
        if osc_target is not None:

            def _leads_to_oscillation(a: Any) -> bool:
                name = getattr(a, "name", str(a)).upper()
                for sig in tried:
                    if sig.name != name:
                        continue
                    result = self.transition_model.get((state_hash, sig))
                    if result and result[0] == osc_target:
                        return True
                return False

            non_oscillating = [a for a in allowed if not _leads_to_oscillation(a)]
            if non_oscillating:
                allowed = non_oscillating

        if len(self.trajectory) >= 5:
            recent = self.trajectory[-5:]
            recent_actions = [step.action_sig for step in recent if step.action_sig is not None]
            if len(recent_actions) == 5:
                first_act_name = recent_actions[0].name
                same_name = all(act.name == first_act_name for act in recent_actions)
                all_noop = all(step.changed is False for step in recent[1:])
                if same_name and all_noop:
                    trimmed = [
                        a
                        for a in allowed
                        if getattr(a, "name", str(a)).upper() != first_act_name.upper()
                    ]
                    if trimmed:
                        allowed = trimmed

        return allowed or list(valid_actions)
