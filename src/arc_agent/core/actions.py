"""ARC Action models, signature tracking, and strict grammar parser."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import re

_ACTION_RE = re.compile(r"^\s*ACTION\s*[:=]\s*([A-Za-z0-9_]+)\s*(.*)$", re.IGNORECASE)
_COORD_RE = re.compile(r"\bX\s*[:=]\s*(-?\d+)\D+Y\s*[:=]\s*(-?\d+)", re.IGNORECASE)


def is_complex_action(action: Any) -> bool:
    """Checks if action requires coordinates (e.g. ACTION6)."""
    val = getattr(action, "is_complex", False)
    return val() if callable(val) else bool(val)


def validate_coordinates(x: int, y: int, grid_shape: Tuple[int, int]) -> bool:
    """Validates (x, y) are within grid dimensions."""
    height, width = grid_shape
    return (0 <= y < height) and (0 <= x < width)


@dataclass(frozen=True)
class ActionSignature:
    name: str
    data: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    @classmethod
    def from_action(cls, action: Any, action_data: Optional[Dict[str, Any]] = None) -> "ActionSignature":
        name = getattr(action, "name", str(action)).upper()
        data = tuple(sorted((action_data or {}).items()))
        return cls(name=name, data=data)

    def __str__(self) -> str:
        if not self.data:
            return self.name
        params = ",".join(f"{k}={v}" for k, v in self.data)
        return f"{self.name}({params})"


class ARCActionMapper:
    @staticmethod
    def _find_action(name: str, available_actions: List[Any]) -> Optional[Any]:
        name = name.upper()
        for action in available_actions:
            if getattr(action, "name", str(action)).upper() == name:
                return action
        return None

    @staticmethod
    def parse(
        response_text: str,
        available_actions: List[Any],
        grid_shape: Optional[Tuple[int, int]] = None,
        prohibited: Optional[Set[ActionSignature]] = None,
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        """Parses model response into (action_enum, action_data)."""
        if not available_actions:
            raise RuntimeError("Empty action space.")
        prohibited = prohibited or set()

        selected_action, action_data = None, {}

        for line in response_text.splitlines():
            m = _ACTION_RE.match(line)
            if not m:
                continue
            candidate = ARCActionMapper._find_action(m.group(1), available_actions)
            if candidate is None:
                continue
            selected_action = candidate
            rest = m.group(2)

            cm = _COORD_RE.search(rest) or _COORD_RE.search(response_text)
            if cm:
                x, y = int(cm.group(1)), int(cm.group(2))
                if not grid_shape or validate_coordinates(x, y, grid_shape):
                    action_data = {"x": x, "y": y}

            if is_complex_action(candidate) and not action_data:
                return None, {}
            break

        if selected_action is None:
            found = [
                a
                for a in available_actions
                if re.search(
                    r"\b" + re.escape(getattr(a, "name", str(a)).upper()) + r"\b",
                    response_text.upper(),
                )
            ]
            if len(found) == 1:
                selected_action = found[0]
                cm = _COORD_RE.search(response_text)
                if cm:
                    x, y = int(cm.group(1)), int(cm.group(2))
                    if not grid_shape or validate_coordinates(x, y, grid_shape):
                        action_data = {"x": x, "y": y}
            else:
                return None, {}

        sig = ActionSignature.from_action(selected_action, action_data)
        if sig in prohibited:
            return None, {}

        return selected_action, action_data

    @staticmethod
    def parse_plan(
        plan_text: str,
        available_actions: List[Any],
        grid_shape: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[Any, Dict[str, Any]]]:
        """Parses multi-line macro plan into sequential action tuples."""
        plan = []
        for line in plan_text.splitlines():
            if "ACTION" in line.upper():
                action, action_data = ARCActionMapper.parse(line, available_actions, grid_shape)
                if action is not None:
                    plan.append((action, action_data))
        return plan
