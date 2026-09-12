"""ARC Action models, signature tracking, and strict grammar parser."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import re

_ACTION_RE = re.compile(
    r"^\s*[-*]*\s*(?:(?:NEXT\s+)?ACTION\*{0,2}\s*(?:[:=]\s*(?:ACTION\s*[:=]\s*)?|\s+(?=\d))\s*([A-Za-z0-9_]+)|(ACTION[1-7]|RESET|UP|DOWN|LEFT|RIGHT|INTERACT|CLICK)\b)(.*)$",
    re.IGNORECASE,
)
_COORD_RE = re.compile(r"\bX\s*[:=]\s*(-?\d+)\D+Y\s*[:=]\s*(-?\d+)", re.IGNORECASE)

_ACTION_NAME_ALIASES: Dict[str, str] = {
    "UP": "ACTION1",
    "DOWN": "ACTION2",
    "LEFT": "ACTION3",
    "RIGHT": "ACTION4",
    "INTERACT": "ACTION5",
    "SELECT": "ACTION5",
    "EXECUTE": "ACTION5",
    "CLICK": "ACTION6",
    "MOUSE": "ACTION6",
    "UNDO": "ACTION7",
    "RESET": "RESET",
    "1": "ACTION1",
    "2": "ACTION2",
    "3": "ACTION3",
    "4": "ACTION4",
    "5": "ACTION5",
    "6": "ACTION6",
    "7": "ACTION7",
    "0": "RESET",
}


def is_complex_action(action: Any) -> bool:
    """Checks if action requires coordinates (e.g. Action 6)."""
    if action == 6 or getattr(action, "value", None) == 6:
        return True
    name = getattr(action, "name", str(action)).upper()
    if name == "6" or "ACTION_6" in name or "ACTION6" in name or "CLICK" in name:
        return True
    val = getattr(action, "is_complex", False)
    return val() if callable(val) else bool(val)


def canonical_action_name(action: Any) -> str:
    name = getattr(action, "name", str(action)).upper().replace("ACTION_", "ACTION")
    return _ACTION_NAME_ALIASES.get(name, name)


def validate_coordinates(x: int, y: int, grid_shape: Tuple[int, int]) -> bool:
    """Validates (x, y) are within grid dimensions."""
    height, width = grid_shape
    return (0 <= y < height) and (0 <= x < width)


def extract_coordinates(text: str, grid_shape: Optional[Tuple[int, int]] = None) -> Optional[Dict[str, int]]:
    """Extracts (x, y) coordinates across multiple formats (e.g. X=12 Y=34, (12, 34), ACTION6 12 34)."""
    # Pattern 1: X=12, Y=34
    m1 = re.search(r"\bX\s*[:=]\s*(-?\d+)\D+Y\s*[:=]\s*(-?\d+)", text, re.IGNORECASE)
    if m1:
        x, y = int(m1.group(1)), int(m1.group(2))
        if not grid_shape or validate_coordinates(x, y, grid_shape):
            return {"x": x, "y": y}
    # Pattern 2: Y=34, X=12
    m2 = re.search(r"\bY\s*[:=]\s*(-?\d+)\D+X\s*[:=]\s*(-?\d+)", text, re.IGNORECASE)
    if m2:
        y, x = int(m2.group(1)), int(m2.group(2))
        if not grid_shape or validate_coordinates(x, y, grid_shape):
            return {"x": x, "y": y}
    # Pattern 3: (12, 34) or [12, 34]
    m3 = re.search(r"[(\[]\s*(-?\d+)\s*[, ]\s*(-?\d+)\s*[)\]]", text)
    if m3:
        x, y = int(m3.group(1)), int(m3.group(2))
        if not grid_shape or validate_coordinates(x, y, grid_shape):
            return {"x": x, "y": y}
    # Pattern 4: ACTION6 12 34 or ACTION6 12, 34
    m4 = re.search(r"(?:ACTION6|ACTION_6)[^\d\n]*?(\d+)\s*[, \t]+\s*(\d+)", text, re.IGNORECASE)
    if m4:
        x, y = int(m4.group(1)), int(m4.group(2))
        if not grid_shape or validate_coordinates(x, y, grid_shape):
            return {"x": x, "y": y}
    return None


@dataclass(frozen=True)
class ActionSignature:
    name: str
    data: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    @classmethod
    def from_action(cls, action: Any, action_data: Optional[Dict[str, Any]] = None) -> "ActionSignature":
        name = canonical_action_name(action)
        data = tuple(sorted((action_data or {}).items())) if is_complex_action(action) else ()
        return cls(name=name, data=data)

    def __str__(self) -> str:
        if not self.data:
            return self.name
        params = ",".join(f"{k}={v}" for k, v in self.data)
        return f"{self.name}({params})"


class ARCActionMapper:
    @staticmethod
    def _find_action(name: str, available_actions: List[Any]) -> Optional[Any]:
        """Finds matching action supporting enums, integer IDs, string names, and directional aliases."""
        clean_name = name.strip().upper()
        # 1. Direct exact match by action name, str(), or .value
        for action in available_actions:
            act_name = getattr(action, "name", str(action)).upper()
            act_val = str(getattr(action, "value", action)).upper()
            if act_name == clean_name or act_val == clean_name:
                return action
            if act_name.replace("_", "") == clean_name.replace("_", ""):
                return action

        # 2. Match via canonical aliases (e.g. 'UP' -> 'ACTION1', '1' -> 'ACTION1')
        canonical = _ACTION_NAME_ALIASES.get(clean_name, clean_name)
        for action in available_actions:
            act_name = getattr(action, "name", str(action)).upper()
            act_val = str(getattr(action, "value", action)).upper()
            act_canonical = _ACTION_NAME_ALIASES.get(act_name, _ACTION_NAME_ALIASES.get(act_val, act_name))
            if act_name == canonical or act_val == canonical or act_canonical == canonical:
                return action
            if act_name.replace("_", "") == canonical.replace("_", ""):
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

        lines = [line.strip() for line in response_text.splitlines() if line.strip()]
        for line in reversed(lines):
            m = _ACTION_RE.match(line)
            if not m:
                continue
            act_candidate_str = m.group(1) or m.group(2)
            if not act_candidate_str:
                continue
            candidate = ARCActionMapper._find_action(act_candidate_str, available_actions)
            if candidate is None:
                continue
            selected_action = candidate
            rest = m.group(3) if len(m.groups()) >= 3 else ""

            coords = extract_coordinates(rest, grid_shape) or extract_coordinates(line, grid_shape) or extract_coordinates(response_text, grid_shape)
            if coords and is_complex_action(candidate):
                action_data = coords

            if is_complex_action(candidate) and not action_data:
                return None, {}
            break

        if selected_action is None:
            # Fallback search for action names or aliases mentioned in response_text
            matched_candidates = []
            for a in available_actions:
                act_name = getattr(a, "name", str(a)).upper()
                act_val = str(getattr(a, "value", a)).upper()
                if re.search(r"\b" + re.escape(act_name) + r"\b", response_text.upper()):
                    matched_candidates.append(a)
                elif act_val != act_name and re.search(r"\b" + re.escape(act_val) + r"\b", response_text.upper()):
                    matched_candidates.append(a)

            if len(matched_candidates) == 1:
                selected_action = matched_candidates[0]
                coords = extract_coordinates(response_text, grid_shape)
                if coords and is_complex_action(selected_action):
                    action_data = coords
            elif len(matched_candidates) > 1:
                # Pick the action candidate mentioned latest in response_text (decision at the end)
                last_pos = -1
                best_cand = None
                for a in matched_candidates:
                    act_name = getattr(a, "name", str(a)).upper()
                    act_val = str(getattr(a, "value", a)).upper()
                    for pattern in (r"\b" + re.escape(act_name) + r"\b", r"\b" + re.escape(act_val) + r"\b"):
                        for match in re.finditer(pattern, response_text.upper()):
                            if match.start() > last_pos:
                                last_pos = match.start()
                                best_cand = a
                if best_cand is not None:
                    selected_action = best_cand
                    coords = extract_coordinates(response_text, grid_shape)
                    if coords and is_complex_action(selected_action):
                        action_data = coords
            else:
                return None, {}

        if is_complex_action(selected_action) and not action_data:
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
