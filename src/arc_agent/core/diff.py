"""NumPy-based fast visual diffing, HUD step counter filtering, and spatial bounding box extraction."""

from typing import Any, Optional, Set, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# HUD pixel registry — populated dynamically when single-pixel border changes
# are detected on non-gameplay actions. Cleared on game/level resets.
# ---------------------------------------------------------------------------
_HUD_PIXELS: Set[Tuple[int, int]] = set()
_HUD_BORDER_MARGIN: int = 5  # pixels within this many rows/cols from edge are HUD candidates
_HUD_MIN_GRID_DIMENSION: int = 32  # HUD filtering only activates on grids at least this wide/tall


def register_hud_pixel(x: int, y: int) -> None:
    """Mark (x, y) as a known HUD counter pixel to be excluded from gameplay diffs."""
    _HUD_PIXELS.add((x, y))


def clear_hud_pixels() -> None:
    """Clear all registered HUD pixels (call when starting a new game or level)."""
    _HUD_PIXELS.clear()


def get_hud_pixels() -> Set[Tuple[int, int]]:
    """Return the current set of registered HUD counter pixels."""
    return set(_HUD_PIXELS)


def _is_border_location(x: int, y: int, grid_shape: Tuple[int, int], margin: int = 5) -> bool:
    """True if pixel (x, y) lies within *margin* pixels of any grid edge."""
    h, w = grid_shape
    return x < margin or x >= w - margin or y < margin or y >= h - margin


def _is_hud_step_counter_change(
    grid1: np.ndarray,
    grid2: np.ndarray,
    action_name: str = "",
    register: bool = True,
) -> bool:
    """Detect whether the only difference between two grids is an on-screen HUD step counter pixel.

    Classifies a diff as a HUD counter update (not a gameplay change) when ALL of:
      - 1 or 2 pixels differ.
      - All changed pixels lie within the border margin (HUD zone).
      - The grid is large enough (>= 32px in one dimension) for the HUD to be plausible.
      - The action is an explicitly named simple directional action (not click/coordinate action).
        NOTE: empty action_name does NOT trigger HUD filtering — explicit name required.

    When *register* is True and the condition is met, the pixels are added to the
    persistent HUD pixel registry so future diffs mask them out automatically.
    """
    if grid1 is None or grid2 is None or grid1.shape != grid2.shape:
        return False

    # Require an explicit movement action name — empty/unknown actions must not suppress changes
    if not action_name:
        return False

    action_upper = action_name.upper()
    # Must be a directional/simple action, not click/coordinate
    is_movement_action = any(
        kw in action_upper
        for kw in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5",
                   "ACTION7", "UP", "DOWN", "LEFT", "RIGHT", "RESET", "UNDO", "INTERACT")
    ) and "ACTION6" not in action_upper and "CLICK" not in action_upper
    # Also match plain integer names: "1", "2", "3", "4", "5", "7"
    if not is_movement_action and action_upper in ("1", "2", "3", "4", "5", "7"):
        is_movement_action = True

    if not is_movement_action:
        return False

    diff = grid1 != grid2
    num_changes = int(np.sum(diff))

    if num_changes == 0 or num_changes > 4:
        return False

    h, w = grid1.shape
    # Only apply HUD detection on grids large enough to realistically have a HUD counter
    if h < _HUD_MIN_GRID_DIMENSION and w < _HUD_MIN_GRID_DIMENSION:
        return False

    y_indices, x_indices = np.where(diff)

    # All changed pixels must be in the HUD border zone
    for px_y, px_x in zip(y_indices.tolist(), x_indices.tolist()):
        if not _is_border_location(int(px_x), int(px_y), (h, w), _HUD_BORDER_MARGIN):
            return False

    # Register these pixels for future masking
    if register:
        for px_y, px_x in zip(y_indices.tolist(), x_indices.tolist()):
            register_hud_pixel(int(px_x), int(px_y))
        print(
            f"\U0001f515 [HUD FILTER] Registered {num_changes} border pixel(s) as HUD step counter: "
            + ", ".join(f"X={int(x)},Y={int(y)}" for x, y in zip(x_indices, y_indices))
        )

    return True


def extract_grid_array(obs: Any) -> Optional[np.ndarray]:
    """Safely extracts a 2D numpy array regardless of observation structure."""
    if obs is None:
        return None

    raw_g = getattr(obs, "grid", None)
    frame_obj = getattr(obs, "frame", raw_g)

    if frame_obj is None and raw_g is None:
        if isinstance(obs, (list, tuple, np.ndarray)):
            frame_obj = obs
        else:
            return None

    target = frame_obj if frame_obj is not None else raw_g
    try:
        arr = np.asarray(target)
        if arr.ndim == 3 and arr.shape[0] > 0:
            arr = arr[-1]
        if arr.ndim == 2 and arr.size > 0:
            return arr.astype(int)
    except Exception:
        pass
    return None


def get_gameplay_grid(grid: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Return the grid with registered HUD counter pixels masked to their pre-change value.

    If no HUD pixels are registered, returns the grid unchanged (zero copy cost).
    """
    if grid is None:
        return None
    if not _HUD_PIXELS:
        return grid
    # Only copy when we have pixels to mask
    masked = grid.copy()
    h, w = masked.shape
    for (px_x, px_y) in _HUD_PIXELS:
        if 0 <= px_y < h and 0 <= px_x < w:
            masked[px_y, px_x] = 0  # Neutral mask value
    return masked


def detect_real_change(
    grid1: Optional[np.ndarray],
    grid2: Optional[np.ndarray],
    action_name: str = "",
) -> bool:
    """Check for visible *gameplay* changes between two grids.

    Single-pixel or few-pixel changes that are:
      - Confined to the border HUD zone AND
      - Caused by a non-click directional action

    are classified as HUD step counter updates and return **False** (NO-OP for
    gameplay purposes), even though the raw grid bytes differ.

    Already-registered HUD pixels are masked out of the comparison first.
    """
    if grid1 is None or grid2 is None:
        return False
    if grid1.shape != grid2.shape:
        return True

    # First check: is this a pure HUD counter tick?
    if _is_hud_step_counter_change(grid1, grid2, action_name=action_name, register=True):
        return False

    # Normal comparison using masked gameplay grids
    gp_grid1 = get_gameplay_grid(grid1)
    gp_grid2 = get_gameplay_grid(grid2)

    return not np.array_equal(gp_grid1, gp_grid2)


def get_grid_difference_text(
    grid1: Optional[np.ndarray],
    grid2: Optional[np.ndarray],
    action_name: str = "",
) -> str:
    """Calculates grid changes instantly using NumPy to bypass slow vision calls.

    Returns a tight spatial bounding box including changes at the edges.
    HUD step counter updates are identified and reported as NO-OP instead of
    polluting the scratchpad with false change lines.
    """
    if grid1 is None or grid2 is None:
        return "Previous or current grid is unavailable."
    if grid1.shape != grid2.shape:
        return f"Grid size changed from {grid1.shape} to {grid2.shape}."

    # Check for HUD step counter (don't register again – detect_real_change already did)
    if _is_hud_step_counter_change(grid1, grid2, action_name=action_name, register=False):
        return "No visible gameplay changes detected (HUD step counter updated; NO-OP)."

    gp_grid1 = get_gameplay_grid(grid1)
    gp_grid2 = get_gameplay_grid(grid2)

    diff = (gp_grid1 != gp_grid2)
    num_changes = int(np.sum(diff))
    if num_changes == 0:
        return "No visible changes detected on this step (NO-OP); cause unknown."

    y_indices, x_indices = np.where(diff)
    min_x, max_x = int(np.min(x_indices)), int(np.max(x_indices))
    min_y, max_y = int(np.min(y_indices)), int(np.max(y_indices))

    return (
        f"{num_changes} pixels modified in bounding box "
        f"X=[{min_x}, {max_x}], Y=[{min_y}, {max_y}]."
    )


def extract_diff_bounding_box(
    grid1: Optional[np.ndarray], grid2: Optional[np.ndarray]
) -> Optional[Tuple[int, int, int, int]]:
    """Returns (min_x, max_x, min_y, max_y) in full grid coordinates of the gameplay diff."""
    if grid1 is None or grid2 is None or grid1.shape != grid2.shape:
        return None

    gp_grid1 = get_gameplay_grid(grid1)
    gp_grid2 = get_gameplay_grid(grid2)
    diff = (gp_grid1 != gp_grid2)
    if not np.any(diff):
        return None

    y_indices, x_indices = np.where(diff)
    min_x, max_x = int(np.min(x_indices)), int(np.max(x_indices))
    min_y, max_y = int(np.min(y_indices)), int(np.max(y_indices))

    return (min_x, max_x, min_y, max_y)
