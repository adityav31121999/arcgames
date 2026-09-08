"""NumPy-based fast visual diffing, HUD step counter filtering, and spatial bounding box extraction."""

from typing import Any, Optional, Set, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# HUD coordinates are supplied only by verified external environment information.
# Clear them at game/level/reset boundaries. Never infer HUD from border geometry.
_HUD_PIXELS: Set[Tuple[int, int]] = set()

def register_hud_pixel(x: int, y: int) -> None:
    """Mark (x, y) as a known HUD counter pixel to be excluded from gameplay diffs."""
    _HUD_PIXELS.add((x, y))


def clear_hud_pixels() -> None:
    """Clear all registered HUD pixels (call when starting a new game or level)."""
    _HUD_PIXELS.clear()


def get_hud_pixels() -> Set[Tuple[int, int]]:
    """Return the current set of registered HUD counter pixels."""
    return set(_HUD_PIXELS)


def _is_hud_step_counter_change(
    grid1: np.ndarray,
    grid2: np.ndarray,
    action_name: str = "",
    register: bool = True,
) -> bool:
    """Report changes confined to externally verified HUD coordinates.

    Never infer HUD from border location or action names. The ``register`` argument
    is retained for compatibility; this function does not mutate the registry.
    """
    if grid1 is None or grid2 is None or grid1.shape != grid2.shape:
        return False

    # Geometry and action names cannot establish that a pixel is HUD. Only mask
    # coordinates explicitly registered from external/verified environment data.
    diff = grid1 != grid2
    ys, xs = np.where(diff)
    return bool(len(xs)) and all((int(x), int(y)) in _HUD_PIXELS for x, y in zip(xs, ys))



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
    """Return the grid with registered HUD counter pixels masked to zero.

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
    """Compare full boards, excluding only externally verified HUD coordinates."""
    if grid1 is None or grid2 is None:
        return False
    if grid1.shape != grid2.shape:
        return True

    # Only externally verified HUD changes may be ignored.
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
    Changes confined to externally verified HUD pixels are reported as NO-OP.
    """
    if grid1 is None or grid2 is None:
        return "Previous or current grid is unavailable."
    if grid1.shape != grid2.shape:
        return f"Grid size changed from {grid1.shape} to {grid2.shape}."

    # Consult only the externally verified registry; never infer HUD from geometry.
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
