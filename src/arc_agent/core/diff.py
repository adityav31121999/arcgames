"""NumPy-based fast visual diffing, HUD step counter filtering, and spatial bounding box extraction."""

from typing import Any, Optional, Set, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# HUD coordinates are tracked per-level. Clear them at game/level/reset boundaries.
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


def _find_hud_step_counter_pixels(
    grid1: Optional[np.ndarray],
    grid2: Optional[np.ndarray],
) -> Set[Tuple[int, int]]:
    """Identify step-counter / progress-bar pixels on outer boundaries.

    In ARC-AGI-3 games (such as m0r0), an environment step counter/timer
    advances along the outer boundary on action steps.
    Common formats:
    - Horizontal symmetric: 1 pixel on top row y=0 at x_t, and 1 pixel on bottom
      row y=H-1 at x_b, with x_t + x_b == W - 1 (or x_t == x_b).
    - Vertical symmetric: 1 pixel on left col x=0 at y_l, and 1 pixel on right
      col x=W-1 at y_r, with y_l + y_r == H - 1 (or y_l == y_r).
    """
    if grid1 is None or grid2 is None or grid1.shape != grid2.shape:
        return set()

    h, w = grid1.shape
    if h < 8 or w < 8:
        return set()

    diff = (grid1 != grid2)
    total_changed = int(np.sum(diff))
    if total_changed == 0:
        return set()

    hud_pixels: Set[Tuple[int, int]] = set()

    # Check horizontal opposite-boundary timer (top y=0 and bottom y=h-1)
    top_xs = np.where(diff[0, :])[0]
    bottom_xs = np.where(diff[h - 1, :])[0]
    if len(top_xs) == 1 and len(bottom_xs) == 1:
        xt, xb = int(top_xs[0]), int(bottom_xs[0])
        is_symmetric = (xt + xb == w - 1) or (xt == xb)
        if total_changed == 2:
            hud_pixels.add((xt, 0))
            hud_pixels.add((xb, h - 1))
        elif is_symmetric and not diff[1, xt] and not diff[h - 2, xb]:
            hud_pixels.add((xt, 0))
            hud_pixels.add((xb, h - 1))

    # Check vertical opposite-boundary timer (left x=0 and right x=w-1)
    left_ys = np.where(diff[:, 0])[0]
    right_ys = np.where(diff[:, w - 1])[0]
    if len(left_ys) == 1 and len(right_ys) == 1:
        yl, yr = int(left_ys[0]), int(right_ys[0])
        is_symmetric = (yl + yr == h - 1) or (yl == yr)
        if total_changed == 2:
            hud_pixels.add((0, yl))
            hud_pixels.add((w - 1, yr))
        elif is_symmetric and not diff[yl, 1] and not diff[yr, w - 2]:
            hud_pixels.add((0, yl))
            hud_pixels.add((w - 1, yr))

    return hud_pixels


def _is_hud_step_counter_change(
    grid1: np.ndarray,
    grid2: np.ndarray,
    action_name: str = "",
    register: bool = True,
) -> bool:
    """Report changes confined to registered HUD coordinates or detected step counter."""
    if grid1 is None or grid2 is None or grid1.shape != grid2.shape:
        return False

    diff = grid1 != grid2
    if not np.any(diff):
        return False

    ys, xs = np.where(diff)
    if all((int(x), int(y)) in _HUD_PIXELS for x, y in zip(xs, ys)):
        return True

    hud_candidates = _find_hud_step_counter_pixels(grid1, grid2)
    if hud_candidates and all((int(x), int(y)) in hud_candidates or (int(x), int(y)) in _HUD_PIXELS for x, y in zip(xs, ys)):
        if register:
            for px_x, px_y in hud_candidates:
                register_hud_pixel(px_x, px_y)
        return True

    return False


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
    """Compare full boards, excluding registered HUD coordinates and boundary step counters."""
    if grid1 is None or grid2 is None:
        return False
    if grid1.shape != grid2.shape:
        return True

    # Pure HUD / step-counter changes are NO-OP.
    if _is_hud_step_counter_change(grid1, grid2, action_name=action_name, register=True):
        return False

    # Also register any detected step-counter pixels so subsequent steps mask them
    hud_candidates = _find_hud_step_counter_pixels(grid1, grid2)
    for px_x, px_y in hud_candidates:
        register_hud_pixel(px_x, px_y)

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

    Returns a tight spatial bounding box excluding HUD step counter pixels.
    Changes confined to HUD step counter pixels are reported as NO-OP.
    """
    if grid1 is None or grid2 is None:
        return "Previous or current grid is unavailable."
    if grid1.shape != grid2.shape:
        return f"Grid size changed from {grid1.shape} to {grid2.shape}."

    if _is_hud_step_counter_change(grid1, grid2, action_name=action_name, register=False):
        return "No visible gameplay changes detected (HUD step counter updated; NO-OP)."

    hud_candidates = _find_hud_step_counter_pixels(grid1, grid2)
    gp_grid1 = get_gameplay_grid(grid1)
    gp_grid2 = get_gameplay_grid(grid2)
    if hud_candidates:
        if gp_grid1 is grid1:
            gp_grid1 = grid1.copy()
        if gp_grid2 is grid2:
            gp_grid2 = grid2.copy()
        for px_x, px_y in hud_candidates:
            gp_grid1[px_y, px_x] = 0
            gp_grid2[px_y, px_x] = 0

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

    if _is_hud_step_counter_change(grid1, grid2, register=False):
        return None

    hud_candidates = _find_hud_step_counter_pixels(grid1, grid2)
    gp_grid1 = get_gameplay_grid(grid1)
    gp_grid2 = get_gameplay_grid(grid2)
    if hud_candidates:
        if gp_grid1 is grid1:
            gp_grid1 = grid1.copy()
        if gp_grid2 is grid2:
            gp_grid2 = grid2.copy()
        for px_x, px_y in hud_candidates:
            gp_grid1[px_y, px_x] = 0
            gp_grid2[px_y, px_x] = 0

    diff = (gp_grid1 != gp_grid2)
    if not np.any(diff):
        return None

    y_indices, x_indices = np.where(diff)
    min_x, max_x = int(np.min(x_indices)), int(np.max(x_indices))
    min_y, max_y = int(np.min(y_indices)), int(np.max(y_indices))

    return (min_x, max_x, min_y, max_y)
