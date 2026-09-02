"""NumPy-based fast visual diffing and spatial bounding box extraction."""

from typing import Any, Optional, Tuple
import numpy as np


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
    """Crops out the outer 4-pixel border to isolate gameplay and ignore edge trackers."""
    if grid is None:
        return None
    height, width = grid.shape
    if height >= 10 and width >= 10:
        return grid[4:-4, 4:-4]
    return grid


def detect_real_change(grid1: Optional[np.ndarray], grid2: Optional[np.ndarray]) -> bool:
    """Checks if actual gameplay changes occurred, ignoring status bar edge changes."""
    if grid1 is None or grid2 is None:
        return False
    if grid1.shape != grid2.shape:
        return True

    gp_grid1 = get_gameplay_grid(grid1)
    gp_grid2 = get_gameplay_grid(grid2)

    return not np.array_equal(gp_grid1, gp_grid2)


def get_grid_difference_text(grid1: Optional[np.ndarray], grid2: Optional[np.ndarray]) -> str:
    """Calculates grid changes instantly using NumPy to bypass slow vision calls.

    Returns a tight spatial bounding box, excluding status bar ticks.
    """
    if grid1 is None or grid2 is None:
        return "Previous or current grid is unavailable."
    if grid1.shape != grid2.shape:
        return f"Grid size changed from {grid1.shape} to {grid2.shape}."

    gp_grid1 = get_gameplay_grid(grid1)
    gp_grid2 = get_gameplay_grid(grid2)

    diff = (gp_grid1 != gp_grid2)
    num_changes = int(np.sum(diff))
    if num_changes == 0:
        return "No visual changes occurred on this step (Move Blocked / NO-OP)."

    y_indices, x_indices = np.where(diff)
    min_x, max_x = int(np.min(x_indices)), int(np.max(x_indices))
    min_y, max_y = int(np.min(y_indices)), int(np.max(y_indices))

    offset = 4 if grid1.shape[0] >= 10 and grid1.shape[1] >= 10 else 0
    return (
        f"{num_changes} pixels modified in bounding box "
        f"X=[{min_x + offset}, {max_x + offset}], Y=[{min_y + offset}, {max_y + offset}]."
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

    offset = 4 if grid1.shape[0] >= 10 and grid1.shape[1] >= 10 else 0
    return (min_x + offset, max_x + offset, min_y + offset, max_y + offset)
