"""Unit tests for full-board differences and original-grid coordinates."""

import numpy as np
import pytest

from arc_agent.core.diff import (
    detect_real_change,
    extract_diff_bounding_box,
    get_gameplay_grid,
    get_grid_difference_text,
    get_hud_pixels,
)


def test_get_gameplay_grid_preserves_borders():
    grid = np.zeros((12, 12), dtype=int)
    grid[0, 0] = 5
    
    gp = get_gameplay_grid(grid)
    assert gp.shape == grid.shape
    assert gp[0, 0] == 5


def test_detect_real_change_includes_edges():
    g1 = np.zeros((14, 14), dtype=int)
    g2 = np.zeros((14, 14), dtype=int)

    # Edge changes cannot be assumed to be a status bar.
    g2[0, 0] = 11
    g2[1, 0] = 12

    assert detect_real_change(g1, g2) is True

    # Central gameplay change
    g2[6, 6] = 9
    assert detect_real_change(g1, g2) is True


def test_grid_difference_text_and_bbox():
    g1 = np.zeros((16, 16), dtype=int)
    g2 = np.zeros((16, 16), dtype=int)
    g2[6, 7] = 8  # Red pixel at (Y=6, X=7)

    diff_text = get_grid_difference_text(g1, g2)
    assert "1 pixels modified" in diff_text
    assert "X=[7, 7]" in diff_text
    assert "Y=[6, 6]" in diff_text

    bbox = extract_diff_bounding_box(g1, g2)
    assert bbox == (7, 7, 6, 6)


@pytest.mark.parametrize("shape", [(8, 8), (10, 12), (16, 20), (64, 64)])
@pytest.mark.parametrize("location", ["top", "bottom", "left", "right", "corner"])
def test_border_changes_keep_original_coordinates(shape, location):
    height, width = shape
    x, y = {
        "top": (width // 2, 1),
        "bottom": (width // 2, height - 2),
        "left": (1, height // 2),
        "right": (width - 2, height // 2),
        "corner": (width - 1, height - 1),
    }[location]
    before = np.zeros(shape, dtype=int)
    after = before.copy()
    after[y, x] = 9
    assert detect_real_change(before, after)
    assert extract_diff_bounding_box(before, after) == (x, x, y, y)
    description = get_grid_difference_text(before, after)
    assert f"X=[{x}, {x}], Y=[{y}, {y}]" in description
    assert "NO-OP" not in description


def test_unchanged_board_does_not_infer_collision():
    grid = np.zeros((16, 16), dtype=int)
    assert not detect_real_change(grid, grid.copy())
    assert extract_diff_bounding_box(grid, grid.copy()) is None
    description = get_grid_difference_text(grid, grid.copy()).lower()
    assert "cause unknown" in description
    assert "blocked" not in description


def test_hud_step_counter_two_pixels_is_noop():
    # 64x64 grid matching m0r0-492f87ba environment
    before = np.zeros((64, 64), dtype=int)
    after = before.copy()
    # 2 pixels on opposite boundaries: (4, 0) and (59, 63) where 4 + 59 = 63
    after[0, 4] = 1
    after[63, 59] = 1

    assert detect_real_change(before, after) is False
    assert extract_diff_bounding_box(before, after) is None
    description = get_grid_difference_text(before, after)
    assert "NO-OP" in description
    assert (4, 0) in get_hud_pixels()
    assert (59, 63) in get_hud_pixels()


def test_hud_step_counter_masked_during_gameplay_move():
    before = np.zeros((64, 64), dtype=int)
    before[49:59, 14:49] = 8  # 10x35 block
    after = np.zeros((64, 64), dtype=int)
    after[48:58, 14:49] = 8  # shifted up 1 row
    # Step counter ticks at the same time
    after[0, 4] = 1
    after[63, 59] = 1

    assert detect_real_change(before, after) is True
    bbox = extract_diff_bounding_box(before, after)
    assert bbox == (14, 48, 48, 58)  # True gameplay bbox, rows 0 and 63 excluded!
    description = get_grid_difference_text(before, after)
    assert "X=[14, 48], Y=[48, 58]" in description
    assert "Y=[0, 63]" not in description


def test_vertical_hud_step_counter_is_noop():
    before = np.zeros((64, 64), dtype=int)
    after = before.copy()
    after[10, 0] = 1
    after[53, 63] = 1  # 10 + 53 = 63

    assert detect_real_change(before, after) is False
    assert extract_diff_bounding_box(before, after) is None
    description = get_grid_difference_text(before, after)
    assert "NO-OP" in description
