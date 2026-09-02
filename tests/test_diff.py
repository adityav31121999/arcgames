"""Unit tests for grid difference calculation and edge border stripping."""

import numpy as np
import pytest

from arc_agent.core.diff import (
    detect_real_change,
    extract_diff_bounding_box,
    get_gameplay_grid,
    get_grid_difference_text,
)


def test_get_gameplay_grid_border_stripping():
    # 12x12 grid: removing 4px border yields 4x4 central gameplay region
    grid = np.zeros((12, 12), dtype=int)
    # Status bar edge change at (0, 0)
    grid[0, 0] = 5
    
    gp = get_gameplay_grid(grid)
    assert gp.shape == (4, 4)
    assert np.all(gp == 0)


def test_detect_real_change_ignores_status_bar():
    g1 = np.zeros((14, 14), dtype=int)
    g2 = np.zeros((14, 14), dtype=int)

    # Edge change only (status bar step counter)
    g2[0, 0] = 11
    g2[1, 0] = 12

    assert detect_real_change(g1, g2) is False

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
