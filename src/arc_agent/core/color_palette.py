"""ARC-AGI-3 16-color palette and grid rendering helpers."""

from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

ARC_COLOR_PALETTE = [
    "#FFFFFF",  # 0 White
    "#CCCCCC",  # 1 Light Gray
    "#999999",  # 2 Gray
    "#666666",  # 3 Dark Gray
    "#333333",  # 4 Darker Gray
    "#000000",  # 5 Black
    "#E53AA3",  # 6 Magenta
    "#FF7BCC",  # 7 Light Pink
    "#F93C31",  # 8 Red
    "#1E93FF",  # 9 Blue
    "#88D8F1",  # 10 Light Blue
    "#FFDC00",  # 11 Yellow
    "#FF851B",  # 12 Orange
    "#921231",  # 13 Maroon
    "#4FCC30",  # 14 Green
    "#A356D6",  # 15 Purple
]

# RGB tuples for direct PIL / NumPy conversion
ARC_RGB_PALETTE = [
    (255, 255, 255),
    (204, 204, 204),
    (153, 153, 153),
    (102, 102, 102),
    (51, 51, 51),
    (0, 0, 0),
    (229, 58, 163),
    (255, 123, 204),
    (249, 60, 49),
    (30, 147, 255),
    (136, 216, 241),
    (255, 220, 0),
    (255, 133, 27),
    (146, 18, 49),
    (79, 204, 48),
    (163, 86, 214),
]

ARC_CMAP = mcolors.ListedColormap(ARC_COLOR_PALETTE)
ARC_NORM = mcolors.Normalize(vmin=0, vmax=15)


def render_grid_to_pil(grid_data: np.ndarray, scale: int = 16) -> Image.Image:
    """Fast conversion of 2D grid matrix to scaled PIL RGB Image."""
    grid = np.asarray(grid_data, dtype=int)
    if grid.ndim != 2 or grid.size == 0:
        grid = np.zeros((1, 1), dtype=int)

    height, width = grid.shape
    rgb_img = np.zeros((height, width, 3), dtype=np.uint8)
    for color_idx, rgb in enumerate(ARC_RGB_PALETTE):
        rgb_img[grid == color_idx] = rgb

    img = Image.fromarray(rgb_img, mode="RGB")
    if scale > 1:
        img = img.resize((width * scale, height * scale), Image.Resampling.NEAREST)
    return img


def render_grid_to_png(
    grid_data: np.ndarray,
    save_path: str | Path,
    draw_gridlines: bool = True,
    draw_coordinates: bool = True,
    dpi: int = 120,
) -> str:
    """Renders grid data with coordinates and gridlines for visual analysis."""
    grid = np.asarray(grid_data, dtype=int)
    if grid.ndim != 2 or grid.size == 0:
        grid = np.zeros((1, 1), dtype=int)

    height, width = grid.shape
    cell_size = 0.35
    fig_width = max(width * cell_size, 3.2)
    fig_height = max(height * cell_size, 3.2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor="#1E1E1E")
    ax.set_facecolor("#1E1E1E")

    ax.imshow(grid, cmap=ARC_CMAP, norm=ARC_NORM, interpolation="nearest", aspect="equal")

    if draw_gridlines:
        ax.set_xticks(np.arange(width + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(height + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="#555555", linestyle="-", linewidth=0.75)
        ax.tick_params(which="minor", size=0)

    if draw_coordinates:
        ax.set_xticks(np.arange(width))
        ax.set_yticks(np.arange(height))
        font_size = min(9, max(5, 120 // max(width, height)))
        ax.set_xticklabels([str(i) for i in range(width)], fontsize=font_size, color="#CCCCCC")
        ax.set_yticklabels([str(i) for i in range(height)], fontsize=font_size, color="#CCCCCC")
        ax.tick_params(axis="both", which="major", colors="#888888", length=2)
        ax.xaxis.set_ticks_position("both")
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_edgecolor("#870E25")
        spine.set_linewidth(2.0)

    plt.tight_layout(pad=0.25)
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(p), bbox_inches="tight", pad_inches=0.1, dpi=dpi, facecolor="#1E1E1E")
    plt.close(fig)
    return str(p)
