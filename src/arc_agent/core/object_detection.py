"""Connected component segmentation, HUD filtering, and coordinate extraction for click-only games."""

from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import re
import numpy as np
from .diff import get_hud_pixels

# Human-readable color names for standard 16 ARC colors
COLOR_NAMES = {
    0: "White",
    1: "Light Gray",
    2: "Gray",
    3: "Dark Gray",
    4: "Darker Gray",
    5: "Black",
    6: "Magenta",
    7: "Light Pink",
    8: "Red",
    9: "Blue",
    10: "Light Blue",
    11: "Yellow",
    12: "Orange",
    13: "Maroon",
    14: "Green",
    15: "Purple",
}


def is_click_only(action_space: Optional[Iterable[Any]]) -> bool:
    """Checks if the available action space is exclusively ACTION6 (click at x,y)."""
    if not action_space:
        return False
    names = [
        getattr(a, "name", str(a)).upper()
        for a in action_space
        if getattr(a, "name", str(a)).upper() not in ("RESET", "0", "GAMEACTION.RESET")
    ]
    if not names:
        return False
    return all(n in ("ACTION6", "ACTION_6", "6", "CLICK", "MOUSE") for n in names)


def detect_grid_objects(
    grid: Optional[np.ndarray],
    border_margin: int = 4,
    ignore_hud_corners: bool = True,
    max_area_ratio: float = 0.25,
) -> List[Dict[str, Any]]:
    """Segments objects in the grid using monochromatic flood-fill, followed by hierarchical containment merge.

    1. Monochromatic 8-connected flood fill creates base components of uniform color.
       This prevents accidental fusion of adjacent, unrelated objects (e.g. player token touching a wall).
    2. Hierarchical containment pass: If component B is fully nested inside component A's bounding box
       (e.g. a bracket containing an inner marker/core), B is merged into A to form a composite object.
    3. Role hints:
       - Large components, edge strips, and corner glyphs remain possible targets.
       - Their geometry sets possible_hud, which is an unverified role hint.
       - Only components entirely inside externally verified HUD pixels set is_hud.

    Returns a list of dicts for each detected object with:
    - id: int
    - bbox: (min_x, min_y, max_x, max_y)
    - width: int, height: int
    - area: int (pixel count)
    - colors: List[int]
    - color_names: List[str]
    - visual_center: (cx, cy)
    - solid_click_point: (x, y) guaranteed to sit on an actual pixel of the object
    - shape_desc: str (bracket, cluster, bar, etc.)
    - is_hud: bool
    """
    if grid is None or grid.size == 0:
        return []

    h_grid, w_grid = grid.shape
    if h_grid < 2 or w_grid < 2:
        return []

    total_pixels = h_grid * w_grid

    # Determine background color: most frequent color across the grid
    counts = np.bincount(grid.ravel())
    bg_color = int(np.argmax(counts))

    non_bg = np.argwhere(grid != bg_color)
    if len(non_bg) == 0:
        return []

    # Step 1: Monochromatic 8-connected flood fill
    visited: Set[Tuple[int, int]] = set()
    raw_components: List[Dict[str, Any]] = []

    for r_init, c_init in non_bg:
        coord_init = (int(r_init), int(c_init))
        if coord_init in visited:
            continue

        target_color = int(grid[r_init, c_init])
        comp_pixels: List[Tuple[int, int]] = []
        queue = deque([coord_init])
        visited.add(coord_init)

        while queue:
            curr_r, curr_c = queue.popleft()
            comp_pixels.append((curr_c, curr_r))  # (x, y)

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < h_grid and 0 <= nc < w_grid:
                        if (nr, nc) not in visited and grid[nr, nc] == target_color:
                            visited.add((nr, nc))
                            queue.append((nr, nc))

        xs = [p[0] for p in comp_pixels]
        ys = [p[1] for p in comp_pixels]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        raw_components.append({
            "pixels": comp_pixels,
            "colors": {target_color},
            "bbox": (min_x, min_y, max_x, max_y),
            "area": len(comp_pixels),
            "bbox_area": (max_x - min_x + 1) * (max_y - min_y + 1),
        })

    # Step 2: Hierarchical containment pass
    # Sort by bounding box area descending so larger containers absorb nested inner cores
    raw_components.sort(key=lambda c: c["bbox_area"], reverse=True)

    merged_components: List[Dict[str, Any]] = []
    absorbed: Set[int] = set()

    for i, outer in enumerate(raw_components):
        if i in absorbed:
            continue
        out_min_x, out_min_y, out_max_x, out_max_y = outer["bbox"]
        for j in range(i + 1, len(raw_components)):
            if j in absorbed:
                continue
            inner = raw_components[j]
            in_min_x, in_min_y, in_max_x, in_max_y = inner["bbox"]

            # Merge only if inner is strictly contained within outer's bounding box
            # and outer is strictly larger in area (e.g. bracket housing an inner marker)
            if (
                out_min_x <= in_min_x
                and in_max_x <= out_max_x
                and out_min_y <= in_min_y
                and in_max_y <= out_max_y
                and outer["bbox_area"] > inner["bbox_area"]
            ):
                outer["pixels"].extend(inner["pixels"])
                outer["colors"].update(inner["colors"])
                absorbed.add(j)
        merged_components.append(outer)

    # Step 3: Build object representations and apply HUD / Background filters
    detected: List[Dict[str, Any]] = []
    obj_id = 1

    for comp in merged_components:
        comp_pixels = comp["pixels"]
        xs = [p[0] for p in comp_pixels]
        ys = [p[1] for p in comp_pixels]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        area = len(comp_pixels)

        # Geometric center of the bounding box
        cx = (min_x + max_x) // 2
        cy = (min_y + max_y) // 2

        # Crucial: If visual center is in a hollow cavity (like inside a bracket '[' or 'C'),
        # pick the solid pixel of the object closest to the center so click registers on solid surface
        pixel_set = set(comp_pixels)
        if (cx, cy) in pixel_set:
            solid_click_point = (cx, cy)
        else:
            solid_click_point = min(comp_pixels, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)

        # Colors present
        unique_colors = sorted(list(comp["colors"]))
        color_labels = [COLOR_NAMES.get(c, f"Color {c}") for c in unique_colors]

        # Shape heuristic classification
        density = area / float(width * height)
        if area == 1:
            shape_desc = "single pixel"
        elif area == width * height:
            shape_desc = "solid block"
        elif width >= 3 * height:
            shape_desc = "horizontal bar"
        elif height >= 3 * width:
            shape_desc = "vertical bar"
        elif density < 0.65:
            shape_desc = "bracket / hollow shape"
        else:
            shape_desc = "compact cluster"

        # HUD / Tracker / Background Detection:
        # 1. Huge monolithic walls / floors spanning large area
        is_large_background = area >= int(total_pixels * max_area_ratio)

        # 2. Outer frame touching all 4 edges
        touches_all_borders = (
            min_x <= border_margin
            and max_x >= w_grid - 1 - border_margin
            and min_y <= border_margin
            and max_y >= h_grid - 1 - border_margin
            and area >= 40
        )

        # 3. Edge strips flush with outer border margin
        is_edge_strip = (
            (min_y < border_margin and max_y < border_margin)
            or (min_y >= h_grid - border_margin)
            or (min_x < border_margin and max_x < border_margin)
            or (min_x >= w_grid - border_margin)
        )

        # 4. Corner markers (small repeating diamond/square glyphs)
        is_corner_hud = False
        if ignore_hud_corners and area <= 8:
            corner_thresh = 6
            in_tl = (max_x <= corner_thresh and max_y <= corner_thresh)
            in_tr = (min_x >= w_grid - 1 - corner_thresh and max_y <= corner_thresh)
            in_bl = (max_x <= corner_thresh and min_y >= h_grid - 1 - corner_thresh)
            in_br = (min_x >= w_grid - 1 - corner_thresh and min_y >= h_grid - 1 - corner_thresh)
            is_corner_hud = in_tl or in_tr or in_bl or in_br

        verified_hud = get_hud_pixels()
        is_hud = bool(comp_pixels) and all(p in verified_hud for p in comp_pixels)
        # Geometry is a hypothesis, never grounds for hiding a possible target.
        possible_hud = is_large_background or touches_all_borders or is_edge_strip or is_corner_hud

        detected.append({
            "id": obj_id,
            "bbox": (min_x, min_y, max_x, max_y),
            "width": width,
            "height": height,
            "area": area,
            "colors": unique_colors,
            "color_names": color_labels,
            "visual_center": (cx, cy),
            "solid_click_point": solid_click_point,
            "shape_desc": shape_desc,
            "is_hud": is_hud,
            "possible_hud": possible_hud,
        })
        obj_id += 1

    return detected


def render_detected_objects(grid: Optional[np.ndarray]) -> str:
    """Formats detected interactive objects into a clear, numbered list for the LLM."""
    if grid is None or grid.size == 0:
        return "No visual grid available."

    objects = detect_grid_objects(grid)
    # Exclude only verified HUD; retain uncertain border and background candidates.
    candidates = [o for o in objects if not o["is_hud"]]
    candidates.sort(key=lambda o: (o["possible_hud"], o["area"]))

    if not candidates:
        return "No distinct foreground objects detected."

    lines = []
    for obj in candidates:
        color_str = "/".join(obj["color_names"])
        cx, cy = obj["solid_click_point"]
        min_x, min_y, max_x, max_y = obj["bbox"]
        lines.append(
            f"- Object #{obj['id']}: {color_str} {obj['shape_desc']} "
            f"(Area: {obj['area']}px, BBox: X=[{min_x}..{max_x}], Y=[{min_y}..{max_y}]) "
            f"-> Center: ACTION=ACTION6 X={cx} Y={cy}"
            + (" (role uncertain: possible HUD/background)" if obj["possible_hud"] else "")
        )
    return "\n".join(lines)


def render_click_history(
    trajectory_or_steps: Optional[List[Any]] = None,
    actions_log_text: Optional[str] = None,
) -> str:
    """Formats a running table of tried coordinates and their outcomes.

    Supports multiple trajectory schemas robustly (ActionSignature, dicts, actions log text).
    """
    clicks: List[str] = []

    # 1. Parse from structured trajectory steps
    if trajectory_or_steps:
        for step in trajectory_or_steps:
            sig = getattr(step, "action_sig", None)
            if sig is None and isinstance(step, dict):
                sig = step.get("action_sig") or step

            if not sig:
                continue

            name = getattr(sig, "name", str(sig)).upper()
            if "ACTION6" in name or "CLICK" in name or "MOUSE" in name:
                raw_data = getattr(sig, "data", None)
                data_dict: Dict[str, Any] = {}
                if isinstance(raw_data, dict):
                    data_dict = raw_data
                elif isinstance(raw_data, (list, tuple)):
                    try:
                        data_dict = dict(raw_data)
                    except Exception:
                        pass
                elif isinstance(sig, dict):
                    data_dict = sig.get("data", sig)

                x = data_dict.get("x")
                y = data_dict.get("y")
                if x is None or y is None:
                    # Try regex extraction from string representation of signature
                    m = re.search(r"x\s*[:=]\s*(\d+)\D+y\s*[:=]\s*(\d+)", str(sig), re.IGNORECASE)
                    if m:
                        x, y = int(m.group(1)), int(m.group(2))

                if x is not None and y is not None:
                    changed = getattr(step, "changed", None)
                    if isinstance(step, dict) and changed is None:
                        changed = step.get("changed")

                    if changed is True:
                        outcome = "CHANGED (Grid state shifted)"
                    elif changed is False:
                        outcome = "NO-OP (No visual change / Ineffective click)"
                    else:
                        outcome = "EXECUTED"
                    clicks.append(f"- Clicked (X={x}, Y={y}) -> {outcome}")

    # 2. Fallback / Supplement from markdown actions log if trajectory had no parsed clicks
    if not clicks and actions_log_text:
        for line in actions_log_text.splitlines():
            if "ACTION6" in line or "CLICK" in line:
                m = re.search(r"\bX\s*[:=]\s*(\d+)\D+Y\s*[:=]\s*(\d+)", line, re.IGNORECASE)
                if not m:
                    m = re.search(r"\(x\s*[:=]\s*(\d+)\s*,\s*y\s*[:=]\s*(\d+)\)", line, re.IGNORECASE)
                if m:
                    x, y = int(m.group(1)), int(m.group(2))
                    outcome = "CHANGED" if "->" in line and not line.endswith("->  |") else "EXECUTED"
                    clicks.append(f"- Clicked (X={x}, Y={y}) -> {outcome}")

    if not clicks:
        return "No coordinates clicked yet in this attempt."
    return "\n".join(clicks[-15:])
