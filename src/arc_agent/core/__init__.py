"""Core ARC domain structures, state representations, and spatial utilities."""

from .color_palette import ARC_COLOR_PALETTE, ARC_CMAP, ARC_NORM, render_grid_to_png, render_grid_to_pil
from .diff import get_gameplay_grid, detect_real_change, get_grid_difference_text, extract_grid_array
from .state import ARCState, ARCTransition, compute_state_hash, compute_transition
from .actions import ActionSignature, ARCActionMapper, is_complex_action, validate_coordinates
from .resolver import GameStateResolver
from .object_detection import (
    detect_grid_objects,
    is_click_only,
    render_click_history,
    render_detected_objects,
)

__all__ = [
    "ARC_COLOR_PALETTE",
    "ARC_CMAP",
    "ARC_NORM",
    "render_grid_to_png",
    "render_grid_to_pil",
    "get_gameplay_grid",
    "detect_real_change",
    "get_grid_difference_text",
    "extract_grid_array",
    "ARCState",
    "ARCTransition",
    "compute_state_hash",
    "compute_transition",
    "ActionSignature",
    "ARCActionMapper",
    "is_complex_action",
    "validate_coordinates",
    "GameStateResolver",
    "detect_grid_objects",
    "render_detected_objects",
    "render_click_history",
    "is_click_only",
]
