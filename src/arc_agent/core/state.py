"""ARC State and Transition data structures with lazy rendering and metadata hashing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
import json
import numpy as np
from PIL import Image

from .color_palette import render_grid_to_pil, render_grid_to_png
from .diff import detect_real_change, extract_grid_array, get_gameplay_grid


def make_safe_serializable(obj: Any) -> Any:
    """Recursively parses objects to construct completely serializable structures."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {str(k): make_safe_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_safe_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "name"):
        return obj.name
    if hasattr(obj, "value"):
        return obj.value
    return str(obj)


def compute_state_hash(
    grid: Optional[np.ndarray],
    game_state: Any = None,
    levels_completed: Optional[int] = None,
    include_semantic: bool = True,
) -> str:
    """Full SHA-256 state hashing strategy, ignoring tracker borders."""
    if grid is None or grid.size == 0:
        base = "EMPTY_STATE"
    else:
        gameplay_grid = get_gameplay_grid(grid)
        if gameplay_grid is None or gameplay_grid.size == 0:
            base = "EMPTY_STATE"
        else:
            h = hashlib.sha256()
            h.update(str(gameplay_grid.shape).encode())
            h.update(str(gameplay_grid.dtype).encode())
            h.update(gameplay_grid.tobytes())
            base = h.hexdigest()

    if not include_semantic:
        return base

    semantic = hashlib.sha256()
    semantic.update(base.encode())
    semantic.update(str(game_state).encode())
    semantic.update(str(levels_completed).encode())
    return semantic.hexdigest()


@dataclass
class ARCState:
    raw_obs: Any
    grid: Optional[np.ndarray]
    text_repr: str
    state_hash: str
    game_id: str
    level: int
    step: int
    tag: str
    levels_completed: int
    game_state: Any
    _rendered_path: Optional[str] = None
    _pil_image: Optional[Image.Image] = None
    _json_data: Optional[Dict[str, Any]] = None

    @classmethod
    def create(
        cls,
        game_id: str,
        level: int,
        step: int,
        obs: Any,
        tag: str = "state",
        semantic_hash: bool = True,
    ) -> "ARCState":
        grid_arr = extract_grid_array(obs)
        g_state = getattr(obs, "state", None)
        levels_completed = getattr(obs, "levels_completed", 0)

        if grid_arr is not None:
            rows = [" ".join(f"{val:1d}" for val in row) for row in grid_arr]
            text_repr = f"Shape: {grid_arr.shape}\n" + "\n".join(rows)
            state_hash = compute_state_hash(
                grid_arr, g_state, levels_completed, include_semantic=semantic_hash
            )
        else:
            text_repr = "(No active grid matrix)"
            state_hash = "EMPTY_STATE"

        return cls(
            raw_obs=obs,
            grid=grid_arr,
            text_repr=text_repr,
            state_hash=state_hash,
            game_id=game_id,
            level=level,
            step=step,
            tag=tag,
            levels_completed=levels_completed,
            game_state=g_state,
        )

    @property
    def json_data(self) -> Dict[str, Any]:
        """Parses raw observation metadata safely into a dictionary."""
        if self._json_data is None:
            data = {}
            if self.raw_obs is not None:
                if hasattr(self.raw_obs, "model_dump"):
                    try:
                        data = self.raw_obs.model_dump()
                    except Exception:
                        pass
                elif hasattr(self.raw_obs, "model_dump_json"):
                    try:
                        data = json.loads(self.raw_obs.model_dump_json())
                    except Exception:
                        pass
                elif hasattr(self.raw_obs, "__dict__"):
                    try:
                        data = dict(self.raw_obs.__dict__)
                    except Exception:
                        pass

                if not data:
                    try:
                        data = json.loads(json.dumps(self.raw_obs, default=lambda o: o.__dict__))
                    except Exception:
                        pass

            safe_data = {}
            for k, v in data.items():
                safe_data[k] = make_safe_serializable(v)

            if self.grid is not None:
                safe_data["frame"] = self.grid.tolist()
            else:
                safe_data["frame"] = None

            self._json_data = safe_data
        return self._json_data

    @property
    def proper_json_repr(self) -> str:
        """Generates the full JSON representation including the 2D grid matrix."""
        return json.dumps(self.json_data, indent=2)

    @property
    def compact_json_repr(self) -> str:
        """Generates visual-safe JSON representation, summarizing matrix array."""
        data_copy = {}
        for k, v in self.json_data.items():
            if k in ("grid", "frame"):
                if isinstance(v, list):
                    h = len(v)
                    w = len(v[0]) if h > 0 else 0
                    data_copy[k] = f"[{h}x{w} Grid List]"
                else:
                    data_copy[k] = str(v)
            else:
                data_copy[k] = v
        return json.dumps(data_copy, indent=2)

    def get_image_path(self, cache_dir: str = "/tmp/agent_vision") -> Optional[str]:
        """Lazy-renders visual PNG file only when requested."""
        if self.grid is None:
            return None
        if self._rendered_path is None or not Path(self._rendered_path).exists():
            path = Path(cache_dir) / f"{self.game_id}_lvl_{self.level}_{self.tag}_{self.step}.png"
            render_grid_to_png(self.grid, save_path=path)
            self._rendered_path = str(path)
        return self._rendered_path

    def get_pil_image(self, scale: int = 16) -> Optional[Image.Image]:
        """Fast in-memory PIL RGB Image for direct Transformers processing."""
        if self.grid is None:
            return None
        if self._pil_image is None:
            self._pil_image = render_grid_to_pil(self.grid, scale=scale)
        return self._pil_image


@dataclass
class ARCTransition:
    previous: Optional[ARCState]
    current: ARCState
    action_sig: Optional[Any]
    changed: Optional[bool]
    is_noop: bool


def compute_transition(
    previous_state: Optional[ARCState],
    current_state: ARCState,
    action_sig: Optional[Any] = None,
) -> ARCTransition:
    if previous_state is None or previous_state.grid is None or current_state.grid is None:
        changed, is_noop = None, False
    else:
        try:
            changed = detect_real_change(previous_state.grid, current_state.grid)
        except Exception:
            changed = previous_state.state_hash != current_state.state_hash
        is_noop = not changed

    return ARCTransition(
        previous=previous_state,
        current=current_state,
        action_sig=action_sig,
        changed=changed,
        is_noop=is_noop,
    )


def save_step_state_json(
    game_id: str, level: int, step_index: int, state: ARCState, memory_root: str | Path = "./agent_memory"
) -> Path:
    """Saves raw JSON observation metadata of a specific step to disk."""
    level_dir = Path(memory_root) / str(game_id) / f"level_{level}"
    json_path = level_dir / f"s{step_index}_metadata.json"
    try:
        level_dir.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(state.json_data, f, indent=2)
    except OSError:
        pass
    return json_path
