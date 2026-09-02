"""Live terminal and IPython output rendering for agent steps."""

from pathlib import Path
from typing import Any, Optional
import base64
import os

_live_display_id = "arc_live_screen"
_live_initialized = False


def render_live(
    state: Optional[Any] = None,
    status: str = "",
    label: str = "",
    reasoning: str = "",
    header_extra: str = "",
) -> None:
    """Renders agent step live to IPython or logs to stdout if running headless."""
    global _live_initialized

    in_ipython = False
    try:
        from IPython.display import HTML, display, update_display

        # Check if get_ipython is defined
        get_ipython()  # type: ignore # noqa: F821
        in_ipython = True
    except Exception:
        in_ipython = False

    if in_ipython:
        img_tag = ""
        if state is not None and getattr(state, "grid", None) is not None:
            img_path = state.get_image_path()
            if img_path and os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    b64_img = base64.b64encode(f.read()).decode("utf-8")
                img_tag = (
                    f'<img src="data:image/png;base64,{b64_img}" '
                    f'style="max-width:480px; display:block; margin: 8px 0; border-radius:4px;">'
                )

        header = f"{status}\n{header_extra}".strip()
        reasoning_html = (reasoning[:500] + "…") if len(reasoning) > 500 else reasoning
        reasoning_html = reasoning_html.replace("\n", "<br>") if reasoning_html else ""

        payload = HTML(
            f"""
            <div style="background:#111; color:#eee; padding:10px; border-radius:6px; font-family:monospace;">
                <pre style="margin:0; font-size:12px; line-height:1.4;">{header}</pre>
                {img_tag}
                <pre style="margin:4px 0 0 0; font-size:11px; white-space:pre-wrap; max-height:120px; overflow-y:auto; background:#1c1c1c; padding:6px; border-radius:4px; border:1px solid #333;">{reasoning_html}</pre>
            </div>
            """
        )

        if not _live_initialized:
            display(payload, display_id=_live_display_id)
            _live_initialized = True
        else:
            update_display(payload, display_id=_live_display_id)
    else:
        # CLI fallback
        out = f"[{status}]"
        if reasoning:
            out += f" -> {reasoning.splitlines()[0]}"
        print(out)


def live_summary(game_id: str, level: int, step: int, max_steps: int, action_name: str, changed: Optional[bool]) -> str:
    change_tag = "CHANGED" if changed else "NOOP"
    return f"🎮 {game_id} | Lvl {level} | Step {step}/{max_steps} | Action: {action_name} ({change_tag})"
