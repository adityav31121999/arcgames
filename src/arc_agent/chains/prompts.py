"""Centralized system prompts and templates for ARC-AGI-3 Agent."""

from typing import Any, Iterable, Optional
from ..core.actions import canonical_action_name

ACTION_DESCRIPTIONS = {
    "RESET": "Initialize or restarts the game/level state.",
    "ACTION1": "Upward movement / direction.",
    "ACTION2": "Downward movement / direction.",
    "ACTION3": "Leftward movement / direction.",
    "ACTION4": "Rightward movement / direction.",
    "ACTION5": "Interact / select / rotate / execute action.",
    "ACTION6": "Click at coordinate X,Y (0-63 range).",
    "ACTION7": "Undo action.",
}

SYSTEM_PROMPT_TEMPLATE = (
    "You are an expert AI agent solving 2D ARC-AGI-3 grid reasoning puzzles.\n"
    "Respond in clear, concise English only.\n"
    "Available actions for this game:\n"
    "{actions_block}\n\n"
    "Core Guidelines:\n"
    "- Only a click action accepts X/Y coordinates. Movement and interaction buttons are global, not clicks on objects.\n"
    "- Separate the action API from inferred game effects. Do not assume a button moves only one object.\n"
    "- Maintain competing hypotheses, cite observations that support or contradict them, and choose a test that distinguishes them.\n"
    "- Identify the interactive puzzle elements and rule mechanics through structured actions.\n"
    "- Border glyphs may be HUD trackers or interactive elements; treat their role as unknown until tested.\n"
    "- A NO-OP means no detected visible change; it does not prove a wall or invalid action. "
    "Test alternatives and prerequisites.\n"
    "- Check if there is any piece that needs to matched or not, whether reach a goal post, and other possible objectives "
    "like moving from one point to another, placing object over something, etc.\n"
    "- Maintain working memory using labeled prefixes: 'World model:', 'Goal model:', 'Action model:', 'Recent findings:', 'Plan:'.\n"
    "- Avoid repeating an unchanged experiment; retry a NO-OP when state or prerequisites change.\n"
)

SYSTEM_PROMPT_CLICK_ONLY = (
    "You are an expert AI agent solving 2D ARC-AGI-3 click puzzles.\n"
    "Respond in clear, concise English only.\n"
    "The ONLY valid action in this game is ACTION6 (Click at coordinate X,Y).\n\n"
    "Core Guidelines:\n"
    "- Identify distinct foreground shapes, symbols, icons, or clusters.\n"
    "- Border and corner markers may be HUD or interactive; use observed effects to determine their role.\n"
    "- Prefer clicking unclicked objects at their solid center to discover mechanics.\n"
    "- Maintain working memory using labeled prefixes: 'World model:', 'Goal model:', 'Action model:', 'Recent findings:', 'Plan:'."
)


def build_system_prompt(action_space: Optional[Iterable[Any]] = None) -> str:
    """Builds a dynamic system prompt templated with only the actions available in this game.

    Args:
        action_space: List/iterable of GameAction enums or string names actually available.
    """
    if not action_space:
        lines = [f"    - {name}: {desc}" for name, desc in ACTION_DESCRIPTIONS.items()]
        actions_block = "\n".join(lines)
        return SYSTEM_PROMPT_TEMPLATE.format(actions_block=actions_block)

    names = [
        getattr(a, "name", str(a)).upper()
        for a in action_space
        if getattr(a, "name", str(a)).upper() not in ("RESET", "0", "GAMEACTION.RESET")
    ]
    # Check if click-only
    if names and all(n in ("ACTION6", "ACTION_6", "6", "CLICK", "MOUSE") for n in names):
        return SYSTEM_PROMPT_CLICK_ONLY

    lines = []
    for a in action_space:
        name = canonical_action_name(a)
        uname = name.upper()
        if uname in ACTION_DESCRIPTIONS:
            lines.append(f"    - {name}: {ACTION_DESCRIPTIONS[uname]}")
        elif name in ACTION_DESCRIPTIONS:
            lines.append(f"    - {name}: {ACTION_DESCRIPTIONS[name]}")
        else:
            lines.append(f"    - {name}: Action supported by environment.")
    actions_block = "\n".join(lines)
    return SYSTEM_PROMPT_TEMPLATE.format(actions_block=actions_block)


SYSTEM_PROMPT = build_system_prompt()

PROMPT_ASSUME = (
    "Analyze the initial visual layout and metadata to form an initial hypothesis in English.\n"
    "Identify key visual elements. Label proposed object roles and goals as hypotheses, not verified facts.\n"
    "Format your response strictly using these labeled sections (1-2 concise sentences each):\n"
    "World model: <layout, grid dimensions, background vs foreground shapes>\n"
    "Goal model: <perceived puzzle objective and win condition>\n"
    "Action model: <expected effect of allowable actions on these objects>\n"
    "Recent findings: <initial visual state observations>\n"
    "Plan: <concrete first action hypothesis to test mechanics>"
)

PROMPT_COMP_ASSUME = (
    "Compare this level's initial layout with previous level findings to identify rule changes in English.\n"
    "Format your response strictly using these labeled sections:\n"
    "World model: <what visual elements carried over or shifted>\n"
    "Goal model: <updated objective for this level>\n"
    "Action model: <refined understanding of action mechanics>\n"
    "Recent findings: <new shapes or color variations observed in this level>\n"
    "Plan: <updated plan for solving this level>"
)

PROMPT_ANALYSE_VISUAL = (
    "Analyze the visual change caused by the last action using the pixel diff bounding box in English.\n"
    "Use these labeled fields: Recent findings: <observed changes only>; Open questions: <uncertain cause or goal progress>. Put each field on its own line. A changed board alone does not prove progress."
)

PROMPT_ACTION = (
    "Select the next action to advance toward the objective based on current state and verified rules.\n"
    "Avoid repeating an unchanged experiment; retry a NO-OP when state or prerequisites change."
)

PROMPT_CLICK_ONLY_TARGET = (
    "Select the best target coordinates for ACTION6.\n"
    "Prefer unexplored foreground objects at a solid pixel. Do not exclude border targets solely by location; revisit targets if the state or prerequisites changed.\n"
    "Objects detected this frame:\n{object_list}\n"
    "Coordinates already tried:\n{click_history}\n"
)

PROMPT_ITERATION_REVIEW = (
    "An attempt at this level just ended. Review the actions log and knowledge store in English.\n"
    "1) Identify the exploration flaw (e.g. loops, dead ends, unclicked objects).\n"
    "2) Propose navigation hypotheses with supporting action/step references and counterexamples; do not claim verification from your own verdict.\n"
    "Respond strictly in this format:\n"
    "FAILURE_REASON: <one sentence on search strategy>\n"
    "RULES:\n"
    "- <candidate rule and supporting steps>\n"
    "- <candidate rule and supporting steps>\n"
)
