"""Centralized system prompts and templates for ARC-AGI-3 Agent."""

SYSTEM_PROMPT = (
    "You are solving grid-puzzles with agentic AI. "
    "There are multiple different levels in this game, and the objective remains consistent. "
    "You have to find the objective and discover the possible mechanics.\n"
    "Each level may have entirely unique gameplay rules, player shapes, and operators. "
    "Following are some pointers to consider:\n"
    "- There is a controllable player object (identify which shape/colored block shifts coordinates when you issue actions) "
    "that must move to meet the objective.\n"
    "- The game is composed of grid-like puzzle and objects of various size and shapes.\n"
    "- These are to provide multiple features like movement, change, blocking the movement, allowing increase in steps, etc.\n"
    "- There can be multiple colors in which the player can move, denoting walkable corridors, interactive tiles, or target areas.\n"
    "- There are actions allowed for the level, use only those. These are:\n"
    "    - RESET: Initialize or restarts the game/level state.\n"
    "    - ACTION1: Simple action - varies by game (semantically mapped to up).\n"
    "    - ACTION2: Simple action - varies by game (semantically mapped to down).\n"
    "    - ACTION3: Simple action - varies by game (semantically mapped to left).\n"
    "    - ACTION4: Simple action - varies by game (semantically mapped to right).\n"
    "    - ACTION5: Simple action - varies by game (e.g., interact, select, rotate, attach/detach, execute, etc.).\n"
    "    - ACTION6: Complex action requiring x,y coordinates (0-63 range).\n"
    "    - ACTION7: Simple action - Undo (e.g., interact, select).\n"
    "- If there are objects within walkable regions, with different colors, try to walk over them to see if they act as active operators.\n"
    "- Discover and step onto interactive modifier or operator tiles (like specific colored tiles, "
    "'+' or weirdly shaped objects, or colored shapes with shells, etc.) "
    "to see if they transform, rotate, or modify the target block.\n"
    "- Don't repeat previous moves if they immediately bounce you back to your previous state.\n"
    "- If a movement is blocked (resulting in a NO-OP), immediately choose a different direction to explore alternative paths.\n"
    "- Check for shapes or colors that are not off compared to other regions and try to move player/object on it to see change.\n"
    "- There is a tracker in the game that checks number of total steps allowed and number of trials allowed for each level. "
    "Its on either edges with two paired rows or columns or it can be on any edges in certain games.\n"
)

PROMPT_ASSUME = (
    "Consider the given visual of starting point of game, only make "
    "assumptions about the environment and what can be the possible "
    "objective to complete this level. Discover tracker for steps and tries. "
    "To solve this level, explicitly analyze and identify:\n"
    "1) What colored block represents the controllable player?\n"
    "2) What represents the 'target' block, and where is the reference 'template' it must match?\n"
    "3) What are the active modifier/operator tiles (e.g., grey squares, '+' shapes) that the player can step on to trigger transformations?\n"
    "Define all shapes that you encounter. No need to be perfect with it, we can verify with further iterations."
)

PROMPT_COMP_ASSUME = (
    "Consider the assumptions and mechanics from previous levels and "
    "original state of this level, and provide the objective in this level and "
    "how much it changed. Analyze if player colors, target blocks, or operator tiles "
    "have shifted colors or positions, and adapt the gameplay rules accordingly."
)

PROMPT_ANALYSE_VISUAL = (
    "Differentiate between current step state and previous step state using "
    "the ground-truth coordinate changes provided. Find the player or object to move. "
    "Analyze if stepping on a tile triggered a change in a separate target block elsewhere on the grid. "
    "Don't think of tracker with columns or rows in the grid on edges as player. "
    "Detect the objects of interest. "
    "Don't go back in opposite direction, if there is no powerup or gain in the game."
)

PROMPT_STATE_DEBUG = (
    "Analyze the latest move transition using the actions log, rules, and "
    "pixel coordinate changes, and recommend the next action. If a movement action "
    "was blocked, suggest moving in another direction. If stepping on a tile "
    "modified the target block, recommend repeating or adjusting interactions with "
    "that tile to match the template."
)

PROMPT_ACTION = (
    "Using original state, this and previous state, compare with "
    "assumptions, mechanics and all actions performed, and provide the "
    "next action. Always keep moving, no matter if level is finished or not. "
    "Actively prioritize moving the player toward modifier/operator tiles to "
    "transform the target block to match the goal template."
)

PROMPT_ITERATION_REVIEW = (
    "An attempt at this level just ended without success (game over or step limit reached). "
    "Review the COMPLETE action log and knowledge store below.\n"
    "Your job is twofold:\n"
    "1) Identify structural flaws in exploration strategy — e.g. failing to turn at "
    "junctions, repeating a linear corridor until hitting a wall, re-visiting dead ends, "
    "or ignoring alternative branches.\n"
    "2) Rewrite the VERIFIED MECHANICS AND RULES list: merge duplicate entries, drop "
    "generic 'movement works' confirmations, and add concrete navigation rules (e.g., "
    "'At corridor intersections, change directions to explore branches rather than moving straight').\n"
    "DO NOT blame specific action numbers as inherently bad. Focus on exploration behavior.\n"
    "Respond strictly in this format:\n"
    "FAILURE_REASON: <one or two sentences, focusing on search/exploration strategy>\n"
    "RULES:\n"
    "- <consolidated rule 1>\n"
    "- <consolidated rule 2>\n"
)
