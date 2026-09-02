"""Persistent markdown and in-memory knowledge store."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple


def scratchpad_path(game_id: str, memory_root: str | Path = "./agent_memory") -> Path:
    return Path(memory_root) / str(game_id) / "scratchpad.md"


def ostate_path(game_id: str, memory_root: str | Path = "./agent_memory") -> Path:
    return Path(memory_root) / str(game_id) / "ostate.md"


def actions_log_path(game_id: str, level: int, memory_root: str | Path = "./agent_memory") -> Path:
    d = Path(memory_root) / str(game_id) / f"level_{level}"
    d.mkdir(parents=True, exist_ok=True)
    return d / "actions.md"


def read_text(path: Path, default: str = "") -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else default


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")


def tail_text(text: str, max_chars: int = 2500) -> str:
    return text[-max_chars:] if len(text) > max_chars else text


def update_verified_mechanics(game_id: str, rule: str, memory_root: str | Path = "./agent_memory") -> None:
    s_path = scratchpad_path(game_id, memory_root)
    if not s_path.exists():
        return
    content = s_path.read_text(encoding="utf-8", errors="ignore")
    target_header = "## VERIFIED MECHANICS AND RULES"
    if target_header in content:
        parts = content.split(target_header)
        updated_content = parts[0] + target_header + "\n" + parts[1].strip() + f"\n- {rule}\n"
        s_path.write_text(updated_content, encoding="utf-8")
    else:
        append_text(s_path, f"\n## VERIFIED MECHANICS AND RULES\n- {rule}")



class KnowledgeCache:
    """In-memory cache synchronized with disk markdown files for low latency."""

    def __init__(self, memory_root: str | Path = "./agent_memory"):
        self.memory_root = Path(memory_root)
        self._scratch: Dict[str, str] = {}
        self._actions: Dict[Tuple[str, int], str] = {}
        self._ostate: Dict[str, str] = {}

    def scratch(self, game_id: str, max_chars: int = 400) -> str:
        if game_id not in self._scratch:
            self._scratch[game_id] = read_text(scratchpad_path(game_id, self.memory_root))
        return tail_text(self._scratch[game_id], max_chars)

    def append_scratch(self, game_id: str, text: str) -> None:
        self._scratch[game_id] = self._scratch.get(game_id, "") + text
        append_text(scratchpad_path(game_id, self.memory_root), text)

    def write_scratch(self, game_id: str, text: str) -> None:
        self._scratch[game_id] = text
        p = scratchpad_path(game_id, self.memory_root)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def ostate(self, game_id: str, max_chars: int = 1500) -> str:
        if game_id not in self._ostate:
            self._ostate[game_id] = read_text(ostate_path(game_id, self.memory_root))
        return tail_text(self._ostate[game_id], max_chars)

    def append_ostate(self, game_id: str, text: str) -> None:
        self._ostate[game_id] = self._ostate.get(game_id, "") + text
        append_text(ostate_path(game_id, self.memory_root), text)

    def actions_log(self, game_id: str, level: int, max_chars: int = 400) -> str:
        key = (game_id, level)
        if key not in self._actions:
            self._actions[key] = read_text(actions_log_path(game_id, level, self.memory_root))
        return tail_text(self._actions[key], max_chars)

    def append_action_log(self, game_id: str, level: int, text: str) -> None:
        key = (game_id, level)
        self._actions[key] = self._actions.get(key, "") + text
        p = actions_log_path(game_id, level, self.memory_root)
        with open(p, "a", encoding="utf-8") as f:
            f.write(text)

    def refresh_level(self, game_id: str, level: int) -> None:
        self._scratch[game_id] = read_text(scratchpad_path(game_id, self.memory_root))
        self._ostate[game_id] = read_text(ostate_path(game_id, self.memory_root))
        self._actions[(game_id, level)] = read_text(actions_log_path(game_id, level, self.memory_root))


def init_knowledge_files(
    game_id: str,
    level: int,
    valid_actions: Optional[list] = None,
    memory_root: str | Path = "./agent_memory",
) -> None:
    s_path = scratchpad_path(game_id, memory_root)
    if not s_path.exists():
        s_path.parent.mkdir(parents=True, exist_ok=True)
        s_path.write_text(
            f"# Scratchpad — Game: {game_id}\n\n"
            "## OBJECTIVE\nTo be inferred from S0\n\n"
            "## HYPOTHESES & ASSUMPTIONS\n- Observing initial level layout\n\n"
            "## VERIFIED MECHANICS AND RULES\n- Confirmed rules from completed levels carry over here\n",
            encoding="utf-8",
        )

    o_path = ostate_path(game_id, memory_root)
    if not o_path.exists():
        o_path.parent.mkdir(parents=True, exist_ok=True)
        o_path.write_text(f"# Cross-Level S0 Analysis ({game_id})\n\n", encoding="utf-8")

    a_path = actions_log_path(game_id, level, memory_root)
    if not a_path.exists():
        action_names = [getattr(a, "name", str(a)) for a in (valid_actions or [])]
        a_path.write_text(
            f"# Actions Log — Game: {game_id}, Level: {level}\n\n"
            f"## ALLOWABLE ACTIONS FOR THIS GAME\n"
            f"{', '.join(action_names) if action_names else 'Not specified'}\n\n"
            "| Step | Time | Action | Hash Shift |\n"
            "|------|------|--------|------------|\n",
            encoding="utf-8",
        )


def maybe_append_rule(
    game_id: str,
    debugger_verdict: str,
    is_repeat: bool,
    changed: Optional[bool],
    cache: KnowledgeCache,
) -> None:
    """Parses debugger verdict and appends to verified rules or debunked assumptions."""
    if not debugger_verdict or is_repeat:
        return

    verdict_lower = debugger_verdict.lower()
    clean_line = debugger_verdict.strip().split("\n")[0][:150]

    if verdict_lower.startswith("diverged") or "diverg" in verdict_lower:
        _write_scratch_section(cache, game_id, "## DEBUNKED / INVALID ASSUMPTIONS", clean_line)
    elif changed is False or "blocked" in verdict_lower or "wall" in verdict_lower:
        _write_scratch_section(cache, game_id, "## VERIFIED MECHANICS AND RULES", clean_line)
    elif verdict_lower.startswith("expected"):
        _write_scratch_section(cache, game_id, "## VERIFIED MECHANICS AND RULES", clean_line)


def _write_scratch_section(cache: KnowledgeCache, game_id: str, header: str, entry_text: str) -> None:
    entry = f"- {entry_text}\n"
    content = cache.scratch(game_id, max_chars=999999)
    if header in content:
        parts = content.split(header)
        if entry not in parts[1]:
            updated = parts[0] + header + "\n" + entry + parts[1].strip() + "\n"
            cache.write_scratch(game_id, updated)
    else:
        cache.append_scratch(game_id, f"\n{header}\n{entry}")


def apply_iteration_review(
    cache: KnowledgeCache,
    game_id: str,
    level: int,
    iteration: int,
    review_text: str,
) -> None:
    """Updates verified mechanics and failed notes from post-iteration review."""
    if not review_text or "[REVIEW INFERENCE FAILED" in review_text:
        return

    failure_reason = ""
    rules: List[str] = []
    in_rules = False

    for line in review_text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("FAILURE_REASON:"):
            failure_reason = stripped.split(":", 1)[1].strip()
            in_rules = False
        elif stripped.upper().startswith("RULES:"):
            in_rules = True
        elif in_rules and (stripped.startswith("-") or stripped.startswith("*")):
            cleaned = stripped[1:].strip()
            if cleaned:
                rules.append(cleaned)

    rules_header = "## VERIFIED MECHANICS AND RULES"
    if rules:
        content = cache.scratch(game_id, max_chars=999999)
        rule_block = "\n".join(f"- {r}" for r in rules)
        if rules_header in content:
            before, _, after_header = content.partition(rules_header)
            next_idx = after_header.find("\n## ")
            trailing = after_header[next_idx:] if next_idx != -1 else ""
            new_content = before + rules_header + "\n" + rule_block + "\n" + trailing
        else:
            new_content = content.rstrip() + f"\n\n{rules_header}\n{rule_block}\n"
        cache.write_scratch(game_id, new_content)

    if failure_reason:
        notes_header = "## FAILED ITERATION NOTES"
        entry = f"- (Level {level}, Iter {iteration}) {failure_reason}\n"
        content = cache.scratch(game_id, max_chars=999999)
        if notes_header in content:
            before, _, after_header = content.partition(notes_header)
            next_idx = after_header.find("\n## ")
            trailing = after_header[next_idx:] if next_idx != -1 else ""
            body = after_header[:next_idx] if next_idx != -1 else after_header
            new_content = before + notes_header + "\n" + body.rstrip() + "\n" + entry + trailing
        else:
            new_content = content.rstrip() + f"\n\n{notes_header}\n{entry}"
        cache.write_scratch(game_id, new_content)
