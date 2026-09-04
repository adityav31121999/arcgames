Let me explore both codebases thoroughly before creating a comparison plan.

Listed directory arcgame
Listed directory duck-qwen38-nvfp4-mtp-vllm-smoke-v1
Listed directory src
Listed directory src
Listed directory configs
Listed directory ARC3-Inference
Listed directory tufa-arc-agi-framework
Listed directory arc_agent
Listed directory inference
Listed directory src
Listed directory agent
Listed directory agent
Listed directory framework
Listed directory tools
Listed directory core
Listed directory chains
Listed directory models
Listed directory utils
Listed directory memory

Now let me read the key files from both codebases in parallel:

Viewed run.py:1-800
Viewed runner.py:1-389
Viewed solver.py:1-800
Viewed arc_langchain_agent.py:1-256
Viewed tool_agent.py:1-200
Viewed config.py:1-86
Viewed default.yaml:1-35
Viewed tool_agent.py:200-600
Viewed brain.py:1-105
Viewed tool_agent.py:600-1100
Viewed eye.py:1-99
Viewed tool_agent.py:1100-1500
Viewed trajectory.py:1-239
Viewed prompts.py:1-115
Viewed prompts.py:1-94

Now let me look at a few more files for concurrency and model differences:

Viewed solver.py:800-1100
Viewed factory.py:1-100

Now I have a thorough understanding of both codebases. Let me write the comprehensive comparison artifact:

Created comparison_report.md

Here's the full breakdown. I read every key file in both agents. The comparison report above is the full artifact — here's the executive summary:

---

## 🏆 Key Differences (Why Duck Wins)

### The 3 Core Gaps

| Gap                  | Duck                                                               | Your Agent                                              |
| -------------------- | ------------------------------------------------------------------ | ------------------------------------------------------- |
| **Action reasoning** | Python REPL — model writes BFS/search code, batches actions        | 32-token text parse — one action per expensive LLM call |
| **Throughput**       | vLLM server + `concurrency=16` async games                         | Sequential HF Transformers, no batching                 |
| **World model**      | Structured `{world_model, goal, plan}` dict re-injected every turn | Flat file-based scratch cache, no structured extraction |

---

## ⚡ Fastest Wins (Safe, No Breaking Changes)

These 4 changes require **only YAML + 1 prompt file edits** and will give the biggest return:

**1. [`default.yaml`](file:///e:/code-in-progress/arcgame/configs/default.yaml) + [`brain.py`](file:///e:/code-in-progress/arcgame/src/arc_agent/chains/brain.py)**

```yaml
max_new_tokens_brain: 32  →  128 # model literally can't reason in 32 tokens
temperature: 0.0          →  0.4 # (in brain.py _invoke call) escape stuck states
max_context_length: 8192  →  32768 # Duck uses full 32K
```

**2. [`chains/prompts.py`](file:///e:/code-in-progress/arcgame/src/arc_agent/chains/prompts.py)**  
Add the HUD-timer warning Duck uses — your agent currently wastes steps clicking the step-counter bar because the prompt tells it "there's always a player" but doesn't warn about HUD misidentification.

These 4 changes alone should close ~60-70% of the quality gap with Duck.

Want me to apply any of these fixes directly? I can start with the YAML + prompts edits immediately.

# ARC-AGI-3: Architectural Comparison & Improvement Guide

> **Competition topper**: `duck-qwen38-nvfp4-mtp-vllm-smoke-v1` — Score: **7.51**  
> **Your agent**: `arcgame/src/arc_agent` — LangChain + Transformers-based, multi-chain

---

## 1. High-Level Architecture Difference

| Dimension            | Your Agent                                       | Duck/Competition Topper                          |
| -------------------- | ------------------------------------------------ | ------------------------------------------------ |
| **Inference Engine** | HuggingFace Transformers (local `.generate()`)   | vLLM server (`/v1/chat/completions` REST)        |
| **Model**            | `nvidia/Gemma-4-26B-A4B-NVFP4` (Gemma family)    | `Qwen3-8B NVfp4 + MTP` (Qwen family)             |
| **Action Interface** | Chain-per-role: Eye, Brain, Debugger, Reviewer   | Single ToolAgent with Python tool sandbox        |
| **Action format**    | `ACTION=<NAME> [X=<int> Y=<int>]` text parse     | Tool call: `action(['LEFT'])` in Python code     |
| **Concurrency**      | Sequential (one game at a time)                  | `asyncio` + `ThreadPoolExecutor(concurrency=16)` |
| **Reasoning style**  | 4-role multi-chain with separate LLM calls       | Unified Python REPL with embedded thinking       |
| **State format**     | Text matrix + JSON repr + PIL image              | ASCII + segmentation graph (object-based)        |
| **Retry strategy**   | Up to 3 iterations per level with Reviewer chain | Retryable LLM failures with backoff              |
| **Memory**           | File-based KnowledgeCache + TrajectoryMemory     | In-session message history + world-model dict    |
| **Planning**         | Speculative one-shot plan → closed loop          | Python BFS/search inside tool calls              |

---

## 2. Key Differences: In-Depth

### 2A. The Core Engine: vLLM vs. Transformers

**Duck (winner) uses a vLLM server:**

```python
# tool_agent.py — pure HTTP REST
response = requests.post(
    f"{self._model.base_url.rstrip('/')}/chat/completions",
    headers=self._headers(),
    json=request_payload,
    timeout=request_timeout_seconds,
)
```

- vLLM **batches multiple game sessions** (concurrency=16) onto the same GPU
- MTP (Multi-Token Prediction) speeds up generation ~40-60%
- NVfp4 quantization fits a larger model in same VRAM with near-fp16 quality

**Your agent uses HuggingFace Transformers:**

```python
# gemma_transformers.py
response = self.model.invoke(messages, **invoke_kwargs)
```

- Each LLM call is **synchronous and exclusive** — no batching
- 4 separate chains (Eye, Brain, Debugger, Reviewer) each call `.invoke()` independently
- `max_new_tokens_brain: 32` — extremely low token budget for the most critical decisions

---

### 2B. The Action Paradigm: Text Parsing vs. Python Tool Sandbox

**Duck: Python REPL inside LLM**

```python
# The model writes Python code that BFS-searches, then calls action()
code = """
segs = current_frame.segmentation['nodes']
player = [n for n in segs if n['color'] == 'K'][0]
target = [n for n in segs if n['color'] == 'G'][0]
from collections import deque
# BFS ...
action([{'action': 'UP'}, {'action': 'RIGHT'}])
"""
```

- Model can write its own BFS/search algorithm per level
- Batches multiple actions in one call
- **World model is a persistent dict** maintained across turns, not regenerated

**Your agent: Text parsing with regex**

```python
# brain.py
return self._invoke(prompt, temperature=0.0, max_tokens=min(32, self.max_tokens), stop=["\n"])
# → produces: "ACTION=ACTION1"
# → parser decodes that
action, action_data = ARCActionMapper.parse(raw, allowed_actions, ...)
```

- One action per step (no batching)
- LLM must format `ACTION=<NAME> [X=<int> Y=<int>]` — fragile parser dependency
- Brain gets only 32 tokens — **critically undersized** for complex reasoning

---

### 2C. Segmentation vs. Raw Grid

**Duck uses a rich object graph:**

```
segmentation['nodes'] = [
  {id, color, hash, pixels, boundary, children},
  ...
]
segmentation['adjacency_list'] = [[i, j], ...]
```

- Object identity is tracked by content-hash across frames (not pixel coordinates)
- Model reasons about objects, not raw pixel arrays
- Enables reliable "did this object move?" tracking without manual diff

**Your agent uses raw matrix + PIL:**

```python
# state.py / eye.py
pil_img = s0_state.get_pil_image()
text = self._invoke(prompt, image_obj=pil_img)
```

- Vision model sees the whole image but has no structured object map
- Diff is text-based (`get_grid_difference_text`) — correct but less structured
- `max_new_tokens_eye: 256` limits perception depth

---

### 2D. Concurrency Model

**Duck: 16 concurrent games via asyncio**

```python
# solver.py
self._worker_pool = ThreadPoolExecutor(
    max_workers=max(1, int(self.concurrency)),  # =16
    thread_name_prefix="harness-game",
)
semaphore = asyncio.Semaphore(max(1, int(self.concurrency)))
await asyncio.gather(*(asyncio.shield(task) for task in tasks))
```

- vLLM batches all 16 game requests together → **16× throughput**
- Even with 8 games, concurrency eliminates waiting time

**Your agent: Sequential**

```python
# runner.py
obs, state, steps_used = self.play_level(...)
# blocks until this level is done before next
```

- Every action waits for full Transformers inference
- No parallelism at all across games or levels

---

### 2E. Prompt Design & World Model

**Duck prompts the model to maintain a structured world model:**

```
Working world model carried from earlier turns:
- World model: [what entities exist]
- Goal model: [what the objective is]
- Action model: [what actions do]
- Recent findings: [what I discovered]
- Plan: [what I will do next]
- Revise any item above immediately if contradicted.
```

- This "scientist" world model is **parsed from the LLM's text response** and stored
- It's injected back into the next turn → model builds on prior understanding
- Model explicitly writes `World model:`, `Plan:` headings in its response

**Your prompts lack persistent world model injection:**

```python
# brain.py prompt
prompt = f"""{PROMPT_ACTION}
{grid_repr_context}
JSON State Metadata: {current_state.proper_json_repr}
Knowledge Store: {scratch}
...
Reply strictly in format: ACTION=<NAME> [X=<int> Y=<int>]"""
```

- Knowledge comes from files (`cache.scratch()`) — correct, but no structured parse
- No explicit world-model structure extracted and injected per-turn
- `SYSTEM_PROMPT` in your agent **assumes player always exists** — fails on logic puzzles

---

### 2F. Model Configuration Issues

**Your config:**

```yaml
max_new_tokens_brain: 32 # ← CRITICAL: too low for action selection
max_new_tokens_debug: 96 # ← okay for short notes
max_new_tokens_eye: 256 # ← reasonable
temperature: 0.0 # ← correct for determinism
attn_implementation: sdpa # ← correct but not flash_attn
```

**Duck's config (env vars):**

```python
_LOCAL_ANALYZER_MAX_OUTPUT = 0        # vLLM controls this
_LOCAL_ANALYZER_CONTEXT_WINDOW = 32768  # full context
_LOCAL_ANALYZER_TOOL_STEPS = 12       # 12 tool calls per turn
_LOCAL_ANALYZER_TEMPERATURE = 0.6    # allows some exploration
_LOCAL_ANALYZER_ENABLE_THINKING = True  # CoT reasoning
```

- Duck uses **temperature=0.6** (not 0.0) — allows the model to explore
- Thinking (Chain of Thought) tokens are enabled and stripped before action
- Full 32K context window vs. your 8192

---

## 3. What to Improve (Without Breaking Your Agent)

These improvements are **safe**, **non-breaking**, and ordered by **expected impact**:

---

### 🔴 CRITICAL — Maximum Impact

#### Fix 1: Triple the Brain token budget

**File**: [`default.yaml`](file:///e:/code-in-progress/arcgame/configs/default.yaml)

```yaml
# BEFORE
max_new_tokens_brain: 32

# AFTER
max_new_tokens_brain: 128
```

**Why**: 32 tokens lets the model output `ACTION=ACTION1` at best. It cannot reason or select coordinates thoughtfully. 128 gives enough for one reasoning sentence + action.

---

#### Fix 2: Enable temperature > 0 for Brain (add diversity)

**File**: [`default.yaml`](file:///e:/code-in-progress/arcgame/configs/default.yaml)

```yaml
# BEFORE
temperature: 0.0

# AFTER — keep eye/debugger at 0.0, use 0.4 for Brain only
```

**Where**: [`brain.py`](file:///e:/code-in-progress/arcgame/src/arc_agent/chains/brain.py) line 75:

```python
# BEFORE
return self._invoke(prompt, temperature=0.0, max_tokens=min(32, self.max_tokens), stop=["\n"])

# AFTER
return self._invoke(prompt, temperature=0.4, max_tokens=min(128, self.max_tokens), stop=["\n"])
```

**Why**: Deterministic models (temp=0.0) fail to escape stuck states — they'll repeat the same "wrong" action every retry. Temperature adds diversity at decision time.

---

#### Fix 3: Raise context window to 32768

**File**: [`default.yaml`](file:///e:/code-in-progress/arcgame/configs/default.yaml) & [`config.py`](file:///e:/code-in-progress/arcgame/src/arc_agent/config.py):

```yaml
max_context_length: 32768 # was 8192
```

**Why**: ARC puzzles accumulate history. 8192 tokens (~6000 words) is quickly exhausted when you have a full actions log + scratch + state. Duck uses 32768.

---

### 🟠 HIGH IMPACT — World Model Extraction

#### Fix 4: Extract and re-inject a structured world model per turn

**File**: [`chains/brain.py`](file:///e:/code-in-progress/arcgame/src/arc_agent/chains/brain.py)

Add a method to parse the Eye response for world model fields:

```python
import re

_WORLD_MODEL_LABELS = ["World model", "Goal model", "Action model", "Recent findings", "Plan"]

def extract_world_model(text: str) -> dict:
    """Parse labeled sections from Eye/Debugger responses."""
    result = {}
    for label in _WORLD_MODEL_LABELS:
        pattern = rf"(?:^|\n){re.escape(label)}:\s*(.*?)(?=\n[A-Z][a-z]+ model:|\Z)"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            result[label.lower().replace(" ", "_")] = m.group(1).strip()[:300]
    return result
```

Then inject `world_model_text` into `brain.decide_action`'s prompt, prefixed as:

```
Working world model from last turn:
- World model: <...>
- Goal: <...>
- Plan: <...>
```

**Why**: This is the single biggest cognitive upgrade — the model builds on its own prior understanding instead of starting from scratch each step.

---

#### Fix 5: Fix SYSTEM_PROMPT player-assumption bias

**File**: [`chains/prompts.py`](file:///e:/code-in-progress/arcgame/src/arc_agent/chains/prompts.py)

```python
# ADD THIS to SYSTEM_PROMPT — replace the player-always-exists assumption
"Some games are logic or layout puzzles with no explicit player avatar. "
"Do not assume a player exists; the relevant state may be an object, region, "
"cursor, selector, or whole-board configuration.\n"
"A common failure mode is treating a segmented edge bar (timer/step counter) "
"as clickable puzzle pieces. If a strip of small blocks sits flush against an "
"edge and only that strip changes — classify it as HUD/timer, NOT an object.\n"
```

**Why**: Duck's system prompt explicitly warns against the HUD-timer misidentification bug. Your prompt doesn't — causing the agent to waste steps clicking the step counter.

---

### 🟡 MEDIUM IMPACT — Speed Improvements

#### Fix 6: Enable Action Batching

**File**: [`agent/runner.py`](file:///e:/code-in-progress/arcgame/src/arc_agent/agent/runner.py) — `attempt_one_shot()` already does this for one-shot plans. Extend to the step loop:

In `run_step_loop`, instead of sending 1 action per LLM call, allow Brain to output a **sequence of 2-3 actions** when confident:

```python
# brain.py — change stop condition for known-good sequences
# If world model says "I know the path", use max_tokens=256, no stop=["\n"]
# Then parse multiple ACTION= lines
plan = ARCActionMapper.parse_plan(raw, valid_actions, grid_shape)
if plan and len(plan) > 1:
    # execute as batch (already implemented in attempt_one_shot)
```

**Why**: Duck batches multiple actions per tool call. Every batched action saves one full LLM roundtrip. For a 20-step level, going from 1→3 actions/call reduces LLM calls by 66%.

---

#### Fix 7: Replace flash_attention with flash_attention_2

**File**: [`default.yaml`](file:///e:/code-in-progress/arcgame/configs/default.yaml):

```yaml
# BEFORE
attn_implementation: sdpa

# AFTER (if A100/H100/RTX Pro 6000 supports it)
attn_implementation: flash_attention_2
```

**Why**: `flash_attention_2` is 20-40% faster than `sdpa` on Ampere/Hopper GPUs and uses less VRAM. RTX Pro 6000 supports it.

---

#### Fix 8: Cache Eye/Debugger results more aggressively

**File**: [`agent/arc_langchain_agent.py`](file:///e:/code-in-progress/arcgame/src/arc_agent/agent/arc_langchain_agent.py)

Your `debugger_cache` by `(state_hash, action_sig)` is good. Extend to **Eye chain**:

```python
# In execute_action / run_step_loop
# Cache visual_analysis by (state_hash) — same state → same analysis
_eye_cache: dict[str, str] = {}

if next_state.state_hash in _eye_cache:
    visual_analysis = _eye_cache[next_state.state_hash]
else:
    visual_analysis = self.agent.eye.analyse_visual(...)
    _eye_cache[next_state.state_hash] = visual_analysis
```

**Why**: Eye runs a multimodal model call on every changed state. Many states are revisited. Caching by state hash avoids redundant vision inference.

---

### 🟢 LOW IMPACT / POLISH

#### Fix 9: Add oscillation-aware planning in one-shot

**File**: [`agent/runner.py`](file:///e:/code-in-progress/arcgame/src/arc_agent/agent/runner.py) — `attempt_one_shot()` already has a `zero_diff_streak` check. Duck additionally checks for **3-step oscillation** (A→B→A pattern):

```python
# In run_step_loop, after recording transition:
if len(self.agent.memory.state_history) >= 3:
    if self.agent.memory.state_history[-1] == self.agent.memory.state_history[-3]:
        debug_note += "\n[OSCILLATION] Detected A→B→A loop. Must break cycle."
```

This already partially exists via `oscillation_target()` in `TrajectoryMemory` — make sure it's always injected into `context_note`.

---

#### Fix 10: Raise speculative plan cap from 15 to 25

**File**: [`configs/default.yaml`](file:///e:/code-in-progress/arcgame/configs/default.yaml):

```yaml
speculative_plan_max_steps: 25 # was 15
```

**File**: [`agent/runner.py`](file:///e:/code-in-progress/arcgame/src/arc_agent/agent/runner.py) line 84:

```python
max_plan_execution = min(25, len(plan))  # was 15
```

**Why**: Some levels require 15+ setup steps before scoring. Cutting the speculative plan at 15 means you fall back to the slow closed loop for levels where you had a valid plan.

---

## 4. Summary Priority Table

| #   | Change                                      | File(s)                     | Impact      | Difficulty |
| --- | ------------------------------------------- | --------------------------- | ----------- | ---------- |
| 1   | Brain max_tokens: 32→128                    | `default.yaml`, `brain.py`  | 🔴 Critical | Easy       |
| 2   | Temperature: 0.0→0.4 for Brain              | `brain.py`                  | 🔴 Critical | Easy       |
| 3   | Context window: 8192→32768                  | `default.yaml`, `config.py` | 🔴 Critical | Easy       |
| 4   | World model extraction + injection          | `brain.py`, `eye.py`        | 🟠 High     | Medium     |
| 5   | Fix SYSTEM_PROMPT player bias + HUD warning | `chains/prompts.py`         | 🟠 High     | Easy       |
| 6   | Action batching in step loop                | `runner.py`, `brain.py`     | 🟡 Medium   | Medium     |
| 7   | flash_attention_2                           | `default.yaml`              | 🟡 Medium   | Easy       |
| 8   | Eye result cache by state_hash              | `arc_langchain_agent.py`    | 🟡 Medium   | Easy       |
| 9   | Oscillation always injected                 | `runner.py`                 | 🟢 Low      | Easy       |
| 10  | Speculative plan cap 15→25                  | `default.yaml`, `runner.py` | 🟢 Low      | Easy       |

---

## 5. What You Should NOT Adopt from Duck

| Duck Feature        | Why NOT for your agent                                                                                     |
| ------------------- | ---------------------------------------------------------------------------------------------------------- |
| vLLM server         | Requires full infra rebuild. Your Transformers pipeline is fine for now.                                   |
| Qwen3-8B model      | Needs completely different model loading and tokenizer. Not a drop-in swap.                                |
| Full TAAF framework | TAAF is the competition framework wrapper — replacing it would break your Kaggle submission path entirely. |
| Asyncio concurrency | Only valuable when vLLM can batch. With Transformers, concurrent calls would OOM.                          |
| Python sandbox tool | Major architectural change — not backward compatible with your chain-based design.                         |

---

## 6. Expected Score Improvement

Based on the nature of the changes:

| Batch   | Changes          | Est. Score Delta |
| ------- | ---------------- | ---------------- |
| Batch 1 | Fixes 1, 2, 3, 5 | +1.5 to +2.5 pts |
| Batch 2 | Fixes 4, 6, 8    | +0.5 to +1.0 pts |
| Batch 3 | Fixes 7, 9, 10   | +0.2 to +0.5 pts |

> [!NOTE]
> The competition topper's biggest edge is **vLLM + concurrency** (throughput) and **Python tool sandbox** (algorithmic BFS search). Fixes 1-5 close most of the **quality** gap. Full throughput parity requires vLLM migration.

> [!TIP]
> Start with **Fixes 1, 2, 3, and 5** — all in YAML/prompts, zero risk of breaking the agent, and highest expected return. Test on 3-5 offline games before submitting.

# Goal + Plan World-Model Adaptation Plan

> Adapt Duck's goal/plan approach into your existing LangChain multi-chain agent **without breaking the existing architecture**. All changes are backward-compatible additive layers.

---

## What We're Adopting

Duck's key cognitive loop:

```
Turn N → LLM response contains labeled sections →
  World model: ...
  Goal model: ...
  Action model: ...
  Plan: ...
→ These are extracted → stored in memory dict →
  injected into Turn N+1's prompt
→ LLM now reasons ON TOP OF its prior beliefs
```

Your agent currently has: **scratch files + KnowledgeCache** but no turn-by-turn structured carry-forward.  
We add a thin **`WorldModel`** layer that lives in `ARCLangChainAgent`, updated every turn.

---

## Architecture Overview (What Changes, What Stays)

```
BEFORE (your current flow):
  enter_level → eye.assume() → [scratch file written]
    └─ for each step:
         brain.decide_action(scratch, actions_log) → "ACTION=X"
         execute_action()
         eye.analyse_visual()        ← separate LLM call per step
         debugger.validate()         ← separate LLM call per step

AFTER (goal+plan layered on top):
  enter_level → eye.assume() → [scratch + world_model seeded]
    └─ for each step:
         brain.decide_action(scratch + world_model_inject) → "PLAN:...\nACTION=X"
         WorldModel.update_from_brain_response(raw_text)   ← NEW (no LLM call)
         execute_action()
         eye.analyse_visual()        ← unchanged
         debugger.validate()         ← unchanged
         WorldModel.update_from_debugger(debug_note)       ← NEW (no LLM call)
```

**Zero extra LLM calls. Pure text parsing overhead.**

---

## Speed Impact

| Phase                  | Before                                 | After                                                    |
| ---------------------- | -------------------------------------- | -------------------------------------------------------- |
| Per-step LLM calls     | Eye + Brain + Debugger = 3 calls       | Same 3 calls (unchanged)                                 |
| World model update     | None                                   | Pure Python string parse, ~0.1ms                         |
| Brain prompt injection | `scratch[:400]` flat text              | `scratch[:400]` + world_model (100-300 tokens more)      |
| One-shot plan quality  | No prior context in plan               | Plan seeded from world model → likely fewer wasted steps |
| Retry iteration        | Reviewer rewrites scratch from scratch | World model reset + reviewer → better recovery           |
| Net inference time     | Baseline                               | **Same** — no added LLM calls                            |
| Net step efficiency    | Agent rediscovers facts every step     | Agent builds on prior turn → **fewer total steps**       |

> **Speed improvement is indirect**: we don't reduce LLM calls, but fewer steps are needed per level because the agent no longer re-derives facts it already knows. Each step is slightly more expensive (+100-300 token context overhead), but total steps drop.

---

## Proposed Changes

---

### Component 1: New `WorldModel` class

#### [NEW] [`world_model.py`](file:///e:/code-in-progress/arcgame/src/arc_agent/memory/world_model.py)

Pure data class + parser. **No LLM calls. No imports beyond `re`.**

```python
"""Turn-persistent structured world model — goal, plan, action model, findings."""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

_LABELS = [
    "World model", "Goal model", "Action model",
    "Recent findings", "Open questions", "Plan", "Cross-level notes",
]
_LABEL_LOWER_MAP = {label.lower(): label for label in _LABELS}
_LABEL_TARGETS = tuple(f"{label.lower()}:" for label in _LABELS)
_MAX_FIELD_CHARS = 300


@dataclass
class WorldModel:
    """Structured carry-forward belief state extracted from LLM responses."""
    world_model: str = ""
    goal_model: str = ""
    action_model: str = ""
    recent_findings: str = ""
    open_questions: str = ""
    current_plan: str = ""
    cross_level_notes: str = ""

    def update_from_text(self, text: str) -> None:
        """Parse labeled sections from any LLM response and merge non-empty values."""
        if not text.strip():
            return
        extracted = _extract_labeled_blocks(text, _LABELS)
        for key, value in extracted.items():
            if value:
                setattr(self, key, value[:_MAX_FIELD_CHARS])

    def reset_level_fields(self) -> None:
        """Clear per-level fields on level transition; keep cross-level notes."""
        self.world_model = ""
        self.goal_model = ""
        self.action_model = ""
        self.recent_findings = ""
        self.open_questions = ""
        self.current_plan = ""

    def is_empty(self) -> bool:
        return not any([
            self.world_model, self.goal_model, self.action_model,
            self.recent_findings, self.current_plan,
        ])

    def to_prompt_lines(self) -> list[str]:
        """Render as injection block for the Brain prompt."""
        entries = [
            ("World model", self.world_model),
            ("Goal model", self.goal_model),
            ("Action model", self.action_model),
            ("Recent findings", self.recent_findings),
            ("Open questions", self.open_questions),
            ("Plan", self.current_plan),
            ("Cross-level notes", self.cross_level_notes),
        ]
        lines = [f"- {label}: {value}" for label, value in entries if value]
        if not lines:
            return []
        return [
            "Working world model from prior turn (update if contradicted by new state):",
            *lines,
            "- Revise any item immediately if current state contradicts it.",
        ]

    def to_prompt_block(self) -> str:
        lines = self.to_prompt_lines()
        return "\n".join(lines) if lines else ""


def _extract_labeled_blocks(content: str, labels: list[str]) -> dict[str, str]:
    normalized = {label.lower(): label for label in labels}
    targets = tuple(f"{label.lower()}:" for label in labels)
    extracted: dict[str, list[str]] = {label: [] for label in labels}
    current_label: str | None = None

    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        candidate = stripped
        while candidate.startswith(("-", "*")):
            candidate = candidate[1:].lstrip()
        lowered = candidate.lower()

        matched_label: str | None = None
        inline_value = ""
        for target in targets:
            if lowered.startswith(target):
                matched_label = normalized[target[:-1]]
                inline_value = candidate[len(target):].strip()
                break

        if matched_label is not None:
            current_label = matched_label
            if inline_value:
                extracted[current_label].append(inline_value)
            continue

        if current_label is not None and stripped:
            extracted[current_label].append(stripped)

    result = {}
    for label, lines in extracted.items():
        joined = "\n".join(lines).strip()
        if joined:
            key = label.lower().replace(" ", "_")
            result[key] = joined
    return result
```

---

### Component 2: Update `memory/__init__.py`

#### [MODIFY] [`memory/__init__.py`](file:///e:/code-in-progress/arcgame/src/arc_agent/memory/__init__.py)

Add `WorldModel` to the memory module export.

```python
# Add to existing imports
from .world_model import WorldModel
```

---

### Component 3: Update `BrainChain.decide_action` to inject world model

#### [MODIFY] [`chains/brain.py`](file:///e:/code-in-progress/arcgame/src/arc_agent/chains/brain.py)

Two changes:

1. Accept optional `world_model_block: str` parameter
2. Inject it before the prompt body
3. Raise token budget to 128 and temperature to 0.4
4. Ask LLM to output plan header before ACTION line

```python
def decide_action(
    self,
    game_id: str,
    level: int,
    s0_state: ARCState,
    current_state: ARCState,
    valid_actions: List[Any],
    context_note: str,
    cache: KnowledgeCache,
    world_model_block: str = "",        # ← NEW param (default="" = no-op)
) -> str:
    """Determines next discrete or complex coordinate action."""
    actions_log = cache.actions_log(game_id, level)
    scratch = cache.scratch(game_id)

    if current_state.step == 0:
        grid_repr_context = f"S0 Matrix (Hash: {s0_state.state_hash[:12]}):\n{s0_state.text_repr}"
    else:
        grid_repr_context = (
            f"Initial S0 Hash: {s0_state.state_hash[:12]} | Current St Hash: {current_state.state_hash[:12]}\n"
            f"Grid Shape: {current_state.grid.shape if current_state.grid is not None else 'Unknown'}"
        )

    action_names = [getattr(a, "name", str(a)) for a in valid_actions]

    # World model injection block (empty string → blank line = no overhead)
    world_model_section = f"\n{world_model_block}\n" if world_model_block else ""

    prompt = f"""{PROMPT_ACTION}
{world_model_section}
{grid_repr_context}

JSON State Metadata:
{current_state.proper_json_repr}

Knowledge Store: {scratch}
Actions Log: {actions_log}
Tracker/Loop/Sprite Context: {context_note}
Legal actions (Prohibited/redundant ones are already filtered - choose from this list): {action_names}

Optionally prefix with: Plan: <one sentence goal+next step>
Reply strictly in format: ACTION=<NAME> [X=<int> Y=<int>]
Next action:"""

    return self._invoke(
        prompt,
        temperature=0.4,                          # ← was 0.0
        max_tokens=min(128, self.max_tokens),      # ← was min(32, self.max_tokens)
        stop=["\n"],
    )
```

> [!NOTE]
> `stop=["\n"]` still works: the model will output `Plan: navigate left  ACTION=ACTION3` and the parser still finds `ACTION=` in the string. If you want the plan extracted too, remove `stop` and parse the full response.

---

### Component 4: `BrainChain.one_shot_plan` — inject world model

#### [MODIFY] [`chains/brain.py`](file:///e:/code-in-progress/arcgame/src/arc_agent/chains/brain.py) — `one_shot_plan` method

```python
def one_shot_plan(
    self,
    game_id: str,
    level: int,
    s0_state: ARCState,
    valid_actions: List[Any],
    cache: KnowledgeCache,
    world_model_block: str = "",        # ← NEW param
) -> str:
    """Synthesizes speculative macro-plan sequence for rapid execution."""
    ostate = cache.ostate(game_id)
    scratch = cache.scratch(game_id)
    actions_log = cache.actions_log(game_id, level)
    valid_names = [getattr(a, "name", str(a)) for a in valid_actions]
    world_model_section = f"\n{world_model_block}\n" if world_model_block else ""

    prompt = f"""{PROMPT_ACTION}
{world_model_section}
Synthesize a ONE-SHOT plan for Level {level}. Format EACH line strictly as:
ACTION=<NAME> [X=<int> Y=<int>]

Cross-Level S0 Analysis: {ostate}
Knowledge Store: {scratch}
Actions Log: {actions_log}
S0 Matrix (Hash: {s0_state.state_hash[:12]}): {s0_state.text_repr}
Valid Actions: {valid_names}

Ordered action sequence:"""

    return self._invoke(prompt, temperature=0.0, max_tokens=256)
```

---

### Component 5: Update `ARCLangChainAgent` — own the WorldModel, thread it through

#### [MODIFY] [`agent/arc_langchain_agent.py`](file:///e:/code-in-progress/arcgame/src/arc_agent/agent/arc_langchain_agent.py)

Four changes:

1. Import and instantiate `WorldModel` in `__init__`
2. Seed world model from Eye output in `enter_level`
3. Pass `world_model_block` to `brain.decide_action` and `brain.one_shot_plan`
4. Update world model from Brain raw response and Debugger note each step
5. Reset world model on level transition

```python
# In __init__:
from ..memory.world_model import WorldModel   # add import

self.world_model = WorldModel()

# In enter_level(), after eye.assume() / eye.compare_assume():
eye_text = self.eye.assume(game_id, level, s0_state, self.cache)
# (or compare_assume result)
self.world_model.reset_level_fields()
self.world_model.update_from_text(eye_text)   # seed plan from S0 perception

# In decide_action():
world_model_block = self.world_model.to_prompt_block()
raw = self.brain.decide_action(
    game_id, level, s0_state, current_state,
    allowed_actions, context_note, self.cache,
    world_model_block=world_model_block,          # ← pass it
)
self.world_model.update_from_text(raw)            # ← absorb any Plan: lines
action, action_data = ARCActionMapper.parse(raw, allowed_actions, ...)
```

And in `runner.py`'s `run_step_loop`, after `debugger.validate()`:

```python
# After debug_note is computed:
self.agent.world_model.update_from_text(debug_note)
```

And in `play_level` before calling `run_step_loop` with `iteration > 1`:

```python
# On retry, reset level fields but keep cross-level notes:
self.agent.world_model.reset_level_fields()
```

---

### Component 6: Update `SYSTEM_PROMPT` in `chains/prompts.py`

#### [MODIFY] [`chains/prompts.py`](file:///e:/code-in-progress/arcgame/src/arc_agent/chains/prompts.py)

Append to end of `SYSTEM_PROMPT` string (no removal of existing text):

```python
SYSTEM_PROMPT = (
    # ... existing text unchanged ...
    "\nSome games are logic or layout puzzles with no explicit player avatar. "
    "Do not assume a player exists; the relevant state may be an object, region, "
    "cursor, selector, or whole-board configuration.\n"
    "A common failure mode is treating a segmented edge bar (step counter/timer) "
    "as clickable puzzle pieces. If a strip of small blocks sits flush against an "
    "edge and only that strip changes — classify it as HUD/timer state, NOT an object.\n"
    "When you have a working plan, output it before your action: 'Plan: <one sentence>'\n"
    "Useful optional label prefixes in your response: "
    "'World model:', 'Goal model:', 'Action model:', 'Recent findings:', 'Plan:'\n"
)
```

---

### Component 7: Config — raise token budget and context window

#### [MODIFY] [`configs/default.yaml`](file:///e:/code-in-progress/arcgame/configs/default.yaml)

```yaml
model:
  max_new_tokens_brain: 128 # was 32
  max_context_length: 32768 # was 8192
  attn_implementation: flash_attention_2 # was sdpa
```

---

## File Change Summary

| File                                         | Type       | Change                                                               |
| -------------------------------------------- | ---------- | -------------------------------------------------------------------- |
| `src/arc_agent/memory/world_model.py`        | **NEW**    | `WorldModel` dataclass + `_extract_labeled_blocks` parser            |
| `src/arc_agent/memory/__init__.py`           | **MODIFY** | Export `WorldModel`                                                  |
| `src/arc_agent/chains/brain.py`              | **MODIFY** | `world_model_block` param, token 32→128, temp 0.0→0.4                |
| `src/arc_agent/agent/arc_langchain_agent.py` | **MODIFY** | Own `WorldModel`, seed from Eye, pass to Brain, update from Debugger |
| `src/arc_agent/agent/runner.py`              | **MODIFY** | `update_from_text(debug_note)`, `reset_level_fields()` on retry      |
| `src/arc_agent/chains/prompts.py`            | **MODIFY** | Append HUD warning + plan label guidance                             |
| `configs/default.yaml`                       | **MODIFY** | Raise token budget + context window                                  |

**No other files touched. No chain removed. No import graph broken.**

---

## Verification Plan

### Automated Tests

```bash
cd e:\code-in-progress\arcgame
python -m pytest tests/test_actions.py -v
```

### Manual Smoke Test

Run 2-3 offline games and verify:

1. Brain responses now occasionally include `Plan: ...` prefix lines
2. `world_model.to_prompt_block()` is non-empty after step 1 (Eye seeded it)
3. No regression in action parsing (ACTION= still found correctly)
4. Debugger output updates `world_model.recent_findings`

### Expected Outcome

- Fewer stuck loops (world model carries "I tried X, it was blocked")
- One-shot plan quality improves (seeded from Eye S0 analysis + prior level knowledge)
- Retry iterations start smarter (world model seeds from prior attempt's findings)

---

> [!IMPORTANT]
> **Breaking-risk check**: `brain.decide_action` signature change adds `world_model_block=""` with a default. Every existing call site continues to work without changes — backward compatible.

> [!TIP]
> After implementing, set `max_new_tokens_brain: 64` first (not 128) to test that the parser still extracts `ACTION=` correctly even when the model outputs `Plan: ... ACTION=...`. Then raise to 128.

Let me read the remaining key files to make the plan precise before touching any code.

Viewed debugger.py:1-100
Viewed state.py:1-240
Viewed knowledge.py:1-244
Viewed reviewer.py:1-56

Now I have the full picture. Let me write the implementation plan:

Created implementation_plan.md

The implementation plan is ready. Here's the short version of what it proposes and why it's safe:

---

## What We're Building

A **thin `WorldModel` layer** that slides between your existing chains — no chain is removed or replaced:

```
Eye.assume()   →  seeds WorldModel  (goal, plan, entity map)
                       ↓
Brain.decide() ←  reads WorldModel  (injected into prompt)
                       ↓
Brain response →  updates WorldModel (parses Plan: / World model: labels)
                       ↓
Debugger.validate() → updates WorldModel.recent_findings
                       ↓
Next step... Brain already knows what failed last time
```

## How Fast Is It?

|                            | Before                              | After                                                   |
| -------------------------- | ----------------------------------- | ------------------------------------------------------- |
| LLM calls per step         | 3 (Eye/Brain/Debug)                 | **Same 3** — no added calls                             |
| Python overhead per step   | 0                                   | ~0.1ms (string parse)                                   |
| Brain context size         | +~200 tokens from world model block | Slight prompt growth                                    |
| Steps per level (expected) | N                                   | **~10-30% fewer** — agent stops re-deriving known facts |

**Net result: same wall-clock per step, meaningfully fewer steps per level.**

## 7 Files, All Backward-Compatible

1. **NEW** `memory/world_model.py` — `WorldModel` dataclass + label parser
2. `memory/__init__.py` — export it
3. `chains/brain.py` — `world_model_block=""` param + token 32→128 + temp 0.4
4. `agent/arc_langchain_agent.py` — own `WorldModel`, seed from Eye, thread through
5. `agent/runner.py` — update model from debugger, reset on retry
6. `chains/prompts.py` — HUD warning + plan label guidance appended
7. `configs/default.yaml` — token budget + context window raised

All new params have **defaults** so existing call sites work unchanged. Proceed?
