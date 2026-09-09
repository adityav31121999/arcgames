# ARC-AGI-3 LangChain Inference Agent

## Reliability and context handling

Run `notebooks/assumeAndPlay.ipynb` in a fresh session after updating the source
dataset. Its setup preserves native Transformers classes instead of applying
global configuration patches. Before gameplay it tests raw text generation and
recognition of a solid red image, saving runtime details and unsanitized answers
to `/kaggle/working/model_health.json`. A failure stops startup; inspect the report
to distinguish garbled text, image failures, backend exceptions, and strict answer
format mismatches. Passing these small checks does not establish game competence.
NVIDIA documents this NVFP4 checkpoint for vLLM; loading its weights in Transformers
alone does not establish runtime compatibility.

Board arrays are rendered to PIL images and passed to the checkpoint's multimodal
processor with the text prompt. The native chat template inserts image placeholders;
the processor supplies image start/end markers, image token slots, and pixel tensors
for the vision model. Eye receives both the previous and current board
for transition evaluation. Eye also receives Brain's intended plan and expected
observable effect.

The wrapper counts the processor-expanded input tokens, including image slots, and
reserves generation tokens before moving inputs to the GPU. It uses the smaller of
`model.max_context_length` and the loaded model's supported context size. When needed,
it compacts optional memory/history sections by priority and recency. Required task
instructions, native system messages, and images are preserved. If those alone do
not fit, inference returns an explicit context-budget error. Working memory keeps
recent entries; consolidated updates remain in `memory_history.md` alongside the
action logs and level analyses.

Set `DEBUG_LLM_CONTEXT=true` to print input tokens, image slots, output reservation,
effective context limit, and compacted sections. The same data is available in
`GemmaTransformersChatModel.last_context_usage`. Token capacity does not guarantee
that a particular prompt and image count fit GPU memory.

Fast evaluation uses measured board differences, with full Eye evaluation
on stalled, repeated, or uncertain transitions, after a failed evaluation, and every
`agent.full_eval_interval` steps (default 8). Speculative execution is capped at this
interval and hands off for evaluation before Brain replans. Set
`agent.speculative_plan_max_steps=0` to disable speculative planning. Malformed
Eye observations and Brain reviews receive one retry; `eye.last_result` and
`brain.last_review_result` report failure separately from world-model facts. Only externally verified HUD coordinates
are masked, so border movement and uncertain border targets remain visible.

Generation defaults to `repeat_penalty=1.0`. Brain requests an explicit
`[END_ACTION]` marker so a blank line does not cut off its action. The sanitizer
rescues explicit actions across lines and only applies action rescue to action
responses. See [the repair plan](repair_plan.md) for validation and runtime checks.

An agentic reasoning framework for **ARC-AGI-3** (ARC Prize 2026), powered by **LangChain** and **Hugging Face `transformers`**, specifically optimized for **NVIDIA RTX PRO 6000 (96 GB VRAM)** running **[`nvidia/Gemma-4-26B-A4B-NVFP4`](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4)** (and Gemma 4 MoE architectures).

---

## 🚀 Key Highlights

- **Hugging Face Transformers Backend**: Native integration with `AutoProcessor` and `AutoModelForImageTextToText` / `AutoModelForCausalLM` with `torch.bfloat16`, SDPA/FlashAttention, and `device="cuda:0"`.
- **Hardware Optimization (RTX PRO 6000 96GB VRAM)**: With 96GB of high-bandwidth VRAM, the entire Gemma 4 MoE 26B-A4B model (both vision tower and all expert routing layers) fits directly into GPU memory with zero CPU bottlenecks, enabling full context handling and rapid KV-cache throughput.
- **LangChain Modular Architecture**: Structured with LangChain Expression Language (LCEL) Runnables and custom `BaseChatModel` wrappers:
  - 👁️ **Perception (Eye Chain)**: Multimodal spatial understanding of S0 grid layout, goal anchors, and cross-level state deltas.
  - 🧠 **Policy (Brain Chain)**: High-level reasoning, legal action filtering (excluding visited loops and oscillating paths), sprite bounding-box guidance, and speculative one-shot macro planning, and failed-attempt reviews saved as unverified hypotheses.
- **Robust Spatial & Trajectory Memory**:
  - Active sprite bounding-box tracker (isolating cursor movement from static terrain).
  - Oscillation detection ($A \rightarrow B \rightarrow A$ filter).
  - Dynamic step budgets derived from competition environment baselines.
  - Tournament timeout budget monitor (preventing Kaggle 9-hour runtime kills).

---

## 📁 Repository Structure

```
arcgame/
├── pyproject.toml              # Modern Python packaging & dependencies
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation & setup guide
├── .env.example                # Example environment variables
├── configs/
│   ├── default.yaml            # Single RTX PRO 6000 (96GB VRAM) configuration
│   └── kaggle_offline.yaml     # Kaggle offline input paths configuration
├── src/
│   └── arc_agent/
│       ├── config.py           # Pydantic settings & schema validation
│       ├── core/
│       │   ├── color_palette.py # ARC 16-color colormap & PIL/PNG rendering
│       │   ├── diff.py         # Fast NumPy pixel diff & border stripping
│       │   ├── state.py        # ARCState, ARCTransition, lazy rendering & JSON metadata
│       │   ├── actions.py      # ActionSignature, ARCActionMapper & fallback heuristics
│       │   └── resolver.py     # GameStateResolver (WIN, GAME_OVER, LEVEL_UP)
│       ├── memory/
│       │   ├── trajectory.py   # TrajectoryMemory (loop alerts, oscillation avoidance, sprite tracking)
│       │   └── knowledge.py    # KnowledgeCache & persistent markdown scratchpads
│       ├── models/
│       │   ├── gemma_transformers.py # LangChain ChatModel wrapping Transformers & AutoProcessor
│       │   └── factory.py      # Model factory with local offline Kaggle weight locator
│       ├── chains/
│       │   ├── prompts.py      # System and prompt templates
│       │   ├── eye.py          # Multimodal perception chains (S0 assumption & visual diff)
│       │   ├── brain.py        # Next-action selection & speculative macro planning
│       ├── agent/
│       │   ├── arc_langchain_agent.py # High-level ARCAgent coordinating chains & memory
│       │   └── runner.py       # Execution loops, baseline step limits & timeout monitor
│       └── utils/
│           ├── suppression.py  # C/C++ and HF warning suppressors
│           ├── locator.py      # Offline weight directory auto-locator
├── notebooks/
│   ├── sample_run_single_game.ipynb # Interactive single game runner with visual diagnostics
│   └── submission_run.ipynb         # Full Kaggle tournament submission runner
├── scripts/
│   ├── run_inference.py        # CLI entry point to test single games or batch offline/online
│   └── bundle_kaggle.py        # Packages repository into wheel or Kaggle zip bundle
└── tests/
    ├── test_state.py           # Tests for ARCState, hashing, and metadata
    ├── test_actions.py         # Tests for action parsing, coordinate heuristics & fallbacks
    ├── test_diff.py            # Tests for NumPy visual diff and border stripping
    ├── test_trajectory.py      # Tests for loop detection, oscillation prevention & sprite tracking
    └── test_chains.py          # Tests for Eye perception and Brain planning/review
```


---

## 🛠️ Quickstart

### 1. Installation

```bash
# Clone and install dependencies
pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and set your local environment paths:

```bash
cp .env.example .env
```

Default configuration in `configs/default.yaml`:
```yaml
model:
  model_id: "nvidia/Gemma-4-26B-A4B-NVFP4"
  device: "cuda:0"
  torch_dtype: "bfloat16"
  attn_implementation: "sdpa"
  max_context_length: 81930
```

---

## 🎮 Running Inference

### Run a Single Game (Local or Kaggle Offline Mode)

```bash
python scripts/run_inference.py --config configs/default.yaml --game s5i5
```

### Run Dry-Run / Test with Mock LLM (No GPU/Weights Required)

```bash
python scripts/run_inference.py --mock --game s5i5
```

### Run Full Offline Batch Evaluation

```bash
python scripts/run_inference.py --config configs/kaggle_offline.yaml
```

Upon completion, the agent automatically renders `/kaggle/working/submission.parquet` containing evaluation results.

---

## 📓 Jupyter Notebooks

Two ready-to-use notebooks are available in `notebooks/`:

### 1. `notebooks/sample_run_single_game.ipynb`
- **Purpose**: Interactive single-game exploration and debugging (e.g. `s5i5`).
- **Features**: Live step-by-step visual grid display, perception/reasoning chain introspection, trajectory inspection, and markdown scratchpad review.

### 2. `notebooks/submission_run.ipynb`
- **Purpose**: Full automated Kaggle tournament submission notebook.
- **Features**:
  - Automatically detects real competition rerun (`KAGGLE_IS_COMPETITION_RERUN` / `TRUE_SUBMISSION`).
  - Prepares CUDA libraries, offline `arc-agi` wheels, and mounted source package.
  - Gateway polling and synchronization (`_wait_for_gateway`) for live competition mode.
  - Multi-level execution across all tournament environments with soft deadline protection.
  - Exports `/kaggle/working/submission.parquet` conforming to Kaggle competition requirements.

---

## 🧪 Running Unit Tests


Execute the comprehensive test suite with `pytest`:

```bash
pytest tests/ -v
```

---

## 📦 Bundling for Kaggle Notebook Submissions

To export the codebase for uploading as a Kaggle Dataset or Wheel:

```bash
python scripts/bundle_kaggle.py --format both --out dist
```

This generates `dist/arc_agent-0.1.0-py3-none-any.whl` and `dist/arc_agent_source.zip`.

---

## 📜 License

MIT License. Designed for the ARC-AGI-3 Competition (ARC Prize 2026).


### Root Cause Analysis: Why the LLM is Producing Non-English Tokens

The appearance of non-English tokens (CJK Chinese ideographs, Cyrillic, etc.) and the subsequent `[CRITICAL LLM FAILURE]` (aborting at Step 12 and Step 41) is caused by a compounding chain of **generation parameters, prompt tokenization, and tokenizer mechanics**:

---

### 1. Primary Cause: `repetition_penalty = 1.05` on a Long Multilingual Prompt

In [`src/arc_agent/models/gemma_transformers.py`](file:///e:/code-in-progress/arcgame/src/arc_agent/models/gemma_transformers.py#L200) and [`src/arc_agent/config.py`](file:///e:/code-in-progress/arcgame/src/arc_agent/config.py#L29):
```python
repeat_penalty: float = Field(default=1.05)
```
In Hugging Face Transformers, `RepetitionPenaltyLogitsProcessor` penalizes **all tokens present in the input prompt (`input_ids`)**, not just newly generated tokens:
- If $\text{logit} > 0$: $\text{logit} = \frac{\text{logit}}{\text{penalty}}$
- If $\text{logit} < 0$: $\text{logit} = \text{logit} \times \text{penalty}$

The prompt passed to the Brain chain is **over 1,000 tokens long**. It contains nearly every common English connective word (`the`, `to`, `is`, `a`, `and`, `in`, `for`), all punctuation, all action names (`ACTION1` through `ACTION6`), direction words (`up`, `down`, `left`, `right`), and numbers `0-9`.

1. **English Tokens Suppressed**: Every single one of these English words and punctuation marks receives a logit penalty right from token 0.
2. **Multilingual Tokens Untouched**: `Gemma-4-26B-A4B` has a **256,000-token multilingual vocabulary** containing Chinese, Japanese, Cyrillic, Arabic, and Devanagari tokens. **None of these foreign tokens appeared anywhere in the prompt.** Their logits receive **zero penalty**.
3. **Greedy Decoding Flips to Foreign Tokens**: Because `BrainChain` runs greedy decoding ([`temperature=0.0`](file:///e:/code-in-progress/arcgame/src/arc_agent/chains/brain.py#L149)), as soon as a heavily penalized English candidate falls below an unpenalized non-English synonym or homoglyph, the model deterministically selects the non-English token.
4. **Self-Attention Cascade**: Once the first non-English token is emitted, self-attention attends to it, causing the model to generate the next 20–45 characters in Chinese or Cyrillic. This explains the exact ratio observed in your log:
   $$\frac{25}{212} \approx 0.12, \quad \frac{38}{239} \approx 0.16, \quad \frac{40}{157} \approx 0.25$$

---

### 2. Secondary Cause: NVFP4 Quantized MoE Routing Instability

The model is **`nvidia/Gemma-4-26B-A4B-NVFP4`** (a 4-bit NormalFloat quantized Mixture-of-Experts).
- In 4-bit MoE models, router gating weights operate at lower numerical precision.
- When English logits are depressed by the repetition penalty, the router's top-$k$ gating easily misroutes tokens to experts trained on multilingual web corpora, accelerating language drift.

---

### 3. Why It Was Added: Missing Stop Sequences in `BrainChain`

Why was `repeat_penalty: 1.05` introduced originally?
In [`src/arc_agent/chains/brain.py`](file:///e:/code-in-progress/arcgame/src/arc_agent/chains/brain.py#L147-L152):
```python
return self._invoke(
    prompt,
    temperature=0.0,
    max_tokens=min(128, self.max_tokens),
    image_obj=current_state.get_pil_image(),
)
```
- **No `stop` sequences are passed.**
- The model only needs ~15 tokens to output `Plan: ...\nACTION=ACTION1`.
- With `max_tokens=128` and no stop sequence, the model was previously entering repetition loops (`ACTION=ACTION1 ACTION=ACTION1 ...`).
- Setting `repeat_penalty: 1.05` was applied as a band-aid to stop that loop, which in turn caused the non-English language drift.

---

### 4. Why This Triggers `[CRITICAL LLM FAILURE]` (Halting at Step 12 & Step 41)

When the foreign characters are emitted, the sanitizer in [`gemma_transformers.py`](file:///e:/code-in-progress/arcgame/src/arc_agent/models/gemma_transformers.py#L126-L141) triggers:
```python
clean = re.sub(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff\u0900-\u097f]+", "", clean)
```
1. Stripping the foreign characters leaves a fragmented sentence.
2. Because `len(clean) >= 3` (there are still English words like `Plan:` or `State:`), line 135:
   ```python
   if len(clean) < _MIN_RESPONSE_CHARS:  # NOT triggered!
       rescued = _rescue_action_from_corrupted_text(...)
   ```
   **is bypassed**, so the rescue logic never executes.
3. The broken string is passed to `ARCActionMapper.parse()`, which fails to find a valid `ACTION=`.
4. `decide_action` executes a format retry with `[FORMAT NOTICE]`, which increases the prompt length, worsening the repetition penalty.
5. `consecutive_parse_failures` increments. Once it reaches 6, the safety halt terminates the level attempt prematurely.

---

### Recommended Permanent Fix

1. **Disable `repetition_penalty` (set to `1.0`)**:
   - In [`configs/default.yaml`](file:///e:/code-in-progress/arcgame/configs/default.yaml): set `repeat_penalty: 1.0`.
   - In [`src/arc_agent/config.py`](file:///e:/code-in-progress/arcgame/src/arc_agent/config.py#L29): default `repeat_penalty: 1.0`.
   - In [`src/arc_agent/models/gemma_transformers.py`](file:///e:/code-in-progress/arcgame/src/arc_agent/models/gemma_transformers.py#L200): default `repeat_penalty: 1.0`.
   - Eliminating the penalty stops the artificial suppression of English tokens.

2. **Add Proper Stop Sequences to `BrainChain`**:
   - Pass `stop=["\n\n", "\nPlan:", "\nState Metadata:"]` or terminate immediately after the `ACTION=...` line is produced. This prevents repetition loops naturally without distorting token logits.

3. **Improve the Sanitizer Rescue**:
   - In [`_sanitize_llm_text`](file:///e:/code-in-progress/arcgame/src/arc_agent/models/gemma_transformers.py#L135), always attempt `_rescue_action_from_corrupted_text(text)` if foreign characters were stripped and `ACTION=` is absent or malformed in `clean`.
   - Log `raw_decoded` whenever foreign characters are detected (`ratio > 0.05`) so you can see the exact unstripped output.

Would you like me to prepare an implementation plan and apply these fixes?
