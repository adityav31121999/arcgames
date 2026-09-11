# ARC-AGI-3 LangChain Inference Agent

## Kaggle vLLM setup

`notebooks/assumeAndPlay.ipynb` now runs Eye and Brain through vLLM's in-process
`LLM.chat()` API. It uses the local checkpoint and its chat template, passes board
images as PNG data URLs, and enables native thinking for Brain decisions and failed-attempt reviews. Only the final answer is parsed into actions and memory.
Transformers supplies the CPU processor for context budgeting; it does not load
model weights. The default YAML configurations select `backend: vllm`.

Attach the updated source bundle, model files, and a **Linux vLLM wheelhouse with
all dependencies** matching Kaggle's Python and GPU runtime. Build/download that
wheelhouse on a compatible Linux environment using `requirements-vllm.txt`; do not
use Windows wheels. The source ZIP does not contain these third-party wheels.
Start a fresh Kaggle session and run the notebook from the top. The first cell
installs the dependencies together, offline, before importing the GPU runtime.

Startup runs the text and red-image checks before gameplay and writes
`/kaggle/working/model_health.json`, including the vLLM version. Defaults are one GPU,
32,768 context tokens, two images per request, 85% GPU memory utilization, and eager
execution. Adjust these in the initialization cell if required by the runtime.
There is no automatic fallback to Transformers when vLLM loading fails.

References: [vLLM chat](https://docs.vllm.ai/en/latest/models/generative_models/),
[multimodal inputs](https://docs.vllm.ai/en/latest/features/multimodal_inputs/).

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
interval and hands off for evaluation before Brain replans. The single-game notebook disables speculative execution and fast evaluation to inspect every move. Set
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
