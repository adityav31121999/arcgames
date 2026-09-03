# ARC-AGI-3 LangChain Inference Agent

An agentic reasoning framework for **ARC-AGI-3** (ARC Prize 2026), powered by **LangChain** and **Hugging Face `transformers`**, specifically optimized for **NVIDIA RTX PRO 6000 (96 GB VRAM)** running **[`nvidia/Gemma-4-26B-A4B-NVFP4`](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4)** (and Gemma 4 MoE architectures).

---

## 🚀 Key Highlights

- **Hugging Face Transformers Backend**: Native integration with `AutoProcessor` and `AutoModelForImageTextToText` / `AutoModelForCausalLM` with `torch.bfloat16`, SDPA/FlashAttention, and `device="cuda:0"`.
- **Hardware Optimization (RTX PRO 6000 96GB VRAM)**: With 96GB of high-bandwidth VRAM, the entire Gemma 4 MoE 26B-A4B model (both vision tower and all expert routing layers) fits directly into GPU memory with zero CPU bottlenecks, enabling full context handling and rapid KV-cache throughput.
- **LangChain Modular Architecture**: Structured with LangChain Expression Language (LCEL) Runnables and custom `BaseChatModel` wrappers:
  - 👁️ **Perception (Eye Chain)**: Multimodal spatial understanding of S0 grid layout, goal anchors, and cross-level state deltas.
  - 🔍 **Verification (Debugger Chain)**: Fast NumPy ground-truth pixel difference bounding-box analysis and transition collision/divergence verification.
  - 🧠 **Policy (Brain Chain)**: High-level reasoning, legal action filtering (excluding visited loops and oscillating paths), sprite bounding-box guidance, and speculative one-shot macro planning.
  - 📝 **Meta-Reflection (Reviewer Chain)**: Post-failure iteration review that updates and consolidates verified mechanics into persistent markdown memory (`scratchpad.md`).
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
│       │   ├── debugger.py     # Transition validation & rule divergence checks
│       │   ├── brain.py        # Next-action selection & speculative macro planning
│       │   └── reviewer.py     # Post-failure reflection & rule consolidation
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
    └── test_chains.py          # Tests for LangChain perception, debug, brain & review chains
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
  max_context_length: 8192
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