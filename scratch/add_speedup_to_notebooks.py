import json
from pathlib import Path

sample_path = Path("notebooks/sample_run_single_game.ipynb")
sub_path = Path("notebooks/submission_run.ipynb")

speedup_lines = [
    "\n",
    "# ⚡ GPU Performance Acceleration on RTX PRO 6000\n",
    "if torch.cuda.is_available():\n",
    "    torch.backends.cuda.matmul.allow_tf32 = True\n",
    "    torch.backends.cudnn.allow_tf32 = True\n",
    "\n",
    "# Fast token limits: Brain actions only need ~10 tokens (ACTION=...)\n",
    "config.model.max_new_tokens_brain = 32\n",
    "config.model.max_new_tokens_debug = 96\n",
    "config.model.max_new_tokens_eye = 256\n",
    "config.model.max_new_tokens_review = 256\n",
]

post_llm_speedup_lines = [
    "\n",
    "# Force use_cache=True for fast O(1) token generation instead of O(N^2) KV recomputation\n",
    "if hasattr(llm, \"_generate\"):\n",
    "    _orig_llm_gen = llm._generate\n",
    "    def _fast_llm_gen(messages, stop=None, run_manager=None, **kwargs):\n",
    "        kwargs[\"use_cache\"] = True\n",
    "        return _orig_llm_gen(messages, stop=stop, run_manager=run_manager, **kwargs)\n",
    "    llm._generate = _fast_llm_gen\n",
    "\n",
    "# Enforce greedy decoding and early stopping on newline for Brain\n",
    "if hasattr(brain_chain, \"_invoke\"):\n",
    "    _orig_b_inv = brain_chain._invoke\n",
    "    def _fast_b_inv(prompt, temperature=0.0, max_tokens=32, stop=None):\n",
    "        return _orig_b_inv(prompt, temperature=0.0, max_tokens=min(32, max_tokens), stop=stop or [\"\\n\"])\n",
    "    brain_chain._invoke = _fast_b_inv\n",
]

for p in [sample_path, sub_path]:
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "ModelFactory.create_model(" in src and "allow_tf32" not in src:
                # Find where config.model.attn_implementation is set
                idx_cfg = -1
                for idx, line in enumerate(cell["source"]):
                    if "config.model.attn_implementation" in line:
                        idx_cfg = idx + 1
                        break
                if idx_cfg != -1:
                    cell["source"] = cell["source"][:idx_cfg] + speedup_lines + cell["source"][idx_cfg:]

                # Find where runner is initialized
                idx_runner = -1
                for idx, line in enumerate(cell["source"]):
                    if "runner = ARCRunner(" in line:
                        idx_runner = idx
                        break
                if idx_runner != -1:
                    cell["source"] = cell["source"][:idx_runner] + post_llm_speedup_lines + cell["source"][idx_runner:]

                print(f"Added speed optimizations to Cell 3 of {p.name}")

    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

print("Finished speedup updates.")
