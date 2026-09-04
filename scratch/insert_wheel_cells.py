import json
from pathlib import Path

sample_path = Path("notebooks/sample_run_single_game.ipynb")
sub_path = Path("notebooks/submission_run.ipynb")

cell_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 0. Offline Wheels & Environment Upgrade (Transformers >= 5.10.1 for Gemma 4)\n",
        "\n",
        "Installs updated Hugging Face `transformers` and dependencies from the offline wheel dataset at `/kaggle/input/.../hf_transformers_wheels` before importing `transformers`."
    ]
}

cell_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import sys\n",
        "import subprocess\n",
        "from pathlib import Path\n",
        "\n",
        "# 1. Locate offline Hugging Face / LangChain wheels\n",
        "wheel_candidates = [\n",
        "    Path(\"/kaggle/input/datasets/spiritofvishwakarma/llama-cuda-wheel/hf_transformers_wheels\"),\n",
        "    Path(\"/kaggle/input/llama-cuda-wheel/hf_transformers_wheels\"),\n",
        "    Path(\"/kaggle/input/datasets/spiritofvishwakarma/hf-transformers-langchain-wheels\"),\n",
        "    Path(\"/kaggle/input/hf-transformers-langchain-wheels\"),\n",
        "]\n",
        "\n",
        "wheel_dir = None\n",
        "for cand in wheel_candidates:\n",
        "    if cand.exists() and any(cand.glob(\"*.whl\")):\n",
        "        wheel_dir = cand\n",
        "        break\n",
        "\n",
        "# Fallback: search all of /kaggle/input for any hf_transformers_wheels folder\n",
        "if not wheel_dir and Path(\"/kaggle/input\").exists():\n",
        "    for p in Path(\"/kaggle/input\").rglob(\"hf_transformers_wheels\"):\n",
        "        if p.is_dir() and any(p.glob(\"*.whl\")):\n",
        "            wheel_dir = p\n",
        "            break\n",
        "\n",
        "# 2. Install updated transformers wheel offline\n",
        "if wheel_dir:\n",
        "    print(f\"📦 Installing updated Transformers & dependencies offline from: {wheel_dir}\")\n",
        "    try:\n",
        "        subprocess.check_call([\n",
        "            sys.executable, \"-m\", \"pip\", \"install\",\n",
        "            \"--no-index\",\n",
        "            \"--find-links\", str(wheel_dir),\n",
        "            \"--upgrade\",\n",
        "            \"transformers\"\n",
        "        ])\n",
        "        print(\"✅ Successfully upgraded transformers from offline wheelhouse!\")\n",
        "    except Exception as e:\n",
        "        print(f\"⚠️ Note during transformers installation: {e}\")\n",
        "else:\n",
        "    print(\"ℹ️ No dedicated hf_transformers_wheels dataset found; using pre-installed environment.\")\n",
        "\n",
        "# Purge cached in-memory modules if an older transformers version was already loaded\n",
        "for mod_name in list(sys.modules.keys()):\n",
        "    if mod_name == \"transformers\" or mod_name.startswith(\"transformers.\"):\n",
        "        del sys.modules[mod_name]\n",
        "\n",
        "# Hotfix: ensure transformers.utils has hf_api to prevent version mismatch crash\n",
        "try:\n",
        "    import transformers\n",
        "    import transformers.utils\n",
        "    if not hasattr(transformers.utils, \"hf_api\"):\n",
        "        try:\n",
        "            from huggingface_hub import HfApi\n",
        "            transformers.utils.hf_api = HfApi\n",
        "        except Exception:\n",
        "            pass\n",
        "    print(f\"🚀 Active Transformers version: {transformers.__version__}\")\n",
        "except Exception as e:\n",
        "    print(f\"Transformers import check: {e}\")"
    ]
}

def update_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Find index of cell 0 (markdown or code) and replace
    new_cells = []
    skip_next = False
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if "Offline Wheels & Environment Upgrade" in src:
            skip_next = True
            continue
        if skip_next:
            skip_next = False
            continue
        new_cells.append(cell)

    # Insert updated cell 0
    new_cells.insert(1, cell_markdown)
    new_cells.insert(2, cell_code)
    nb["cells"] = new_cells

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Updated {path.name}")

update_notebook(sample_path)
update_notebook(sub_path)
