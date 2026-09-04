import json
from pathlib import Path

sample_path = Path("notebooks/sample_run_single_game.ipynb")
sub_path = Path("notebooks/submission_run.ipynb")

safetensors_patch_lines = [
    "# Hotfix for safetensors.safe_open unexpected keyword argument 'backend'\n",
    "try:\n",
    "    import safetensors\n",
    "    _orig_safe_open = safetensors.safe_open\n",
    "    def _compat_safe_open(*args, **kwargs):\n",
    "        try:\n",
    "            return _orig_safe_open(*args, **kwargs)\n",
    "        except TypeError as te:\n",
    "            if \"backend\" in str(te):\n",
    "                kwargs.pop(\"backend\", None)\n",
    "                return _orig_safe_open(*args, **kwargs)\n",
    "            raise te\n",
    "    safetensors.safe_open = _compat_safe_open\n",
    "    if hasattr(safetensors, \"torch\"):\n",
    "        safetensors.torch.safe_open = _compat_safe_open\n",
    "    import sys\n",
    "    for mod in list(sys.modules.values()):\n",
    "        if hasattr(mod, \"safe_open\") and getattr(mod, \"safe_open\") is _orig_safe_open:\n",
    "            setattr(mod, \"safe_open\", _compat_safe_open)\n",
    "    print(\"✅ Applied safetensors safe_open compatibility patch!\")\n",
    "except Exception as e:\n",
    "    pass\n",
]

for p in [sample_path, sub_path]:
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Add to Cell 1 (under Environment detection)
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "GLOBAL HOTFIX: Prevent pad_token_id list crash" in src and "_compat_safe_open" not in src:
                # Insert before the last print or before os.environ
                idx_insert = -1
                for idx, line in enumerate(cell["source"]):
                    if 'print("✅ Applied global runtime hotfixes' in line:
                        idx_insert = idx
                        break
                if idx_insert != -1:
                    cell["source"] = cell["source"][:idx_insert] + safetensors_patch_lines + cell["source"][idx_insert:]
                    print(f"Added safetensors patch to Cell 1 of {p.name}")

    # 2. Add to Cell 3 (right before ModelFactory.create_model)
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "ModelFactory.create_model(" in src and "_compat_safe_open" not in src:
                idx_insert = -1
                for idx, line in enumerate(cell["source"]):
                    if "llm = ModelFactory.create_model(" in line:
                        idx_insert = idx
                        break
                if idx_insert != -1:
                    cell["source"] = cell["source"][:idx_insert] + safetensors_patch_lines + cell["source"][idx_insert:]
                    print(f"Added safetensors patch to Cell 3 of {p.name}")

    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

print("Notebooks updated successfully.")
