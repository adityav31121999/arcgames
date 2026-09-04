import json
from pathlib import Path

notebook_paths = [
    Path("notebooks/sample_run_single_game.ipynb"),
    Path("notebooks/submission_run.ipynb"),
]

target_code_old = """# 2. Register 'gemma4' architecture with Hugging Face AutoConfig to prevent KeyError
try:
    import transformers
    from transformers import AutoConfig, GemmaConfig
    gemma_base_cfg = getattr(transformers, "Gemma2Config", GemmaConfig)
    AutoConfig.register("gemma4", gemma_base_cfg)
    print("✅ Mapped 'gemma4' model_type to Gemma configuration in AutoConfig!")
except Exception as e:
    print(f"ℹ️ AutoConfig registration note: {e}")"""

replacement_code = """# 2. Register Gemma 4 / Gemma 4 MoE architecture with Hugging Face AutoConfig
try:
    import transformers
    from transformers import AutoConfig, GemmaConfig
    gemma4_cfg = getattr(transformers, "Gemma4Config", None)
    if gemma4_cfg is not None:
        for m_type in ["gemma4", "gemma4_moe"]:
            try:
                AutoConfig.register(m_type, gemma4_cfg, exist_ok=True)
            except TypeError:
                AutoConfig.register(m_type, gemma4_cfg)
        print("✅ Registered native Gemma4Config with AutoConfig!")
    else:
        gemma_base_cfg = getattr(transformers, "Gemma2Config", GemmaConfig)
        AutoConfig.register("gemma4", gemma_base_cfg)
except Exception as e:
    print(f"ℹ️ AutoConfig registration note: {e}")"""

for nb_path in notebook_paths:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    updated = False
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if 'AutoConfig.register("gemma4", gemma_base_cfg)' in src and 'gemma4_cfg = getattr(transformers, "Gemma4Config", None)' not in src:
                # Replace the snippet inside this cell
                old_lines = [
                    '# 2. Register \'gemma4\' architecture with Hugging Face AutoConfig to prevent KeyError\n',
                    'try:\n',
                    '    import transformers\n',
                    '    from transformers import AutoConfig, GemmaConfig\n',
                    '    gemma_base_cfg = getattr(transformers, "Gemma2Config", GemmaConfig)\n',
                    '    AutoConfig.register("gemma4", gemma_base_cfg)\n',
                    '    print("✅ Mapped \'gemma4\' model_type to Gemma configuration in AutoConfig!")\n',
                    'except Exception as e:\n',
                    '    print(f"ℹ️ AutoConfig registration note: {e}")\n'
                ]
                new_lines = [line + '\n' for line in replacement_code.split('\n')]
                
                # Check if old_lines matches part of source
                joined_src = "".join(cell["source"])
                joined_old = "".join(old_lines)
                if joined_old in joined_src:
                    joined_new = joined_src.replace(joined_old, "".join(new_lines))
                    cell["source"] = [l + "\n" for l in joined_new.split("\n")][:-1]
                    updated = True
                else:
                    # Alternative replacement by find
                    idx_start = -1
                    for idx, line in enumerate(cell["source"]):
                        if "# 2. Register 'gemma4' architecture" in line:
                            idx_start = idx
                            break
                    if idx_start != -1:
                        # find end of except block
                        idx_end = idx_start + 1
                        while idx_end < len(cell["source"]):
                            if 'print(f"ℹ️ AutoConfig registration note:' in cell["source"][idx_end]:
                                idx_end += 1
                                break
                            idx_end += 1
                        cell["source"] = cell["source"][:idx_start] + [l + "\n" for l in replacement_code.split("\n")] + cell["source"][idx_end:]
                        updated = True

    if updated:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully updated cell 1 in {nb_path.name}")
    else:
        print(f"No update needed in {nb_path.name}")
