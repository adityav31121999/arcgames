import json
from pathlib import Path

sample_path = Path("notebooks/sample_run_single_game.ipynb")
sub_path = Path("notebooks/submission_run.ipynb")

for p in [sample_path, sub_path]:
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            new_source = []
            for line in cell["source"]:
                if 'config.agent.memory_root = "/kaggle/working/agent_memory"' in line:
                    new_source.append('config.agent.memory_root = "/tmp/agent_memory"\n')
                elif 'scratch_path = Path(f"/kaggle/working/agent_memory/{TARGET_GAME}/scratchpad.md")' in line:
                    new_source.append('scratch_path = Path(f"{config.agent.memory_root}/{TARGET_GAME}/scratchpad.md")\n')
                elif 'actions_path = Path(f"/kaggle/working/agent_memory/{TARGET_GAME}/level_1/actions.md")' in line:
                    new_source.append('actions_path = Path(f"{config.agent.memory_root}/{TARGET_GAME}/level_1/actions.md")\n')
                else:
                    new_source.append(line)
            cell["source"] = new_source

    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

print("Updated memory_root to /tmp/agent_memory in both notebooks.")
