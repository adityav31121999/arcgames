import json
from pathlib import Path

sample_path = Path("notebooks/sample_run_single_game.ipynb")
with open(sample_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if "env = arcade.make(TARGET_GAME)" in src and "Target Game" not in src:
            new_source = []
            for line in cell["source"]:
                if "final_obs = runner.play_game(" in line:
                    new_source.append("    act_space = getattr(env, 'action_space', None)\n")
                    new_source.append("    print(f\"🕹️ Target Game '{TARGET_GAME}' Action Space: {act_space}\")\n")
                new_source.append(line)
            cell["source"] = new_source
            print("Added action space inspection to Cell 5.")

with open(sample_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
