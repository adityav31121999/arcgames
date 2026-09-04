import json
from pathlib import Path

sample_path = Path("notebooks/sample_run_single_game.ipynb")
sub_path = Path("notebooks/submission_run.ipynb")

for p in [sample_path, sub_path]:
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Update Cell 2 (imports) to inject render_live into builtins and runner
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "from arc_agent.agent.runner import ARCRunner" in src and "builtins.render_live" not in src:
                # Append builtins injection
                cell["source"].extend([
                    "\n",
                    "# Ensure render_live is globally available to runner.py\n",
                    "import builtins\n",
                    "from arc_agent.utils.display import render_live\n",
                    "builtins.render_live = render_live\n",
                    "import arc_agent.agent.runner as runner_mod\n",
                    "runner_mod.render_live = render_live\n"
                ])
                print(f"Injected render_live into Cell 2 of {p.name}")

    # 2. Update Cell 5 / play_game cell to ensure render_live is set right before execution
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "runner.play_game(" in src and "builtins.render_live" not in src:
                # Prepend builtins injection
                cell["source"] = [
                    "import builtins\n",
                    "from arc_agent.utils.display import render_live\n",
                    "builtins.render_live = render_live\n",
                    "import arc_agent.agent.runner as runner_mod\n",
                    "runner_mod.render_live = render_live\n",
                    "\n"
                ] + cell["source"]
                print(f"Injected render_live into play_game cell of {p.name}")

    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

print("Finished notebook updates.")
