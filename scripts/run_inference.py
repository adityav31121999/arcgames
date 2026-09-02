#!/usr/bin/env python3
"""CLI Entry point for ARC-AGI-3 LangChain Agent inference."""

import argparse
from pathlib import Path
import os
import sys

# Ensure src is in sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


from arc_agent.agent.arc_langchain_agent import ARCLangChainAgent
from arc_agent.agent.runner import ARCRunner
from arc_agent.chains.brain import BrainChain
from arc_agent.chains.debugger import DebuggerChain
from arc_agent.chains.eye import EyeChain
from arc_agent.chains.reviewer import ReviewerChain
from arc_agent.config import AppConfig
from arc_agent.core.resolver import GameStateResolver
from arc_agent.models.factory import ModelFactory
from arc_agent.utils.locator import find_arc_agi_wheels


def setup_offline_arcade(config: AppConfig):
    """Initializes arcade instance with offline or competition gateway configuration."""
    try:
        import arc_agi
    except ImportError:
        # Check if wheel can be installed offline
        whls = find_arc_agi_wheels(search_root="/kaggle/input")
        if whls:
            import subprocess
            wheel_dir = str(whls[0].parent)
            print(f"📦 Installing arc_agi wheel from: {whls[0]}")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
                "--find-links", wheel_dir, "arc-agi"
            ])
            import arc_agi
        else:
            raise ImportError(
                "Could not import 'arc_agi'. Please install competition wheel or run with a mock environment."
            )

    is_competition_rerun = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
    if is_competition_rerun or config.environment.mode == "competition":
        base_url = os.getenv("ARC_BASE_URL", config.environment.base_url)
        print(f"🌐 [ARCADE] Online Tournament Mode connecting to: {base_url}")
        return arc_agi.Arcade(operation_mode=arc_agi.OperationMode.COMPETITION, arc_base_url=base_url)
    else:
        env_dir = os.getenv("ENVIRONMENTS_DIR", config.environment.environments_dir)
        print(f"📂 [ARCADE] Offline Validation Mode loading from: {env_dir}")
        return arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI-3 LangChain Inference Agent")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--game", type=str, default=None, help="Specific game_id to run (e.g. s5i5)")
    parser.add_argument("--mock", action="store_true", help="Run with Mock LLM for rapid testing without weights")
    parser.add_argument("--max-levels", type=int, default=10, help="Max levels per game")
    parser.add_argument("--device", type=str, default=None, help="Override target device (e.g. cuda:0 or cpu)")
    args = parser.parse_args()

    config = AppConfig.from_yaml(args.config)
    config.apply_env_overrides()
    if args.device:
        config.model.device = args.device

    print("=" * 60)
    print("🤖 ARC-AGI-3 LangChain Agent Inference Runner")
    print(f"Model ID: {config.model.model_id} | Device: {config.model.device} | Precision: {config.model.torch_dtype}")
    print("=" * 60)

    # 1. Initialize LangChain model
    llm = ModelFactory.create_model(config.model, use_mock=args.mock)

    # 2. Build LangChain chains
    eye_chain = EyeChain(llm, max_tokens=config.model.max_new_tokens_eye)
    debugger_chain = DebuggerChain(llm, max_tokens=config.model.max_new_tokens_debug)
    brain_chain = BrainChain(llm, max_tokens=config.model.max_new_tokens_brain)
    reviewer_chain = ReviewerChain(llm, max_tokens=config.model.max_new_tokens_review)

    # 3. Game state resolver
    try:
        from arcengine import GameState
        resolver = GameStateResolver(GameState)
    except ImportError:
        resolver = GameStateResolver(None)

    # 4. Instantiate Agent & Runner
    agent = ARCLangChainAgent(
        eye_chain=eye_chain,
        debugger_chain=debugger_chain,
        brain_chain=brain_chain,
        reviewer_chain=reviewer_chain,
        resolver=resolver,
        stuck_threshold=config.agent.stuck_threshold,
        memory_root=config.agent.memory_root,
        vision_cache_dir=config.agent.vision_cache_dir,
    )
    runner = ARCRunner(agent=agent, time_budget_hours=config.agent.time_budget_hours)

    # 5. Load Arcade Environment
    try:
        arcade = setup_offline_arcade(config)
        available_envs = arcade.available_environments
        game_list = [env_info.game_id for env_info in available_envs]
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize official arcade ({e}).")
        if not args.game:
            print("❌ No official arcade environment found and no specific game provided.")
            return
        game_list = [args.game]
        arcade = None

    if args.game:
        target_games = [args.game]
    else:
        target_games = game_list

    print(f"🎮 Games in Queue ({len(target_games)}): {target_games}")

    results = []
    for game_id in target_games:
        print(f"\n{'='*50}\n🎬 Starting Game: {game_id}\n{'='*50}")
        if arcade is not None:
            env = arcade.make(game_id)
            if env is None:
                print(f"❌ Failed to instantiate environment for {game_id}")
                continue
            try:
                final_obs = runner.play_game(game_id=game_id, env=env, max_levels=args.max_levels)
                state_name = getattr(getattr(final_obs, "state", None), "name", str(getattr(final_obs, "state", "UNKNOWN")))
                completed = getattr(final_obs, "levels_completed", 0)
                print(f"✅ Finished {game_id} | Final State: {state_name} | Levels Completed: {completed}")
                results.append([f"{game_id}_0", game_id, True, completed])
            except Exception as e:
                print(f"❌ Error during game {game_id}: {e}")
        else:
            print(f"ℹ️ Dry run completed for {game_id} with mock environment.")
            results.append([f"{game_id}_0", game_id, True, 1])

    # 6. Save competition submission target
    if results:
        submission_path = Path(config.environment.submission_parquet)
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import pandas as pd
            df = pd.DataFrame(results, columns=["row_id", "game_id", "end_of_game", "score"])
            df.to_parquet(submission_path, index=False)
            print(f"\n💾 Saved submission artifact to: {submission_path}")
        except ImportError:
            print(f"\n⚠️ Pandas not found, results recorded: {results}")



if __name__ == "__main__":
    main()
