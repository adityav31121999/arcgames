"""Verify the paste-in cell changes stages without reloading the shared model."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arc_agent.config import AppConfig
from arc_agent.chains.brain import BrainChain
from arc_agent.chains.debugger import DebuggerChain
from arc_agent.chains.eye import EyeChain
from arc_agent.core.state import ARCState, compute_transition
from arc_agent.memory.knowledge import KnowledgeCache
from arc_agent.models.gemma_transformers import MockChatModel


def patch_namespace():
    path = Path(__file__).resolve().parents[1] / "notebooks/kaggle_fast_inference.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    module.body = module.body[:-1]  # Load definitions without applying to notebook globals.
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def test_stage_budgets_keep_shared_model_and_restore_routine_eye(tmp_path, monkeypatch):
    calls = []
    original = MockChatModel._generate

    def record(self, *args, **kwargs):
        calls.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MockChatModel, "_generate", record)
    model = MockChatModel()
    agent = SimpleNamespace(eye=EyeChain(model), brain=BrainChain(model),
                            debugger=DebuggerChain(model))
    patch = patch_namespace()["apply_fast_inference"]
    config = AppConfig()
    patch(agent, config)
    cache = KnowledgeCache(memory_root=tmp_path)
    obs = SimpleNamespace(grid=np.zeros((8, 8), dtype=int), state="PLAYING", levels_completed=0)
    state = ARCState.create("game", 1, 0, obs)
    agent.eye.assume("game", 1, state, cache)
    assert calls[-1]["enable_thinking"] is True
    assert calls[-1]["max_tokens"] == 1536
    assert agent.eye.model is agent.brain.model is agent.debugger.model is model
    assert agent.eye.max_tokens == 384

    # Applying the cell again must preserve completed assumptions and avoid duplicate logging.
    patch(agent, config)
    assert len(model.callbacks) == 1
    agent.eye.assume("game", 1, state, cache)
    assert calls[-1].get("enable_thinking", False) is False
    assert calls[-1]["max_tokens"] == 384
    agent.eye.compare_assume("game", 2, state, cache)
    agent.eye.analyse_visual("game", state, compute_transition(state, state), "No change")
    assert calls[-1].get("enable_thinking", False) is False
    assert calls[-1]["max_tokens"] == 384
    assert agent.brain.enable_thinking and agent.brain.max_tokens == 1536
    assert not agent.debugger.enable_thinking and agent.debugger.max_tokens == 384


def test_initial_assumption_exception_restores_eye_model(monkeypatch):
    model = MockChatModel()
    eye = patch_namespace()["BudgetedEyeChain"](model)

    def fail(*args, **kwargs):
        raise RuntimeError("failed assumption")

    monkeypatch.setattr(EyeChain, "assume", fail)
    with pytest.raises(RuntimeError, match="failed assumption"):
        eye.assume("game")
    assert eye.model is model and eye.max_tokens == 384
    assert not eye.assumed_games


def test_complete_notebook_initializes_stage_budgets_without_patch(tmp_path):
    from arc_agent.agent.arc_langchain_agent import ARCLangChainAgent
    from arc_agent.agent.runner import ARCRunner
    from arc_agent.core.resolver import GameStateResolver

    path = Path(__file__).resolve().parents[1] / "notebooks/assumeAndPlay.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for source in code:
        ast.parse(source)
    assert not any("apply_fast_inference" in source for source in code)
    definitions = next(source for source in code if "class BudgetedEyeChain" in source)
    initialization = next(source for source in code if "STAGE_BUDGETS =" in source)
    assert code.index(definitions) < code.index(initialization)
    # Redirect notebook memory locations while exercising its actual initialization.
    initialization = initialization.replace('"/tmp/agent_memory"', repr(str(tmp_path / "memory")))
    initialization = initialization.replace('"/tmp/agent_vision"', repr(str(tmp_path / "vision")))
    model = MockChatModel()
    namespace = dict(Path=Path, AppConfig=AppConfig, source_dir=tmp_path / "src",
                     ModelFactory=SimpleNamespace(create_model=lambda *a, **kw: model),
                     USE_MOCK=True, BrainChain=BrainChain, DebuggerChain=DebuggerChain,
                     ARCLangChainAgent=ARCLangChainAgent, ARCRunner=ARCRunner,
                     GameStateResolver=GameStateResolver,
                     torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False,
                                                                is_bf16_supported=lambda: False)))
    exec(compile(definitions, str(path), "exec"), namespace)
    exec(compile(initialization, str(path), "exec"), namespace)
    agent = namespace["agent"]
    assert agent.eye is namespace["eye_chain"]
    assert agent.eye.assumption_max_tokens == 1536 and agent.eye.max_tokens == 384
    assert agent.brain.enable_thinking and agent.brain.max_tokens == 1536
    assert not agent.debugger.enable_thinking and agent.debugger.max_tokens == 384
    assert namespace["runner"].agent is agent
    assert len(model.callbacks) == 1
