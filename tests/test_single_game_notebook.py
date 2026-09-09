"""Execute notebook logic with local files and dependency/model doubles."""

import ast
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from arc_agent.config import ModelConfig
from arc_agent.models.factory import ModelFactory


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/sample_run_single_game.ipynb"


def cell(index):
    return "".join(json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"][index]["source"])


def functions(index):
    module = ast.parse(cell(index))
    module.body = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    namespace = {"Path": Path, "json": json}
    exec(compile(module, str(NOTEBOOK), "exec"), namespace)
    return namespace


def test_game_selection_does_not_silently_switch_or_choose_ambiguous_version():
    resolve = functions(10)["resolve_game_id"]
    assert resolve("m0r0", ["m0r0-123"]) == "m0r0-123"
    assert resolve("M0R0-123", ["m0r0-123", "m0r0-456"]) == "m0r0-123"
    for target, available in [("m0r0", ["s5i5-123"]), ("m0r0", ["m0r0-123", "m0r0-456"]), ("", ["m0r0-123"])]:
        with pytest.raises(ValueError):
            resolve(target, available)


def test_checkpoint_discovery_rejects_missing_shards_and_ambiguous_copies(tmp_path):
    select = functions(8)["select_checkpoint"]
    directory = tmp_path / "Gemma-4-26B-NVFP4"
    directory.mkdir()
    metadata = {"model_type": "gemma4", "vision_config": {"hidden_size": 768}}
    (directory / "config.json").write_text(json.dumps(metadata))
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "one.safetensors", "b": "two.safetensors"}}))
    (directory / "one.safetensors").touch()
    with pytest.raises(RuntimeError):
        select(tmp_path)
    (directory / "two.safetensors").touch()
    assert select(tmp_path) == directory.resolve()
    duplicate = tmp_path / "copy-26-NVFP4"
    duplicate.mkdir()
    (duplicate / "config.json").write_text(json.dumps(metadata))
    (duplicate / "model.safetensors").touch()
    with pytest.raises(RuntimeError):
        select(tmp_path)
    assert select(tmp_path, directory) == directory.resolve()


def test_source_discovery_requires_explicit_choice_for_multiple_bundles(tmp_path):
    locate = functions(6)["locate_and_mount_source_package"]
    for name in ("old", "new"):
        package = tmp_path / name / "src/arc_agent"
        package.mkdir(parents=True)
        (package / "__init__.py").touch()
    with pytest.raises(RuntimeError):
        locate(tmp_path)
    chosen = tmp_path / "new/src"
    assert locate(tmp_path, chosen) == chosen.resolve()


@pytest.fixture
def native_runtime(monkeypatch, tmp_path):
    generation = SimpleNamespace(eos_token_id=[1, 106])
    model = SimpleNamespace(generation_config=generation, eval=Mock())
    processor = SimpleNamespace(image_processor=object(), chat_template="native")
    transformers = SimpleNamespace(
        Gemma4ForConditionalGeneration=SimpleNamespace(from_pretrained=Mock(return_value=model)),
        AutoProcessor=SimpleNamespace(from_pretrained=Mock(return_value=processor)),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr("arc_agent.models.factory.silence_hf_warnings", lambda: None)
    monkeypatch.delenv("USE_MOCK_MODEL", raising=False)
    (tmp_path / "config.json").write_text('{"model_type": "gemma4"}')
    return transformers, model, ModelConfig(model_id=str(tmp_path))


def test_local_gemma4_load_preserves_eos_and_requested_attention(native_runtime):
    transformers, model, config = native_runtime
    wrapper = ModelFactory.create_model(config)
    assert wrapper.model is model
    assert model.generation_config.eos_token_id == [1, 106]
    kwargs = transformers.Gemma4ForConditionalGeneration.from_pretrained.call_args.kwargs
    assert kwargs["attn_implementation"] == "sdpa"
    assert kwargs["local_files_only"] is True
    model.eval.assert_called_once()


def test_native_loader_preserves_quantization_failure(native_runtime):
    transformers, _, config = native_runtime
    error = RuntimeError("NVFP4 kernel unavailable")
    transformers.Gemma4ForConditionalGeneration.from_pretrained.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        ModelFactory.create_model(config)
    assert caught.value is error
    transformers.Gemma4ForConditionalGeneration.from_pretrained.assert_called_once()


def test_native_loader_rejects_tokenizer_only_processor(native_runtime):
    transformers, _, config = native_runtime
    transformers.AutoProcessor.from_pretrained.return_value.image_processor = None
    with pytest.raises(RuntimeError, match="multimodal processor"):
        ModelFactory.create_model(config)
    transformers.Gemma4ForConditionalGeneration.from_pretrained.assert_not_called()


def test_notebook_execution_cell_respects_explicit_move_budget(tmp_path, monkeypatch):
    from tests.test_agent_reliability import Environment
    from tests import test_runner_budget as components
    from arc_agent.config import AppConfig

    monkeypatch.setattr("arc_agent.agent.runner.render_live", lambda *a, **kw: None)
    monkeypatch.setattr("arc_agent.agent.arc_langchain_agent.render_live", lambda *a, **kw: None)
    env = Environment(complete=True)
    env.environment_info = SimpleNamespace(baseline_actions=[10, 10], win_levels=2)
    namespace = {name: getattr(components, name) for name in (
        "ARCRunner", "ARCLangChainAgent", "EyeChain", "BrainChain",
        "GameStateResolver",
    )}
    namespace.update({
        "Path": Path, "json": json, "time": time, "USE_MOCK": True,
        "AppConfig": AppConfig, "ModelFactory": ModelFactory,
        "source_dir": NOTEBOOK.parents[1] / "src", "WORKING_DIR": tmp_path,
        "TARGET_GAME": "m0r0-test", "arcade": SimpleNamespace(make=lambda game: env),
    })
    exec(compile(cell(8), str(NOTEBOOK), "exec"), namespace)
    namespace["config"].agent.max_total_actions_per_game = 1
    namespace["runner"].speculative_plan_max_steps = 0
    exec(compile(cell(12), str(NOTEBOOK), "exec"), namespace)
    assert env.ticks == namespace["runner"].total_actions_taken == 1
    assert namespace["final_obs"].levels_completed == 1
