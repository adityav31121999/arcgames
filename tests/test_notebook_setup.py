"""Installation reruns must not force an endless kernel restart loop."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def setup(tmp_path):
    notebook = Path(__file__).resolve().parents[1] / "notebooks/assumeAndPlay.ipynb"
    source = "".join(json.loads(notebook.read_text(encoding="utf-8"))["cells"][1]["source"])
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    process = SimpleNamespace(getpid=Mock(return_value=100))
    calls, changes = [], []

    def check_call(command):
        calls.append(command)
        if "--report" in command:
            Path(command[command.index("--report") + 1]).write_text(json.dumps({"install": changes}))

    def path(value):
        if str(value) == "/proc/self/stat":
            return SimpleNamespace(read_text=lambda: "100 (python) " + " ".join(["0"] * 20))
        return Path(value)

    namespace = dict(Path=path, json=json, os=process,
                     sys=SimpleNamespace(executable="python", modules={"torch": object()}),
                     subprocess=SimpleNamespace(check_call=check_call))
    exec(compile(tree, str(notebook), "exec"), namespace)
    inputs = tmp_path / "input"
    inputs.mkdir()
    return namespace["setup_vllm"], inputs, tmp_path / "work", calls, changes, process


def test_satisfied_dependencies_skip_install_even_with_torch_imported(setup):
    function, inputs, work, calls, _, _ = setup
    function(inputs, work)
    assert len(calls) == 1 and "--dry-run" in calls[0]
    assert "--upgrade" not in calls[0]
    assert "cuda-toolkit[nvcc]>=12.9" in calls[0]


def test_install_requires_one_restart_then_rerun_skips(setup):
    function, inputs, work, calls, changes, process = setup
    changes.append({"metadata": {"name": "vllm", "version": "test"}})
    with pytest.raises(RuntimeError, match="Installation complete"):
        function(inputs, work)
    assert len(calls) == 2
    with pytest.raises(RuntimeError, match="changed in this Python kernel"):
        function(inputs, work)
    assert len(calls) == 2
    process.getpid.return_value = 101
    changes.clear()
    function(inputs, work)
    assert len(calls) == 3 and "--dry-run" in calls[-1]


def test_selected_lock_is_used_without_mixing_other_wheelhouses(setup):
    function, inputs, work, calls, _, _ = setup
    selected = inputs / "selected"
    selected.mkdir()
    (selected / "requirements-lock.txt").write_text("vllm==test")
    old = inputs / "old"
    old.mkdir()
    (old / "old.whl").touch()
    function(inputs, work)
    assert str(selected / "requirements-lock.txt") in calls[0]
    assert str(old) not in calls[0]
