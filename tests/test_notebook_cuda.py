"""Catch the reported FlashInfer compiler mismatch before loading weights."""

import ast
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


@pytest.fixture
def configure():
    notebook = Path(__file__).resolve().parents[1] / "notebooks/assumeAndPlay.ipynb"
    source = "".join(json.loads(notebook.read_text(encoding="utf-8"))["cells"][3]["source"])
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    env = SimpleNamespace(environ={}, pathsep=os.pathsep)
    namespace = dict(Path=Path, os=env, subprocess=SimpleNamespace(
        check_output=lambda command, **kw: "release 12.8," if Path(command[0]).parent.parent.name == "old" else "release 12.9,",
        STDOUT=subprocess.STDOUT, CalledProcessError=subprocess.CalledProcessError),
        sys=SimpleNamespace(modules={}),
        torch=SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda _: (12, 0))))
    exec(compile(tree, str(notebook), "exec"), namespace)
    return namespace["configure_blackwell_toolkit"], namespace


def toolkit(tmp_path, name):
    root = tmp_path / name
    (root / "include").mkdir(parents=True)
    (root / "include/cuda_runtime.h").touch()
    return root


def test_selects_new_toolkit_and_sets_worker_environment(configure, tmp_path):
    fn, ns = configure
    old, new = [toolkit(tmp_path, name) for name in ("old", "new")]
    fn([old, new])
    assert ns["os"].environ["CUDA_HOME"] == str(new)
    assert ns["os"].environ["PATH"].startswith(str(new / "bin"))


def test_old_toolkit_reports_offline_remedy(configure, tmp_path):
    fn, _ = configure
    with pytest.raises(RuntimeError, match="Attach/install"):
        fn([toolkit(tmp_path, "old")])


def test_inherited_old_toolkit_falls_back_to_installed_toolkit(configure, tmp_path, capsys):
    fn, ns = configure
    ns["os"].environ["CUDA_HOME"] = str(toolkit(tmp_path, "old"))
    new = toolkit(tmp_path, "new")
    fn([new])
    assert ns["os"].environ["CUDA_HOME"] == str(new)
    assert ns["os"].environ["CUDA_PATH"] == str(new)
    assert "Skipped unusable CUDA toolkit" in capsys.readouterr().out


def test_usable_environment_toolkit_is_preferred(configure, tmp_path):
    fn, ns = configure
    preferred = toolkit(tmp_path, "preferred")
    ns["os"].environ["CUDA_HOME"] = str(preferred)
    fn([toolkit(tmp_path, "new")])
    assert ns["os"].environ["CUDA_HOME"] == str(preferred)


def test_cached_flashinfer_requires_restart(configure, tmp_path):
    fn, ns = configure
    ns["sys"].modules["flashinfer"] = object()
    with pytest.raises(RuntimeError, match="Restart the kernel"):
        fn([toolkit(tmp_path, "new")])
    assert not ns["os"].environ


def test_other_gpu_keeps_existing_environment(configure):
    fn, ns = configure
    ns["torch"].cuda.get_device_capability = lambda _: (9, 0)
    fn([])
    assert not ns["os"].environ
