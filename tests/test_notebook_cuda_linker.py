import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def linker(monkeypatch):
    notebook = Path(__file__).resolve().parents[1] / "notebooks/assumeAndPlay.ipynb"
    code = "".join(json.loads(notebook.read_text(encoding="utf-8"))["cells"][3]["source"])
    if "configure_cuda_linker" not in code:
        code = (Path(__file__).resolve().parents[1] / "notebooks/kaggle_cuda_linker.py").read_text(encoding="utf-8")
    module = ast.parse(code)
    module.body = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    environment = {"LD_LIBRARY_PATH": "/existing/runtime"}
    import re
    namespace = dict(Path=Path, os=SimpleNamespace(environ=environment, pathsep=":"),
                     re=re,
                     subprocess=SimpleNamespace(check_output=lambda *a, **kw: "release 13.0,"))
    exec(compile(module, str(notebook), "exec"), namespace)
    return namespace["configure_cuda_linker"], environment


def library(root, relative):
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch()
    return target


def test_pip_lib_layout_and_versioned_driver_are_discovered(linker, tmp_path, monkeypatch):
    configure, environment = linker
    root = tmp_path / "site/nvidia/cu13"
    cudart = library(root, "lib/libcudart.so.13")
    nvrtc = library(root, "lib/libnvrtc.so.13")
    driver = library(tmp_path / "driver", "libcuda.so.1")
    library(root, "lib/stubs/libcuda.so")
    # Windows test hosts need not grant Linux symlink creation privileges.
    aliases = {}
    monkeypatch.setattr(Path, "symlink_to", lambda self, target: aliases.update({self: target}))
    selected = configure(root, [], [driver.parent], tmp_path / "work")
    assert selected == {"cudart": cudart, "nvrtc": nvrtc, "cuda": driver}
    assert {p.name for p in aliases} == {"libcudart.so", "libnvrtc.so", "libcuda.so"}
    assert environment["LIBRARY_PATH"] == str(tmp_path / "work/.cuda-linker-cu13")
    assert environment["LD_LIBRARY_PATH"] == "/existing/runtime"


def test_rejects_old_runtime_wheels_before_changing_paths(linker, tmp_path):
    configure, environment = linker
    root = tmp_path / "site/nvidia/cu13"
    root.mkdir(parents=True)
    library(tmp_path / "site", "nvidia/cuda_runtime/lib/libcudart.so.12")
    library(tmp_path / "site", "nvidia/cuda_nvrtc/lib/libnvrtc.so.12")
    driver = library(tmp_path, "driver/libcuda.so.1")
    with pytest.raises(RuntimeError, match="Missing compatible CUDA 13"):
        configure(root, [tmp_path / "site"], [driver.parent], tmp_path / "work")
    assert "LIBRARY_PATH" not in environment


def test_finds_matching_nvrtc_in_separate_wheel(linker, tmp_path, monkeypatch):
    configure, environment = linker
    root = tmp_path / "site/nvidia/cu13"
    library(root, "lib/libcudart.so.13")
    nvrtc = library(tmp_path / "site", "nvidia/cuda_nvrtc/lib/libnvrtc.so.13.0")
    driver = library(tmp_path, "driver/libcuda.so.1")
    monkeypatch.setattr(Path, "symlink_to", lambda *a: None)
    selected = configure(root, [tmp_path / "site"], [driver.parent], tmp_path / "work")
    assert selected["nvrtc"] == nvrtc
