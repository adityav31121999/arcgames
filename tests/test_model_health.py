import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from arc_agent.models.health import check_model_health


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="test", version=SimpleNamespace(cuda="test")))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="test"))
    return SimpleNamespace(model=SimpleNamespace(), processor=SimpleNamespace(), _generate=Mock())


def result(text):
    return SimpleNamespace(generations=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_health_checks_raw_text_then_real_image(runtime, tmp_path):
    runtime._generate.side_effect = [result("READY"), result("red")]
    report = check_model_health(runtime, tmp_path / "health.json")
    assert all(c["ok"] for c in report["checks"])
    first, second = runtime._generate.call_args_list
    assert first.kwargs["raw_output"] is True
    image = second.args[0][0].content[1]["image"]
    assert image.getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.parametrize("failure", [result("much own READY way way"), RuntimeError("kernel failed")])
def test_failed_health_persists_raw_evidence_and_stops(runtime, tmp_path, failure):
    runtime._generate.side_effect = [failure]
    with pytest.raises(RuntimeError, match="startup check failed"):
        check_model_health(runtime, tmp_path / "health.json")
    assert runtime._generate.call_count == 1
    check = json.loads((tmp_path / "health.json").read_text())["checks"][0]
    assert not check["ok"]
    assert "raw" in check or "error" in check


def test_health_detects_image_failure_after_text_passes(runtime, tmp_path):
    runtime._generate.side_effect = [result("READY"), result("blue")]
    with pytest.raises(RuntimeError, match="failed \\(image\\)"):
        check_model_health(runtime, tmp_path / "health.json")
    assert runtime._generate.call_count == 2
