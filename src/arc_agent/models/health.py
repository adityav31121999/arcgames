"""Small raw-generation checks run before spending any game moves."""

import json
from pathlib import Path

from langchain_core.messages import HumanMessage
from PIL import Image


def check_model_health(llm, report_path):
    """Check basic text and image answers without action rescue or sanitization.

    This is a startup diagnostic, not a benchmark or proof of game competence.
    Keep the raw answers even on failure so checkpoint/runtime issues can be traced.
    """
    import torch
    import transformers

    model = llm.model
    processor = llm.processor
    config = getattr(model, "config", None)
    generation = getattr(model, "generation_config", None)
    tokenizer = getattr(processor, "tokenizer", processor)
    report = {
        "transformers": transformers.__version__, "torch": torch.__version__,
        "cuda": torch.version.cuda, "model_class": type(model).__name__,
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "model_config": config.to_dict() if hasattr(config, "to_dict") else str(config),
        "generation_config": generation.to_dict() if hasattr(generation, "to_dict") else str(generation),
        "checks": [],
    }
    probes = [
        ("text", "Reply with just the word READY. Do not explain.", "ready"),
        ("image", [
            {"type": "text", "text": "What color fills this image? Reply with one color word only."},
            {"type": "image", "image": Image.new("RGB", (128, 128), (255, 0, 0))},
        ], "red"),
    ]
    for name, content, expected in probes:
        check = {"name": name, "expected": expected, "ok": False}
        try:
            # Call the wrapper directly to avoid any previously cached response.
            result = llm._generate([HumanMessage(content=content)],
                                   max_tokens=64, temperature=0.0, raw_output=True)
            raw = str(result.generations[0].message.content)
            check.update(raw=raw, ok=raw.strip().strip(' .!\n\t"\'`').lower() == expected)
        except Exception as exc:
            check["error"] = f"{type(exc).__name__}: {exc}"
        report["checks"].append(check)
        print(json.dumps(check, ensure_ascii=True))
        if not check["ok"]:
            break
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if not all(check["ok"] for check in report["checks"]):
        raise RuntimeError(
            f"Model startup check failed ({report['checks'][-1]['name']}). Gameplay has not started. "
            f"Raw output and runtime/configuration details saved to {path}. "
            "Check checkpoint/tokenizer pairing and quantization runtime support. "
            "A formatting mismatch can also fail this strict diagnostic; inspect the raw answer."
        )
    print(f"Text and image startup checks passed. Report: {path}")
    return report
