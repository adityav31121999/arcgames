"""Run on an internet-enabled Kaggle notebook matching the target Python version.

Downloads wheels without installing or importing the GPU runtime. Offline pip
resolution is checked, but CUDA execution still needs the model startup tests.
"""

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path


def download_vllm_wheels(output_dir=None, vllm_requirement="vllm", extra_wheel_dirs=()):
    if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        raise RuntimeError("Run this on Kaggle/Linux x86_64 with the same Python version as the inference notebook.")
    if output_dir is None:
        out = Path(tempfile.mkdtemp(prefix="vllm_wheels_", dir="/kaggle/working"))
    else:
        out = Path(output_dir).resolve()
        if out.exists() and any(out.iterdir()):
            raise RuntimeError("Choose an empty output directory so old wheel versions cannot be mixed in.")
        out.mkdir(parents=True, exist_ok=True)

    # Resolve everything together. vLLM supplies torch/transformers constraints.
    # NVFP4 FlashInfer JIT also needs the compiler and CUDA headers at runtime.
    packages = [vllm_requirement, "cuda-toolkit[nvcc]", "langchain-core", "pydantic", "pyyaml",
                "pillow", "numpy", "matplotlib", "arc-agi", "arcengine"]
    (out / "requirements.in").write_text("\n".join(packages) + "\n", encoding="utf-8")
    command = [sys.executable, "-m", "pip", "download", "--only-binary=:all:",
               "--dest", str(out)]
    for directory in extra_wheel_dirs:
        command += ["--find-links", str(directory)]
    print(f"Resolving and downloading the complete stack into {out}")
    subprocess.check_call(command + packages)

    # Ignore installed packages so none can conceal a missing offline wheel.
    report_path = out / "offline-resolution.json"
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed",
        "--no-index", "--only-binary=:all:", "--find-links", str(out),
        "--report", str(report_path), *packages,
    ])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pins = sorted(f"{item['metadata']['name']}=={item['metadata']['version']}"
                  for item in report["install"])
    (out / "requirements-lock.txt").write_text("\n".join(pins) + "\n", encoding="utf-8")
    (out / "build-environment.json").write_text(json.dumps({
        "python": sys.version, "platform": platform.platform(),
        "machine": platform.machine(), "packages": packages,
        "validation": "offline dependency resolution only; GPU inference not tested",
    }, indent=2), encoding="utf-8")
    print(f"SUCCESS: {len(list(out.glob('*.whl')))} wheels; offline resolution passed.")
    print(f"Upload this entire directory as your Kaggle wheel dataset: {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="Empty directory; default creates a new directory in /kaggle/working")
    parser.add_argument("--vllm", default="vllm", help="vLLM requirement, e.g. vllm==<tested-version>")
    parser.add_argument("--find-links", action="append", default=[], help="Additional local wheels, e.g. ARC support wheels")
    # Notebook kernels add their own -f argument; pasted-cell use takes defaults.
    args = parser.parse_args([] if "ipykernel" in sys.modules else None)
    download_vllm_wheels(args.out, args.vllm, args.find_links)
