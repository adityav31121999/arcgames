import argparse
from pathlib import Path
import shutil
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")



def bundle_as_wheel(repo_root: Path, output_dir: Path):
    """Builds a pip wheel from the repository."""
    import subprocess
    import sys

    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(output_dir), str(repo_root)
    ])
    print(f"✅ Generated wheel in: {output_dir}")


def bundle_as_zip(repo_root: Path, output_zip: Path):
    """Packages src/, configs/, scripts/ into a zip archive."""
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in ["src", "configs", "scripts"]:
            src_folder = repo_root / folder
            for p in src_folder.rglob("*"):
                if "__pycache__" not in str(p) and not p.name.endswith(".pyc"):
                    rel_path = p.relative_to(repo_root)
                    zf.write(p, arcname=str(rel_path))
    print(f"✅ Created Kaggle dataset zip bundle: {output_zip}")


def main():
    parser = argparse.ArgumentParser(description="Bundle ARC-AGI-3 Agent for Kaggle")
    parser.add_argument("--format", choices=["wheel", "zip", "both"], default="both")
    parser.add_argument("--out", type=str, default="dist", help="Output directory")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / args.out

    if args.format in ("wheel", "both"):
        bundle_as_wheel(repo_root, out_dir)
    if args.format in ("zip", "both"):
        bundle_as_zip(repo_root, out_dir / "arc_agent_source.zip")


if __name__ == "__main__":
    main()
