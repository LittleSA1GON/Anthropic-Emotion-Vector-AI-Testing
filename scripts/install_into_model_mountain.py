"""Copy the self-contained research extension beside Model Mountain v4 sources.

Does not overwrite project files or modify security policy. Dry-run by default.
"""
import argparse
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Extracted Model Mountain v4 repository root")
    p.add_argument("--apply", action="store_true", help="Actually copy the extension")
    args = p.parse_args()
    root = Path(args.root).resolve()
    if not (root / "model_mountain").is_dir() or not (root / "tests").is_dir():
        raise SystemExit("Expected an extracted Model Mountain tree with model_mountain/ and tests/; refusing")
    source = Path(__file__).resolve().parents[1]
    target = root / "experiments" / "emotion_vectors"
    if target.exists():
        raise SystemExit(f"Target exists, refusing to overwrite: {target}")
    print(f"Overlay: {source} -> {target}")
    if args.apply:
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "runs", ".venv-mia", "*.zip"))
        print("Research extension added without modifying model_mountain/gateway.py or any existing modules")
    else:
        print("Dry run only; add --apply to copy")


if __name__ == "__main__":
    main()
