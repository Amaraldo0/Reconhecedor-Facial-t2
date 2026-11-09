"""Procura arquivos proibidos no índice e em todo o histórico git (§18 e Anexo B).

Uso: python scripts/privacy_audit.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

FORBIDDEN_EXTENSIONS = re.compile(
    r"\.(jpe?g|bmp|webp|heic|mp4|avi|mov|mkv|db|sqlite3?|npy|npz|pkl|pt|pth|ckpt|onnx|safetensors|tgz)$", re.I
)
FORBIDDEN_DIRS = re.compile(r"(^|/)(faces|embeddings|database|consent|termos|checkpoints|lfw|private)(/|$)", re.I)
ALLOWED_PNG = re.compile(r"^experiments/results/figures/[^/]+\.png$")


def violations(paths: set[str]) -> list[str]:
    bad = []
    for path in sorted(paths):
        if path == "data/README.md" or path.startswith(("docs/", "src/", "tests/")) and path.endswith((".md", ".py")):
            continue
        is_png = path.lower().endswith(".png") and not ALLOWED_PNG.match(path)
        if FORBIDDEN_EXTENSIONS.search(path) or FORBIDDEN_DIRS.search(path) or is_png or path.startswith("data/"):
            bad.append(path)
    return bad


def tracked_paths(repo: Path) -> set[str]:
    index = subprocess.run(["git", "ls-files"], cwd=repo, capture_output=True, text=True, check=True)
    history = subprocess.run(
        ["git", "log", "--all", "--name-only", "--pretty=format:"], cwd=repo, capture_output=True, text=True, check=True
    )
    return {line.strip() for line in (index.stdout + "\n" + history.stdout).splitlines() if line.strip()}


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    paths = tracked_paths(repo)
    bad = violations(paths)
    print(f"{len(paths)} caminhos verificados (índice e histórico).")
    if bad:
        print("VIOLAÇÕES:", *bad, sep="\n  ")
        sys.exit(1)
    print("Nenhum arquivo proibido encontrado.")


if __name__ == "__main__":
    main()
