"""Patch the Shaka-Labs DETR fork after install.

The upstream fork hardcodes `state_dim=5` (ALOHA's leader-arm dim). This project
uses `state_dim=8` (7 Panda joints + 1 gripper), so two files inside the
installed `detr` package need rewriting. Re-running is safe: a patched file
matches the post-patch text and is skipped.

Usage:
    uv run python scripts/patch_detr.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PATCHES: list[tuple[str, str, str]] = [
    (
        "models/detr_vae.py",
        "nn.Linear(5, hidden_dim)",
        "nn.Linear(state_dim, hidden_dim)",
    ),
    (
        "main.py",
        "args = parser.parse_args()",
        "args, _ = parser.parse_known_args()",
    ),
]


def detr_root() -> Path:
    spec = importlib.util.find_spec("detr")
    if spec is None or spec.origin is None:
        raise SystemExit(
            "detr package not found. Install first: "
            "`uv pip install git+https://github.com/Shaka-Labs/detr.git`"
        )
    return Path(spec.origin).parent


def apply_patch(file_path: Path, old: str, new: str) -> str:
    text = file_path.read_text()
    if new in text and old not in text:
        return "already patched"
    if old not in text:
        return f"MISSING TARGET — neither old nor new pattern found in {file_path}"
    file_path.write_text(text.replace(old, new))
    return "patched"


def main() -> int:
    root = detr_root()
    print(f"detr installed at: {root}")
    failed = False
    for rel, old, new in PATCHES:
        path = root / rel
        if not path.exists():
            print(f"  {rel}: FILE MISSING")
            failed = True
            continue
        status = apply_patch(path, old, new)
        print(f"  {rel}: {status}")
        if status.startswith("MISSING"):
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
