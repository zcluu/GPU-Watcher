#!/usr/bin/env python3
"""Fail when a public source tree contains common secrets or host-local state."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
}
PROHIBITED_NAMES = {
    ".env",
    "config.toml",
    "leases.json",
    "profiles.jsonl",
    "restart.json",
    "shutdown.json",
    "supervisor.pid",
    "watchgpu.sock",
}
PATTERNS = {
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "GitHub token": re.compile(r"(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "concrete NFS/home path": re.compile(r"/(?:nfs\d*|home)/[A-Za-z0-9._-]+/"),
    "concrete NVIDIA GPU UUID": re.compile(
        r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    ),
}


def main() -> int:
    findings: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or any(part in IGNORED_DIRECTORIES for part in path.parts):
            continue
        relative = path.relative_to(ROOT)
        if path.name in PROHIBITED_NAMES or path.name.startswith("watchgpu.log"):
            findings.append(f"prohibited runtime file: {relative}")
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            findings.append(f"cannot read {relative}: {exc}")
            continue
        if b"\0" in data:
            continue
        text = data.decode("utf-8", errors="replace")
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{relative}:{line}: {label}")

    if findings:
        print("Public-tree check failed:", file=sys.stderr)
        for finding in findings:
            print(f"- {finding}", file=sys.stderr)
        return 1
    print("Public-tree check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
