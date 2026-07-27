# Contributing

Thank you for improving WatchGPU. Changes should preserve its cooperative resource model, transparent process identity, and strict rule that unrelated processes are never signalled.

## Development Setup

Python 3.11 or 3.12 is supported. GPU hardware is not required for the default test suite.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/check_public_tree.py
python -m ruff check src tests examples scripts
python -m mypy src/watchgpu
python -m pytest -q
```

CUDA tests are opt-in and must be run only on a host where allocating the test amount is authorized:

```bash
WATCHGPU_RUN_ALLOCATION_TESTS=1 python -m pytest -q tests/gpu
```

## Pull Requests

- Keep changes scoped and include a regression test for behavioral fixes.
- Preserve compatibility with host-local XDG paths and arbitrary Python locations.
- Do not add machine paths, real GPU UUIDs, logs, profiles, or credentials.
- Document user-facing flags and operational behavior.
- Explain concurrency, failure, and cleanup behavior for lease or lifecycle changes.

## Reporting Diagnostics

Before posting logs or JSON output, remove hostnames, usernames, absolute paths, PIDs, GPU UUIDs, task names, and any training arguments that reveal private data.
