## Summary

Describe the user-visible outcome and why the change is needed.

## Verification

- [ ] Regression or feature tests added
- [ ] `python -m ruff check src tests`
- [ ] `python -m mypy src/watchgpu`
- [ ] `python -m pytest -q`
- [ ] No machine paths, real GPU UUIDs, logs, profiles, or credentials added

## Operational impact

Describe effects on GPU allocation, leases, process signalling, persistence, or background lifecycle. Write `None` when not applicable.
