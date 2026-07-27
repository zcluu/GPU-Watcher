from __future__ import annotations

import subprocess
import sys


def test_public_sdk_import_does_not_import_torch() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from watchgpu import GroupMemoryRequest, MemoryRequest, acquire; "
            "assert GroupMemoryRequest and MemoryRequest and acquire; "
            "assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_package_exposes_a_version() -> None:
    import watchgpu

    assert watchgpu.__version__ == "0.1.0"
