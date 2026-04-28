from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.parametrize(
    "script",
    ["01_quickstart.py", "02_cost_overrides.py", "03_custom_cost_model.py"],
)
def test_example_runs_without_error(script: str) -> None:
    target = EXAMPLES / script
    assert target.exists(), f"example missing: {target}"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(EXAMPLES), env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, str(target)],
        capture_output=True,
        text=True,
        cwd=EXAMPLES,
        env=env,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
