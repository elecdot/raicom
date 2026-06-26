import subprocess
import sys
from pathlib import Path


def test_submission_entrypoint_contract_matches_shared_model():
    result = subprocess.run(
        [sys.executable, "scripts/check-submission-entrypoint.py"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
