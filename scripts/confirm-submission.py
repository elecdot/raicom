#!/usr/bin/env python3
"""Print the Submission Confirmation checklist."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_ARTIFACT = REPO_ROOT / "results/model_sample.pth"
EXPERIMENT_LOG = REPO_ROOT / "docs/experiments/README.md"


def relative(path):
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return path


def require_file(path, description):
    if path.is_file():
        print(f"OK: {description}: {relative(path)}")
        return True
    print(f"MISSING: {description}: {relative(path)}", file=sys.stderr)
    return False


def main():
    ok = True
    ok &= require_file(SUBMISSION_ARTIFACT, "Submission Artifact")
    ok &= require_file(EXPERIMENT_LOG, "Experiment review log")
    if not ok:
        return 1

    print()
    print("Manual Submission Confirmation checklist:")
    print("- Identify the Experiment Run promoted to results/model_sample.pth.")
    print(
        "- Confirm docs/experiments/README.md records its Internal Validation Macro F1."
    )
    print("- Confirm main.py matches the promoted Model Candidate architecture.")
    print("- Run `just smoke-predict` in a runtime with platform dependencies.")
    print(
        "- Confirm the Platform Run will include requirements.txt and results/model_sample.pth."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
