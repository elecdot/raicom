#!/usr/bin/env python3
"""Promote a Model Artifact to the fixed Submission Artifact path."""

import argparse
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_ARTIFACT = REPO_ROOT / "results/model_sample.pth"


def relative(path):
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Promote a Model Artifact to results/model_sample.pth."
    )
    parser.add_argument(
        "artifact",
        type=Path,
        help="Path to an Experiment Run model artifact, usually results/runs/<run-id>/model.pth.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.artifact
    if not source.is_absolute():
        source = REPO_ROOT / source

    if not source.exists():
        print(f"missing model artifact: {relative(source)}", file=sys.stderr)
        return 1
    if not source.is_file():
        print(f"model artifact is not a file: {relative(source)}", file=sys.stderr)
        return 1

    SUBMISSION_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, SUBMISSION_ARTIFACT)

    print("promote submission OK")
    print(f"source: {relative(source)}")
    print(f"submission_artifact: {relative(SUBMISSION_ARTIFACT)}")
    print("next: record the decision in docs/experiments/README.md")
    print("next: run `just smoke-predict` in a runtime with platform dependencies")
    print("next: run `just confirm-submission` before the Platform Run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
