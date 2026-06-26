#!/usr/bin/env python3
"""Smoke-test the platform prediction interface with an existing artifact."""

import argparse
import importlib
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ARTIFACT = REPO_ROOT / "results/model_sample.pth"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a smoke test against main.predict()."
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional image path. Defaults to a synthetic BGR image.",
    )
    return parser.parse_args()


def synthetic_bgr_image(size):
    import numpy as np

    return np.zeros((size, size, 3), dtype=np.uint8)


def read_bgr_image(path):
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image


def main():
    args = parse_args()

    if not MODEL_ARTIFACT.exists():
        print(
            f"missing model artifact: {MODEL_ARTIFACT.relative_to(REPO_ROOT)}",
            file=sys.stderr,
        )
        print(
            "run training first, or place a compatible artifact there", file=sys.stderr
        )
        return 1

    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))

    try:
        submission = importlib.import_module("main")
    except ImportError as exc:
        print(f"failed to import main.py: {exc}", file=sys.stderr)
        print("install the platform CPU runtime dependencies first", file=sys.stderr)
        return 1

    image = (
        read_bgr_image(args.image)
        if args.image
        else synthetic_bgr_image(submission.im_size)
    )
    prediction = submission.predict(image)
    allowed = set(submission.label)
    if prediction not in allowed:
        print(f"unexpected prediction: {prediction}; expected one of {sorted(allowed)}")
        return 1

    print(f"predict smoke OK: {prediction}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
