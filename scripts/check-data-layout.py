#!/usr/bin/env python3
"""Validate the expected local training data layout."""

from pathlib import Path


TRAIN_ROOT = Path("datasets/6a39ed934d7b489daf5f80a4-momodel/train")
WEATHER_CATEGORIES = ("cloudy", "rainy", "snowy", "sunny")


def main() -> int:
    if not TRAIN_ROOT.is_dir():
        print(f"missing training image root: {TRAIN_ROOT}")
        return 1

    missing = [
        category
        for category in WEATHER_CATEGORIES
        if not (TRAIN_ROOT / category).is_dir()
    ]
    if missing:
        print(
            f"{TRAIN_ROOT} must directly contain category directories: "
            f"{', '.join(WEATHER_CATEGORIES)}"
        )
        print(f"missing: {', '.join(missing)}")
        return 1

    empty = [
        category
        for category in WEATHER_CATEGORIES
        if not any((TRAIN_ROOT / category).iterdir())
    ]
    if empty:
        print(f"empty category directories: {', '.join(empty)}")
        return 1

    print(f"training image root OK: {TRAIN_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
