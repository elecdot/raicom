#!/usr/bin/env python3
"""Check that self-contained main.py matches the shared baseline model."""

import ast
from pathlib import Path


MAIN_PATH = Path("main.py")
MODEL_PATH = Path("weather_model.py")


def parse(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def assignment_value(module, name):
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found")


def class_dump(module, name):
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return ast.dump(node, include_attributes=False)
    raise KeyError(f"{name} not found")


def main():
    submission = parse(MAIN_PATH)
    shared = parse(MODEL_PATH)

    expected_labels = assignment_value(shared, "LABELS")
    submission_labels = assignment_value(submission, "label")
    if submission_labels != expected_labels:
        print(f"main.py label drift: {submission_labels} != {expected_labels}")
        return 1

    expected_size = assignment_value(shared, "IMAGE_SIZE")
    submission_size = assignment_value(submission, "im_size")
    if submission_size != expected_size:
        print(f"main.py image size drift: {submission_size} != {expected_size}")
        return 1

    if class_dump(submission, "WeatherCNN") != class_dump(shared, "WeatherCNN"):
        print("main.py WeatherCNN drifted from weather_model.py")
        return 1

    print("submission entrypoint OK: main.py matches weather_model.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
