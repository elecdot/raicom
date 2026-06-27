#!/usr/bin/env python3
"""Evaluate simple test-time augmentation for an Experiment Run."""

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "results/runs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs/diagnostics"

sys.path.insert(0, str(REPO_ROOT))


class PredictionImageDataset(Dataset):
    def __init__(self, rows, labels, transform):
        self.rows = rows
        self.labels = labels
        self.label_to_index = {label: index for index, label in enumerate(labels)}
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = resolve_path(Path(row["path"]))
        if not image_path.is_file():
            raise FileNotFoundError(f"missing validation image: {relative(image_path)}")
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), self.label_to_index[row["true_label"]]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate horizontal-flip TTA for an Experiment Run."
    )
    parser.add_argument(
        "run",
        help="Experiment Run ID or path under results/runs/.",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="Override model artifact path. Defaults to the run's local model.pth.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated Markdown and CSV reports.",
    )
    parser.add_argument(
        "--stem",
        default=None,
        help="Output filename stem. Defaults to <run-id>-hflip-tta.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        default=None,
        help="Enable CUDA automatic mixed precision.",
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable automatic mixed precision.",
    )
    return parser.parse_args()


def resolve_path(path):
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def relative(path):
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_run_dir(run):
    path = Path(run)
    if path.is_absolute() or len(path.parts) > 1:
        run_dir = resolve_path(path)
    else:
        run_dir = RUNS_DIR / run
    if not run_dir.is_dir():
        raise FileNotFoundError(f"missing Experiment Run directory: {run}")
    return run_dir


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"missing JSON: {relative(path)}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_predictions(path):
    if not path.is_file():
        raise FileNotFoundError(f"missing validation predictions: {relative(path)}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])
    required = {"path", "true_label", "pred_label", "correct", "confidence"}
    missing = sorted(required - fieldnames)
    if missing:
        raise ValueError(
            f"{relative(path)} is missing required fields: {', '.join(missing)}"
        )
    if not rows:
        raise ValueError(f"empty validation predictions: {relative(path)}")
    return rows


def resolve_artifact(run_dir, metadata, artifact_arg):
    candidates = []
    if artifact_arg is not None:
        candidates.append(resolve_path(artifact_arg))
    candidates.append(run_dir / "model.pth")
    artifact = metadata.get("artifact")
    if artifact:
        candidates.append(resolve_path(Path(artifact)))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    checked = ", ".join(relative(candidate) for candidate in candidates)
    raise FileNotFoundError(f"missing model artifact; checked: {checked}")


def resolve_device(device_name):
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def resolve_amp(amp_arg, device):
    if amp_arg is None:
        return device.type == "cuda"
    return bool(amp_arg and device.type == "cuda")


def interpolation_mode(name):
    modes = {
        "bilinear": transforms.InterpolationMode.BILINEAR,
        "bicubic": transforms.InterpolationMode.BICUBIC,
    }
    try:
        return modes[name]
    except KeyError as exc:
        choices = ", ".join(sorted(modes))
        raise ValueError(
            f"unknown interpolation {name!r}; choose one of: {choices}"
        ) from exc


def build_transform(metadata, candidate):
    image_size = int(metadata.get("image_size") or candidate.image_size)
    preprocessing = metadata.get("preprocessing") or {}
    interpolation = interpolation_mode(
        preprocessing.get("interpolation") or candidate.interpolation
    )
    mean = tuple(preprocessing.get("mean") or candidate.mean)
    std = tuple(preprocessing.get("std") or candidate.std)
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=interpolation),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def load_model(metadata, artifact_path, device):
    from models.registry import get_model_candidate

    candidate = get_model_candidate(metadata["model_candidate"])
    model = candidate.build_model(pretrained=False)
    try:
        state = torch.load(artifact_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(artifact_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, candidate


def evaluate(model, loader, labels, device, amp_enabled, hflip):
    y_true = []
    y_pred = []
    total = 0
    correct = 0
    non_blocking = device.type == "cuda"
    start = time.perf_counter()
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(x)
                if hflip:
                    logits = (logits + model(torch.flip(x, dims=(3,)))) / 2
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
    return {
        "accuracy": correct / total,
        "macro_f1": macro_f1(y_true, y_pred, labels),
        "per_class": per_class_metrics(y_true, y_pred, labels),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
        ).tolist(),
        "seconds": time.perf_counter() - start,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def per_class_metrics(y_true, y_pred, labels):
    label_ids = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        zero_division=0,
    )
    rows = {}
    for index, label in enumerate(labels):
        rows[label] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
    return rows


def macro_f1(y_true, y_pred, labels):
    per_class = per_class_metrics(y_true, y_pred, labels)
    return sum(row["f1"] for row in per_class.values()) / len(labels)


def prediction_indices_from_csv(rows, labels):
    label_to_index = {label: index for index, label in enumerate(labels)}
    y_true = [label_to_index[row["true_label"]] for row in rows]
    y_pred = [label_to_index[row["pred_label"]] for row in rows]
    return y_true, y_pred


def summarize_changes(rows, labels, tta_pred):
    label_to_index = {label: index for index, label in enumerate(labels)}
    changes = []
    for row, after_index in zip(rows, tta_pred):
        before_index = label_to_index[row["pred_label"]]
        true_index = label_to_index[row["true_label"]]
        if before_index == after_index:
            continue
        correct_before = int(before_index == true_index)
        correct_after = int(after_index == true_index)
        if correct_before == 0 and correct_after == 1:
            change_type = "corrected"
        elif correct_before == 1 and correct_after == 0:
            change_type = "regressed"
        else:
            change_type = "changed_wrong_to_wrong"
        changes.append(
            {
                "path": row["path"],
                "true_label": row["true_label"],
                "pred_before": row["pred_label"],
                "pred_after": labels[after_index],
                "correct_before": correct_before,
                "correct_after": correct_after,
                "confidence_before": row["confidence"],
                "change_type": change_type,
            }
        )
    return changes


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path, payload):
    labels = payload["labels"]
    summary = payload["summary"]
    changes = payload["changes"]
    lines = [
        "# TTA Diagnostic",
        "",
        "Horizontal-flip test-time augmentation evaluated on an Experiment Run's "
        "Internal Validation Split.",
        "",
        f"- Run: `{payload['run_id']}`",
        f"- Artifact: `{payload['artifact']}`",
        f"- Validation predictions: `{payload['predictions']}`",
        f"- Summary CSV: `{payload['summary_csv']}`",
        f"- Changes CSV: `{payload['changes_csv']}`",
        f"- Device: `{payload['device']}`",
        f"- AMP enabled: `{payload['amp_enabled']}`",
        "",
        "## Summary",
        "",
        "| Source | Macro F1 | Accuracy | Delta vs CSV | Seconds |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| `{row['source']}` | "
            f"{row['macro_f1']:.4f} | "
            f"{row['accuracy']:.4f} | "
            f"{row['delta_vs_csv']:.4f} | "
            f"{row['seconds']:.2f} |"
        )

    counts = Counter(row["change_type"] for row in changes)
    lines.extend(
        [
            "",
            "## TTA Changes vs CSV Predictions",
            "",
            f"- Changed: `{len(changes)}`",
            f"- Corrected: `{counts['corrected']}`",
            f"- Regressed: `{counts['regressed']}`",
            f"- Wrong-to-wrong: `{counts['changed_wrong_to_wrong']}`",
            "",
            "| Label | CSV F1 | TTA F1 | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    csv_per_class = payload["csv_per_class"]
    tta_per_class = payload["tta_per_class"]
    for label in labels:
        csv_f1 = csv_per_class[label]["f1"]
        tta_f1 = tta_per_class[label]["f1"]
        lines.append(f"| `{label}` | {csv_f1:.4f} | {tta_f1:.4f} | {tta_f1 - csv_f1:.4f} |")

    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- This is an in-split diagnostic for a fixed validation split.",
            "- Treat positive TTA deltas as candidates for cross-run validation, not as "
            "a guaranteed Submission Prediction Interface improvement.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary_row(source, result, csv_macro_f1):
    return {
        "source": source,
        "macro_f1": result["macro_f1"],
        "accuracy": result["accuracy"],
        "delta_vs_csv": result["macro_f1"] - csv_macro_f1,
        "seconds": result.get("seconds", 0.0),
    }


def main():
    args = parse_args()
    if args.batch_size < 1:
        print("--batch-size must be at least 1", file=sys.stderr)
        return 2
    if args.num_workers < 0:
        print("--num-workers must be non-negative", file=sys.stderr)
        return 2

    try:
        run_dir = resolve_run_dir(args.run)
        metadata = read_json(run_dir / "metadata.json")
        rows = read_predictions(run_dir / "val_predictions.csv")
        artifact_path = resolve_artifact(run_dir, metadata, args.artifact)
        device = resolve_device(args.device)
        amp_enabled = resolve_amp(args.amp, device)
        model, candidate = load_model(metadata, artifact_path, device)
        labels = tuple(metadata.get("labels") or candidate.labels)
        transform = build_transform(metadata, candidate)
        dataset = PredictionImageDataset(rows, labels, transform)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        csv_true, csv_pred = prediction_indices_from_csv(rows, labels)
        csv_result = {
            "accuracy": sum(
                int(true == pred) for true, pred in zip(csv_true, csv_pred)
            )
            / len(csv_true),
            "macro_f1": macro_f1(csv_true, csv_pred, labels),
            "per_class": per_class_metrics(csv_true, csv_pred, labels),
            "seconds": 0.0,
        }
        single_result = evaluate(
            model,
            loader,
            labels,
            device,
            amp_enabled,
            hflip=False,
        )
        tta_result = evaluate(
            model,
            loader,
            labels,
            device,
            amp_enabled,
            hflip=True,
        )
        changes = summarize_changes(rows, labels, tta_result["y_pred"])
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(f"TTA diagnostic failed: {exc}", file=sys.stderr)
        return 1

    run_id = metadata.get("run_id") or run_dir.name
    stem = args.stem or f"{run_id}-hflip-tta"
    output_dir = resolve_path(args.output_dir)
    md_path = output_dir / f"{stem}.md"
    summary_csv_path = output_dir / f"{stem}-summary.csv"
    changes_csv_path = output_dir / f"{stem}-changes.csv"
    summary = [
        build_summary_row("csv_predictions", csv_result, csv_result["macro_f1"]),
        build_summary_row("single_pass_eval", single_result, csv_result["macro_f1"]),
        build_summary_row("hflip_tta", tta_result, csv_result["macro_f1"]),
    ]
    write_csv(
        summary_csv_path,
        [
            {
                "source": row["source"],
                "macro_f1": f"{row['macro_f1']:.6f}",
                "accuracy": f"{row['accuracy']:.6f}",
                "delta_vs_csv": f"{row['delta_vs_csv']:.6f}",
                "seconds": f"{row['seconds']:.6f}",
            }
            for row in summary
        ],
        ["source", "macro_f1", "accuracy", "delta_vs_csv", "seconds"],
    )
    write_csv(
        changes_csv_path,
        changes,
        [
            "path",
            "true_label",
            "pred_before",
            "pred_after",
            "correct_before",
            "correct_after",
            "confidence_before",
            "change_type",
        ],
    )
    write_markdown(
        md_path,
        {
            "run_id": run_id,
            "artifact": relative(artifact_path),
            "predictions": relative(run_dir / "val_predictions.csv"),
            "summary_csv": relative(summary_csv_path),
            "changes_csv": relative(changes_csv_path),
            "device": str(device),
            "amp_enabled": amp_enabled,
            "labels": labels,
            "summary": summary,
            "changes": changes,
            "csv_per_class": csv_result["per_class"],
            "tta_per_class": tta_result["per_class"],
        },
    )
    print("TTA diagnostic OK")
    print(f"summary: {relative(summary_csv_path)}")
    print(f"changes: {relative(changes_csv_path)}")
    print(f"markdown: {relative(md_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
