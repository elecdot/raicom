#!/usr/bin/env python3
"""Summarize validation error overlap across comparable Experiment Runs."""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "results/runs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs/diagnostics"
DEFAULT_STEM = "efficientnet-b0-stable-error-pool"
DEFAULT_RUNS = [
    ("seed42", "efficientnet-b0-seed42-bs32-w4"),
    ("train2025", "efficientnet-b0-split42-train2025-bs32-w4"),
    ("train3407", "efficientnet-b0-split42-train3407-bs32-w4"),
]
BOUNDARY_PAIRS = {
    ("cloudy", "sunny"),
    ("sunny", "cloudy"),
    ("rainy", "cloudy"),
    ("cloudy", "rainy"),
}


def relative(path):
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_run(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--run must use short_id=run_id_or_path format"
        )
    short_id, run = value.split("=", 1)
    if not short_id or not run:
        raise argparse.ArgumentTypeError(
            "--run must include both short_id and run_id_or_path"
        )
    if not short_id.replace("_", "").replace("-", "").isalnum():
        raise argparse.ArgumentTypeError(
            "short_id must contain only letters, numbers, underscores, or hyphens"
        )
    return short_id.replace("-", "_"), run


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize stable validation errors across Experiment Runs."
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        default=None,
        help=(
            "Comparable run as short_id=run_id_or_path. "
            "May be repeated. Defaults to the fixed-split B0 runs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the generated CSV and Markdown report.",
    )
    parser.add_argument(
        "--stem",
        default=DEFAULT_STEM,
        help="Output filename stem without extension.",
    )
    parser.add_argument(
        "--high-confidence-threshold",
        type=float,
        default=0.8,
        help="Mean confidence threshold for stable_high_conf_error.",
    )
    return parser.parse_args()


def resolve_run_dir(value):
    path = Path(value)
    if path.is_absolute():
        return path
    direct = REPO_ROOT / path
    if direct.exists():
        return direct
    return RUNS_DIR / value


def read_predictions(run_dir):
    path = run_dir / "val_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing val_predictions.csv: {relative(path)}"
        )
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty val_predictions.csv: {relative(path)}")
    by_path = {}
    for row in rows:
        image_path = row["path"]
        if image_path in by_path:
            raise ValueError(f"duplicate path in {relative(path)}: {image_path}")
        by_path[image_path] = row
    return path, by_path


def validate_comparable(run_maps):
    first_short, first_rows = run_maps[0]
    first_paths = set(first_rows)
    for short_id, rows in run_maps[1:]:
        paths = set(rows)
        missing = sorted(first_paths - paths)
        extra = sorted(paths - first_paths)
        if missing or extra:
            detail = []
            if missing:
                detail.append(f"missing vs {first_short}: {missing[:3]}")
            if extra:
                detail.append(f"extra vs {first_short}: {extra[:3]}")
            raise ValueError(f"runs are not comparable for {short_id}: {'; '.join(detail)}")
    for image_path in sorted(first_paths):
        true_label = first_rows[image_path]["true_label"]
        for short_id, rows in run_maps[1:]:
            other_label = rows[image_path]["true_label"]
            if other_label != true_label:
                raise ValueError(
                    "true_label mismatch for "
                    f"{image_path}: {first_short}={true_label}, "
                    f"{short_id}={other_label}"
                )


def diagnostic_category(error_count, mean_confidence, error_pairs, threshold):
    if error_count == 3 and mean_confidence >= threshold:
        return "stable_high_conf_error"
    if error_count == 3:
        return "stable_model_error"
    if error_count >= 2 and any(pair in BOUNDARY_PAIRS for pair in error_pairs):
        return "boundary_case"
    if error_count == 1:
        return "random_sensitive"
    return "low_priority"


def summarize_rows(runs, run_maps, high_confidence_threshold):
    rows = []
    run_count = len(runs)
    for image_path in sorted(run_maps[0][1]):
        true_label = run_maps[0][1][image_path]["true_label"]
        predictions = []
        confidences = []
        correct_values = []
        error_pairs = []
        for short_id, prediction_rows in run_maps:
            row = prediction_rows[image_path]
            pred_label = row["pred_label"]
            confidence = float(row["confidence"])
            correct = int(row["correct"])
            predictions.append((short_id, pred_label))
            confidences.append((short_id, confidence))
            correct_values.append((short_id, correct))
            if correct == 0:
                error_pairs.append((true_label, pred_label))

        error_count = len(error_pairs)
        if error_count == 0:
            continue

        wrong_labels = [pred for _, pred in predictions if pred != true_label]
        mean_confidence = sum(confidence for _, confidence in confidences) / run_count
        unique_error_pairs = list(dict.fromkeys(error_pairs))
        row = {
            "path": image_path,
            "true_label": true_label,
            "error_count": error_count,
            "run_count": run_count,
            "stability_tier": f"{error_count}/{run_count}",
            "pred_labels": ";".join(
                f"{short_id}:{pred_label}" for short_id, pred_label in predictions
            ),
            "confidences": ";".join(
                f"{short_id}:{confidence:.6f}"
                for short_id, confidence in confidences
            ),
            "max_confidence": max(confidence for _, confidence in confidences),
            "mean_confidence": mean_confidence,
            "same_wrong_label": int(
                bool(wrong_labels) and len(set(wrong_labels)) == 1
            ),
            "error_pairs": ";".join(
                f"{true}->{pred}" for true, pred in unique_error_pairs
            ),
            "diagnostic_category": diagnostic_category(
                error_count,
                mean_confidence,
                unique_error_pairs,
                high_confidence_threshold,
            ),
        }
        for short_id, pred_label in predictions:
            row[f"pred_{short_id}"] = pred_label
        for short_id, confidence in confidences:
            row[f"confidence_{short_id}"] = confidence
        for short_id, correct in correct_values:
            row[f"correct_{short_id}"] = correct
        rows.append(row)

    rows.sort(
        key=lambda row: (
            -int(row["error_count"]),
            -float(row["mean_confidence"]),
            row["path"],
        )
    )
    return rows


def write_csv(path, rows, runs):
    run_fields = []
    for short_id, _ in runs:
        run_fields.extend(
            [
                f"pred_{short_id}",
                f"confidence_{short_id}",
                f"correct_{short_id}",
            ]
        )
    fieldnames = [
        "path",
        "true_label",
        "error_count",
        "run_count",
        "stability_tier",
        *run_fields,
        "pred_labels",
        "confidences",
        "max_confidence",
        "mean_confidence",
        "same_wrong_label",
        "error_pairs",
        "diagnostic_category",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            for key in ("max_confidence", "mean_confidence"):
                payload[key] = f"{float(payload[key]):.6f}"
            for short_id, _ in runs:
                key = f"confidence_{short_id}"
                payload[key] = f"{float(payload[key]):.6f}"
            writer.writerow(payload)


def counter_table(counter, headers):
    lines = [
        f"| {headers[0]} | {headers[1]} |",
        "| --- | ---: |",
    ]
    for label, count in counter.most_common():
        lines.append(f"| `{label}` | {count} |")
    return lines


def write_markdown(path, rows, run_paths, csv_path):
    stability_counts = Counter(row["stability_tier"] for row in rows)
    category_counts = Counter(row["diagnostic_category"] for row in rows)
    pair_counts = Counter()
    for row in rows:
        for pair in row["error_pairs"].split(";"):
            if pair:
                pair_counts[pair] += 1
    stable_high_conf = [
        row for row in rows if row["diagnostic_category"] == "stable_high_conf_error"
    ][:10]
    lines = [
        "# EfficientNet-B0 Stable Error Pool",
        "",
        "Generated from comparable fixed-split EfficientNet-B0 Experiment Runs.",
        "",
        f"- CSV: `{relative(csv_path)}`",
        f"- Error samples: `{len(rows)}`",
        "",
        "## Input Runs",
        "",
        "| Short ID | Run | Predictions |",
        "| --- | --- | --- |",
    ]
    for short_id, run_dir, predictions_path in run_paths:
        lines.append(
            f"| `{short_id}` | `{relative(run_dir)}` | "
            f"`{relative(predictions_path)}` |"
        )

    lines.extend(["", "## Stability Tiers", ""])
    lines.extend(counter_table(stability_counts, ("Tier", "Samples")))
    lines.extend(["", "## Diagnostic Categories", ""])
    lines.extend(counter_table(category_counts, ("Category", "Samples")))
    lines.extend(["", "## Error Pairs", ""])
    lines.extend(counter_table(pair_counts, ("Error Pair", "Samples")))
    lines.extend(["", "## Stable High-Confidence Errors", ""])
    if stable_high_conf:
        lines.extend(
            [
                "| Mean Confidence | Error Count | Error Pairs | Path |",
                "| ---: | ---: | --- | --- |",
            ]
        )
        for row in stable_high_conf:
            lines.append(
                f"| {float(row['mean_confidence']):.4f} | "
                f"{row['error_count']} | "
                f"`{row['error_pairs']}` | "
                f"`{row['path']}` |"
            )
    else:
        lines.append("No `stable_high_conf_error` samples found.")

    lines.extend(
        [
            "",
            "## Related Evidence",
            "",
            "- Manual review snapshot: "
            "`docs/experiments/reviews/"
            "efficientnet-b0-seed42-bs32-w4-high-conf-errors.csv`",
            "- The manual review snapshot is not used for automatic categories.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if not 0 <= args.high_confidence_threshold <= 1:
        print("--high-confidence-threshold must be between 0 and 1", file=sys.stderr)
        return 2

    runs = args.run or DEFAULT_RUNS
    short_ids = [short_id for short_id, _ in runs]
    duplicate_short_ids = [
        short_id for short_id, count in Counter(short_ids).items() if count > 1
    ]
    if duplicate_short_ids:
        print(
            f"duplicate run short IDs: {', '.join(sorted(duplicate_short_ids))}",
            file=sys.stderr,
        )
        return 2

    run_maps = []
    run_paths = []
    try:
        for short_id, run in runs:
            run_dir = resolve_run_dir(run)
            predictions_path, rows = read_predictions(run_dir)
            run_maps.append((short_id, rows))
            run_paths.append((short_id, run_dir, predictions_path))
        validate_comparable(run_maps)
        rows = summarize_rows(runs, run_maps, args.high_confidence_threshold)
    except (FileNotFoundError, ValueError) as exc:
        print(f"stable error pool failed: {exc}", file=sys.stderr)
        return 1

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    csv_path = output_dir / f"{args.stem}.csv"
    md_path = output_dir / f"{args.stem}.md"
    write_csv(csv_path, rows, runs)
    write_markdown(md_path, rows, run_paths, csv_path)

    print("stable error pool OK")
    print(f"runs: {len(runs)}")
    print(f"error_samples: {len(rows)}")
    print(f"csv: {relative(csv_path)}")
    print(f"markdown: {relative(md_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
