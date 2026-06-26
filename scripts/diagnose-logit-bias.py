#!/usr/bin/env python3
"""Diagnose class-wise logit bias on validation predictions."""

import argparse
import csv
import itertools
import math
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "results/runs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs/diagnostics"
DEFAULT_STEM = "efficientnet-b0-logit-bias"
DEFAULT_RUNS = [
    ("seed42", "efficientnet-b0-seed42-bs32-w4"),
    ("train2025", "efficientnet-b0-split42-train2025-bs32-w4"),
    ("train3407", "efficientnet-b0-split42-train3407-bs32-w4"),
]


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
        description="Grid-search class-wise logit bias on validation predictions."
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        default=None,
        help=(
            "Run as short_id=run_id_or_path. May be repeated. Defaults to "
            "the fixed-split B0 runs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated CSV and Markdown reports.",
    )
    parser.add_argument(
        "--stem",
        default=DEFAULT_STEM,
        help="Output filename stem without extension.",
    )
    parser.add_argument("--min-bias", type=float, default=-1.0)
    parser.add_argument("--max-bias", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument(
        "--fixed-label",
        default="cloudy",
        help="Label whose bias is fixed at 0 to remove translation redundancy.",
    )
    parser.add_argument(
        "--top-limit",
        type=int,
        default=10,
        help="Maximum changed rows shown in each Markdown detail table.",
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
        raise FileNotFoundError(f"missing val_predictions.csv: {relative(path)}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows:
        raise ValueError(f"empty val_predictions.csv: {relative(path)}")
    labels = [
        field.removeprefix("logit_")
        for field in fieldnames
        if field.startswith("logit_")
    ]
    if not labels:
        raise ValueError(f"missing logit columns: {relative(path)}")
    required = {"path", "true_label", "pred_label", "correct", "confidence"}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(
            f"{relative(path)} is missing required fields: {', '.join(sorted(missing))}"
        )
    label_to_index = {label: index for index, label in enumerate(labels)}
    examples = []
    for row in rows:
        try:
            true_index = label_to_index[row["true_label"]]
            logits = [float(row[f"logit_{label}"]) for label in labels]
        except KeyError as exc:
            raise ValueError(f"unknown label in {relative(path)}: {exc}") from exc
        examples.append(
            {
                "path": row["path"],
                "true_label": row["true_label"],
                "true_index": true_index,
                "csv_pred_label": row["pred_label"],
                "csv_correct": int(row["correct"]),
                "csv_confidence": float(row["confidence"]),
                "logits": logits,
            }
        )
    return path, labels, examples


def bias_values(min_bias, max_bias, step):
    count = int(round((max_bias - min_bias) / step))
    values = [round(min_bias + index * step, 10) for index in range(count + 1)]
    if not values or values[-1] < max_bias - (step / 2):
        values.append(max_bias)
    return values


def argmax_with_bias(logits, bias):
    best_index = 0
    best_value = logits[0] + bias[0]
    for index in range(1, len(logits)):
        value = logits[index] + bias[index]
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def confusion_matrix(y_true, y_pred, class_count):
    matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for true_index, pred_index in zip(y_true, y_pred):
        matrix[true_index][pred_index] += 1
    return matrix


def per_class_f1_from_matrix(matrix):
    class_count = len(matrix)
    scores = []
    for index in range(class_count):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(class_count) if row != index)
        fn = sum(matrix[index][col] for col in range(class_count) if col != index)
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return scores


def macro_f1(y_true, y_pred, class_count):
    matrix = confusion_matrix(y_true, y_pred, class_count)
    scores = per_class_f1_from_matrix(matrix)
    return sum(scores) / len(scores), scores, matrix


def predictions_for_bias(examples, bias):
    return [argmax_with_bias(example["logits"], bias) for example in examples]


def evaluate_bias(examples, bias, class_count):
    y_true = [example["true_index"] for example in examples]
    y_pred = predictions_for_bias(examples, bias)
    score, scores, matrix = macro_f1(y_true, y_pred, class_count)
    return {
        "macro_f1": score,
        "per_class_f1": scores,
        "confusion_matrix": matrix,
        "predictions": y_pred,
    }


def search_bias(examples, labels, args):
    if args.fixed_label not in labels:
        raise ValueError(f"--fixed-label must be one of: {', '.join(labels)}")
    if args.step <= 0:
        raise ValueError("--step must be positive")
    if args.min_bias > args.max_bias:
        raise ValueError("--min-bias must be <= --max-bias")

    class_count = len(labels)
    fixed_index = labels.index(args.fixed_label)
    values = bias_values(args.min_bias, args.max_bias, args.step)
    search_indices = [index for index in range(class_count) if index != fixed_index]
    best_bias = [0.0] * class_count
    best_score = -1.0
    best_predictions = None
    y_true = [example["true_index"] for example in examples]

    for combination in itertools.product(values, repeat=len(search_indices)):
        bias = [0.0] * class_count
        for index, value in zip(search_indices, combination):
            bias[index] = value
        y_pred = predictions_for_bias(examples, bias)
        score, _, _ = macro_f1(y_true, y_pred, class_count)
        if score > best_score:
            best_score = score
            best_bias = bias
            best_predictions = y_pred

    best_eval = evaluate_predictions(y_true, best_predictions, class_count)
    best_eval["bias"] = best_bias
    return best_eval


def evaluate_predictions(y_true, y_pred, class_count):
    score, scores, matrix = macro_f1(y_true, y_pred, class_count)
    return {
        "macro_f1": score,
        "per_class_f1": scores,
        "confusion_matrix": matrix,
        "predictions": y_pred,
    }


def softmax_confidence(logits, bias, pred_index):
    shifted = [logit + offset for logit, offset in zip(logits, bias)]
    max_value = max(shifted)
    exp_values = [math.exp(value - max_value) for value in shifted]
    denom = sum(exp_values)
    return exp_values[pred_index] / denom


def matrix_to_string(matrix):
    return ";".join(",".join(str(value) for value in row) for row in matrix)


def bias_to_string(labels, bias):
    return ";".join(f"{label}:{value:.4f}" for label, value in zip(labels, bias))


def summarize_changes(short_id, labels, examples, baseline_eval, biased_eval):
    rows = []
    baseline_predictions = baseline_eval["predictions"]
    biased_predictions = biased_eval["predictions"]
    bias = biased_eval["bias"]
    for example, before_index, after_index in zip(
        examples,
        baseline_predictions,
        biased_predictions,
    ):
        if before_index == after_index:
            continue
        correct_before = int(before_index == example["true_index"])
        correct_after = int(after_index == example["true_index"])
        if correct_before == 0 and correct_after == 1:
            change_type = "corrected"
        elif correct_before == 1 and correct_after == 0:
            change_type = "regressed"
        else:
            change_type = "changed_wrong_to_wrong"
        rows.append(
            {
                "run": short_id,
                "path": example["path"],
                "true_label": example["true_label"],
                "pred_before": labels[before_index],
                "pred_after": labels[after_index],
                "correct_before": correct_before,
                "correct_after": correct_after,
                "confidence_before": example["csv_confidence"],
                "confidence_after": softmax_confidence(
                    example["logits"],
                    bias,
                    after_index,
                ),
                "change_type": change_type,
            }
        )
    return rows


def summary_row(short_id, labels, baseline_eval, biased_eval, changes):
    counts = Counter(row["change_type"] for row in changes)
    row = {
        "run": short_id,
        "baseline_macro_f1": baseline_eval["macro_f1"],
        "biased_macro_f1": biased_eval["macro_f1"],
        "delta_macro_f1": biased_eval["macro_f1"] - baseline_eval["macro_f1"],
        "changed_count": len(changes),
        "corrected_count": counts["corrected"],
        "regressed_count": counts["regressed"],
        "changed_wrong_to_wrong_count": counts["changed_wrong_to_wrong"],
        "baseline_confusion_matrix": matrix_to_string(
            baseline_eval["confusion_matrix"]
        ),
        "biased_confusion_matrix": matrix_to_string(biased_eval["confusion_matrix"]),
    }
    for label, value in zip(labels, biased_eval["bias"]):
        row[f"bias_{label}"] = value
    for label, value in zip(labels, baseline_eval["per_class_f1"]):
        row[f"baseline_f1_{label}"] = value
    for label, value in zip(labels, biased_eval["per_class_f1"]):
        row[f"biased_f1_{label}"] = value
    return row


def write_summary_csv(path, rows, labels):
    fieldnames = [
        "run",
        "baseline_macro_f1",
        "biased_macro_f1",
        "delta_macro_f1",
        *[f"bias_{label}" for label in labels],
        *[f"baseline_f1_{label}" for label in labels],
        *[f"biased_f1_{label}" for label in labels],
        "changed_count",
        "corrected_count",
        "regressed_count",
        "changed_wrong_to_wrong_count",
        "baseline_confusion_matrix",
        "biased_confusion_matrix",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {key: value for key, value in row.items() if not key.startswith("_")}
            for key, value in list(payload.items()):
                if isinstance(value, float):
                    payload[key] = f"{value:.6f}"
            writer.writerow(payload)


def write_changes_csv(path, rows):
    fieldnames = [
        "run",
        "path",
        "true_label",
        "pred_before",
        "pred_after",
        "correct_before",
        "correct_after",
        "confidence_before",
        "confidence_after",
        "change_type",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["confidence_before"] = f"{row['confidence_before']:.6f}"
            payload["confidence_after"] = f"{row['confidence_after']:.6f}"
            writer.writerow(payload)


def markdown_matrix(matrix, labels):
    lines = [
        "| True \\ Pred | " + " | ".join(f"`{label}`" for label in labels) + " |",
        "| --- | " + " | ".join("---:" for _ in labels) + " |",
    ]
    for label, row in zip(labels, matrix):
        lines.append(
            f"| `{label}` | " + " | ".join(str(value) for value in row) + " |"
        )
    return lines


def top_changes(changes, change_type, limit):
    rows = [row for row in changes if row["change_type"] == change_type]
    rows.sort(
        key=lambda row: (
            row["run"],
            -float(row["confidence_after"]),
            row["path"],
        )
    )
    return rows[:limit]


def changes_table(changes, title):
    lines = [
        f"## {title}",
        "",
        "| Run | True | Before | After | Confidence After | Path |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    if not changes:
        lines.append("|  |  |  |  |  |  |")
        return lines
    for row in changes:
        lines.append(
            f"| `{row['run']}` | "
            f"`{row['true_label']}` | "
            f"`{row['pred_before']}` | "
            f"`{row['pred_after']}` | "
            f"{row['confidence_after']:.4f} | "
            f"`{row['path']}` |"
        )
    return lines


def write_markdown(path, labels, summary_rows, changes, outputs, args):
    lines = [
        "# EfficientNet-B0 Logit Bias Diagnostic",
        "",
        "Class-wise logit bias was selected on the same Internal Validation "
        "Split used for scoring each row. Treat this as an in-split diagnostic "
        "upper bound for decision calibration, not as a Submission Prediction "
        "Interface parameter.",
        "",
        f"- Summary CSV: `{relative(outputs['summary'])}`",
        f"- Changes CSV: `{relative(outputs['changes'])}`",
        f"- Fixed label: `{args.fixed_label}`",
        f"- Bias range: `{args.min_bias}` to `{args.max_bias}`",
        f"- Bias step: `{args.step}`",
        "",
        "## Run Summary",
        "",
        "| Run | Baseline Macro F1 | Biased Macro F1 | Delta | Bias | "
        "Changed | Corrected | Regressed | Wrong->Wrong |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        bias = bias_to_string(labels, [row[f"bias_{label}"] for label in labels])
        lines.append(
            f"| `{row['run']}` | "
            f"{row['baseline_macro_f1']:.4f} | "
            f"{row['biased_macro_f1']:.4f} | "
            f"{row['delta_macro_f1']:.4f} | "
            f"`{bias}` | "
            f"{row['changed_count']} | "
            f"{row['corrected_count']} | "
            f"{row['regressed_count']} | "
            f"{row['changed_wrong_to_wrong_count']} |"
        )

    lines.extend(["", "## Per-Class F1", ""])
    header = "| Run | State | " + " | ".join(f"`{label}`" for label in labels) + " |"
    divider = "| --- | --- | " + " | ".join("---:" for _ in labels) + " |"
    lines.extend([header, divider])
    for row in summary_rows:
        baseline_values = " | ".join(
            f"{row[f'baseline_f1_{label}']:.4f}" for label in labels
        )
        biased_values = " | ".join(
            f"{row[f'biased_f1_{label}']:.4f}" for label in labels
        )
        lines.append(f"| `{row['run']}` | baseline | {baseline_values} |")
        lines.append(f"| `{row['run']}` | biased | {biased_values} |")

    for row in summary_rows:
        lines.extend(["", f"## Confusion Matrices: `{row['run']}`", ""])
        lines.append("Baseline:")
        lines.extend(markdown_matrix(row["_baseline_matrix"], labels))
        lines.extend(["", "Biased:"])
        lines.extend(markdown_matrix(row["_biased_matrix"], labels))

    for change_type, title in [
        ("corrected", "Top Corrected Rows"),
        ("regressed", "Top Regressed Rows"),
        ("changed_wrong_to_wrong", "Top Wrong-to-Wrong Changes"),
    ]:
        lines.extend([""])
        lines.extend(changes_table(top_changes(changes, change_type, args.top_limit), title))

    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- Bias values were selected on the same Internal Validation Split.",
            "- This report is a diagnostic upper bound for decision calibration.",
            "- Do not use these bias values in the Submission Prediction Interface "
            "without cross-run or split validation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_diagnostic(short_id, run_dir, args):
    predictions_path, labels, examples = read_predictions(run_dir)
    baseline_bias = [0.0] * len(labels)
    baseline_eval = evaluate_bias(examples, baseline_bias, len(labels))
    biased_eval = search_bias(examples, labels, args)
    changes = summarize_changes(short_id, labels, examples, baseline_eval, biased_eval)
    row = summary_row(short_id, labels, baseline_eval, biased_eval, changes)
    row["_baseline_matrix"] = baseline_eval["confusion_matrix"]
    row["_biased_matrix"] = biased_eval["confusion_matrix"]
    return predictions_path, labels, row, changes


def main():
    args = parse_args()
    if args.top_limit < 1:
        print("--top-limit must be at least 1", file=sys.stderr)
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

    summary_rows = []
    all_changes = []
    expected_labels = None
    try:
        for short_id, run in runs:
            _, labels, row, changes = run_diagnostic(
                short_id,
                resolve_run_dir(run),
                args,
            )
            if expected_labels is None:
                expected_labels = labels
            elif labels != expected_labels:
                raise ValueError(
                    f"label mismatch for {short_id}: {labels}; "
                    f"expected {expected_labels}"
                )
            summary_rows.append(row)
            all_changes.extend(changes)
    except (FileNotFoundError, ValueError) as exc:
        print(f"logit bias diagnostic failed: {exc}", file=sys.stderr)
        return 1

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    outputs = {
        "summary": output_dir / f"{args.stem}-summary.csv",
        "changes": output_dir / f"{args.stem}-changes.csv",
        "markdown": output_dir / f"{args.stem}.md",
    }
    write_summary_csv(outputs["summary"], summary_rows, expected_labels)
    write_changes_csv(outputs["changes"], all_changes)
    write_markdown(outputs["markdown"], expected_labels, summary_rows, all_changes, outputs, args)

    print("logit bias diagnostic OK")
    print(f"runs: {len(summary_rows)}")
    print(f"changed_rows: {len(all_changes)}")
    print(f"summary: {relative(outputs['summary'])}")
    print(f"changes: {relative(outputs['changes'])}")
    print(f"markdown: {relative(outputs['markdown'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
