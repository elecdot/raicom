#!/usr/bin/env python3
"""Compare Experiment Run metrics and validation predictions."""

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = REPO_ROOT / "results/runs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs/diagnostics"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Experiment Run metrics and validation predictions."
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help="Experiment Run IDs or paths under results/runs/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for generated Markdown and CSV reports.",
    )
    parser.add_argument(
        "--stem",
        default=None,
        help="Output filename stem when --output-dir is used.",
    )
    return parser.parse_args()


def relative(path):
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_path(path):
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_run_dir(run):
    candidate = Path(run)
    if candidate.is_absolute() or len(candidate.parts) > 1:
        run_dir = resolve_path(candidate)
    else:
        run_dir = DEFAULT_RUNS_DIR / run
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
    required_fields = {"path", "true_label", "pred_label", "correct", "confidence"}
    missing = sorted(required_fields - set(reader.fieldnames or []))
    if missing:
        raise ValueError(
            f"{relative(path)} is missing required fields: {', '.join(missing)}"
        )
    return rows


def load_run(run):
    run_dir = resolve_run_dir(run)
    metadata = read_json(run_dir / "metadata.json")
    metrics = read_json(run_dir / "metrics.json")
    predictions = read_predictions(run_dir / "val_predictions.csv")
    run_id = metadata.get("run_id") or run_dir.name
    return {
        "run": run,
        "run_id": run_id,
        "run_dir": run_dir,
        "metadata": metadata,
        "metrics": metrics,
        "predictions": predictions,
        "predictions_by_path": {row["path"]: row for row in predictions},
    }


def optional_float(value):
    if value is None:
        return ""
    return f"{float(value):.4f}"


def summary_row(run):
    metadata = run["metadata"]
    metrics = run["metrics"]
    best = metrics["best"]
    return {
        "run_id": run["run_id"],
        "model_candidate": metadata.get("model_candidate", ""),
        "split_strategy": metadata.get("split_strategy", ""),
        "augmentation": metadata.get("augmentation", ""),
        "class_weighting": metadata.get("class_weighting", ""),
        "label_smoothing": metadata.get("label_smoothing", ""),
        "val_rows": len(run["predictions"]),
        "best_macro_f1": best.get("val_macro_f1"),
        "best_accuracy": best.get("val_acc"),
        "best_epoch": metrics.get("best_epoch", ""),
        "final_macro_f1": metadata.get("final_val_macro_f1", ""),
    }


def pairwise_row(baseline, candidate):
    baseline_predictions = baseline["predictions_by_path"]
    candidate_predictions = candidate["predictions_by_path"]
    baseline_paths = set(baseline_predictions)
    candidate_paths = set(candidate_predictions)
    common_paths = baseline_paths & candidate_paths
    same_paths = baseline_paths == candidate_paths
    counts = {
        "baseline_run_id": baseline["run_id"],
        "candidate_run_id": candidate["run_id"],
        "same_val_paths": int(same_paths),
        "common_paths": len(common_paths),
        "baseline_only_paths": len(baseline_paths - candidate_paths),
        "candidate_only_paths": len(candidate_paths - baseline_paths),
        "both_correct": 0,
        "baseline_correct_candidate_wrong": 0,
        "baseline_wrong_candidate_correct": 0,
        "both_wrong": 0,
        "same_prediction": 0,
        "different_prediction": 0,
    }
    for path in common_paths:
        baseline_row = baseline_predictions[path]
        candidate_row = candidate_predictions[path]
        baseline_correct = baseline_row["correct"] == "1"
        candidate_correct = candidate_row["correct"] == "1"
        if baseline_row["pred_label"] == candidate_row["pred_label"]:
            counts["same_prediction"] += 1
        else:
            counts["different_prediction"] += 1
        if baseline_correct and candidate_correct:
            counts["both_correct"] += 1
        elif baseline_correct and not candidate_correct:
            counts["baseline_correct_candidate_wrong"] += 1
        elif not baseline_correct and candidate_correct:
            counts["baseline_wrong_candidate_correct"] += 1
        else:
            counts["both_wrong"] += 1
    return counts


def per_class_delta_rows(baseline, candidate):
    baseline_per_class = baseline["metrics"]["best"]["per_class"]
    candidate_per_class = candidate["metrics"]["best"]["per_class"]
    labels = sorted(set(baseline_per_class) & set(candidate_per_class))
    rows = []
    for label in labels:
        baseline_f1 = float(baseline_per_class[label]["f1"])
        candidate_f1 = float(candidate_per_class[label]["f1"])
        rows.append(
            {
                "baseline_run_id": baseline["run_id"],
                "candidate_run_id": candidate["run_id"],
                "label": label,
                "baseline_f1": baseline_f1,
                "candidate_f1": candidate_f1,
                "delta_f1": candidate_f1 - baseline_f1,
            }
        )
    return rows


def markdown_summary_table(summary_rows):
    lines = [
        "## Run Summary",
        "",
        "| Run | Model | Split | Aug | Class Weights | Smoothing | Val Rows | "
        "Best Macro F1 | Best Acc | Best Epoch | Final Macro F1 |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['run_id']}` | "
            f"`{row['model_candidate']}` | "
            f"`{row['split_strategy']}` | "
            f"`{row['augmentation']}` | "
            f"`{row['class_weighting']}` | "
            f"{float(row['label_smoothing']):.2f} | "
            f"{row['val_rows']} | "
            f"{optional_float(row['best_macro_f1'])} | "
            f"{optional_float(row['best_accuracy'])} | "
            f"{row['best_epoch']} | "
            f"{optional_float(row['final_macro_f1'])} |"
        )
    return lines


def markdown_pairwise_table(pairwise_rows):
    lines = [
        "## Pairwise Against First Run",
        "",
        "| Candidate | Same Val Paths | Common | Baseline Correct / Candidate Wrong | "
        "Baseline Wrong / Candidate Correct | Both Wrong | Different Predictions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not pairwise_rows:
        lines.append("|  | 0 | 0 | 0 | 0 | 0 | 0 |")
        return lines
    for row in pairwise_rows:
        lines.append(
            f"| `{row['candidate_run_id']}` | "
            f"{row['same_val_paths']} | "
            f"{row['common_paths']} | "
            f"{row['baseline_correct_candidate_wrong']} | "
            f"{row['baseline_wrong_candidate_correct']} | "
            f"{row['both_wrong']} | "
            f"{row['different_prediction']} |"
        )
    return lines


def markdown_delta_table(delta_rows):
    lines = [
        "## Per-Class F1 Delta Against First Run",
        "",
        "| Candidate | Label | Baseline F1 | Candidate F1 | Delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    if not delta_rows:
        lines.append("|  |  |  |  |  |")
        return lines
    for row in delta_rows:
        lines.append(
            f"| `{row['candidate_run_id']}` | "
            f"`{row['label']}` | "
            f"{row['baseline_f1']:.4f} | "
            f"{row['candidate_f1']:.4f} | "
            f"{row['delta_f1']:.4f} |"
        )
    return lines


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(output_dir, stem, markdown, summary_rows, pairwise_rows, delta_rows):
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"{stem}.md"
    summary_csv_path = output_dir / f"{stem}-summary.csv"
    pairwise_csv_path = output_dir / f"{stem}-pairwise.csv"
    delta_csv_path = output_dir / f"{stem}-per-class-delta.csv"
    markdown_path.write_text(markdown, encoding="utf-8")
    write_csv(summary_csv_path, summary_rows, list(summary_rows[0]))
    if pairwise_rows:
        write_csv(pairwise_csv_path, pairwise_rows, list(pairwise_rows[0]))
    if delta_rows:
        write_csv(delta_csv_path, delta_rows, list(delta_rows[0]))
    return markdown_path, summary_csv_path, pairwise_csv_path, delta_csv_path


def build_markdown(runs, summary_rows, pairwise_rows, delta_rows):
    lines = [
        "# Experiment Run Comparison",
        "",
        "Comparison generated from Experiment Run `metadata.json`, `metrics.json`, "
        "and `val_predictions.csv` files.",
        "",
        "Input runs:",
    ]
    for run in runs:
        lines.append(f"- `{relative(run['run_dir'])}`")
    lines.extend([""])
    lines.extend(markdown_summary_table(summary_rows))
    lines.extend([""])
    lines.extend(markdown_pairwise_table(pairwise_rows))
    lines.extend([""])
    lines.extend(markdown_delta_table(delta_rows))
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    if args.output_dir and not args.stem:
        print("--stem is required when --output-dir is used", file=sys.stderr)
        return 2
    try:
        runs = [load_run(run) for run in args.runs]
        summary_rows = [summary_row(run) for run in runs]
        baseline = runs[0]
        pairwise_rows = [pairwise_row(baseline, run) for run in runs[1:]]
        delta_rows = []
        for run in runs[1:]:
            delta_rows.extend(per_class_delta_rows(baseline, run))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"run comparison failed: {exc}", file=sys.stderr)
        return 1

    markdown = build_markdown(runs, summary_rows, pairwise_rows, delta_rows)
    if args.output_dir:
        paths = write_outputs(
            args.output_dir,
            args.stem,
            markdown,
            summary_rows,
            pairwise_rows,
            delta_rows,
        )
        print("run comparison OK")
        for path in paths:
            if path.exists():
                print(relative(path))
    else:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
