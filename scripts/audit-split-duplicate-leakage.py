#!/usr/bin/env python3
"""Audit duplicate-group leakage across a train/validation split."""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = REPO_ROOT / "docs/diagnostics/train-image-features.csv"
DEFAULT_VAL_PREDICTIONS = (
    REPO_ROOT / "results/runs/efficientnet-b0-seed42-bs32-w4/val_predictions.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs/diagnostics"
DEFAULT_STEM = "split42-duplicate-leakage"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit exact and near-duplicate dHash groups that cross an Internal "
            "Validation Split boundary."
        )
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES,
        help="Full Training Set image feature CSV from train-image-features.",
    )
    parser.add_argument(
        "--val-predictions",
        type=Path,
        default=DEFAULT_VAL_PREDICTIONS,
        help="Experiment Run val_predictions.csv used as the split membership source.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated CSV and Markdown report.",
    )
    parser.add_argument(
        "--stem",
        default=DEFAULT_STEM,
        help="Output filename stem without extension.",
    )
    parser.add_argument(
        "--top-limit",
        type=int,
        default=10,
        help="Maximum rows shown in each Markdown detail table.",
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


def read_csv_rows(path, required_fields):
    if not path.is_file():
        raise FileNotFoundError(f"missing CSV: {relative(path)}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty CSV: {relative(path)}")
    missing = [field for field in required_fields if field not in reader.fieldnames]
    if missing:
        raise ValueError(
            f"{relative(path)} is missing required fields: {', '.join(missing)}"
        )
    return rows


def read_features(path):
    return read_csv_rows(
        path,
        [
            "path",
            "true_label",
            "read_ok",
            "dhash",
            "exact_dhash_group_size",
            "near_duplicate_group_id",
        ],
    )


def read_val_predictions(path):
    rows = read_csv_rows(path, ["path", "true_label", "pred_label", "correct"])
    return {row["path"]: row for row in rows}


def labels_text(rows):
    labels = Counter(row["true_label"] for row in rows)
    return "; ".join(f"{label}:{count}" for label, count in labels.most_common())


def summarize_group(group_type, group_id, rows, val_predictions):
    val_rows = [row for row in rows if row["path"] in val_predictions]
    train_rows = [row for row in rows if row["path"] not in val_predictions]
    val_error_count = sum(
        int(val_predictions[row["path"]]["correct"]) == 0 for row in val_rows
    )
    label_count = len({row["true_label"] for row in rows})
    return {
        "group_type": group_type,
        "group_id": group_id,
        "group_size": len(rows),
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "split_leak": int(bool(train_rows and val_rows)),
        "cross_label": int(label_count > 1),
        "val_error_count": val_error_count,
        "labels": labels_text(rows),
        "val_paths": "; ".join(row["path"] for row in val_rows),
        "train_paths": "; ".join(row["path"] for row in train_rows),
    }


def build_group_summaries(feature_rows, val_predictions):
    groups = []
    exact_groups = defaultdict(list)
    near_groups = defaultdict(list)
    for row in feature_rows:
        if row["read_ok"] != "1":
            continue
        if int(row["exact_dhash_group_size"] or 0) > 1:
            exact_groups[row["dhash"]].append(row)
        near_group_id = row["near_duplicate_group_id"]
        if near_group_id:
            near_groups[near_group_id].append(row)

    for group_id, rows in sorted(exact_groups.items()):
        groups.append(summarize_group("exact_dhash", group_id, rows, val_predictions))
    for group_id, rows in sorted(near_groups.items()):
        groups.append(
            summarize_group("near_duplicate", group_id, rows, val_predictions)
        )
    return groups


def split_summary(groups, group_type):
    selected = [group for group in groups if group["group_type"] == group_type]
    leaked = [group for group in selected if group["split_leak"]]
    return {
        "group_type": group_type,
        "groups": len(selected),
        "samples": sum(group["group_size"] for group in selected),
        "leaked_groups": len(leaked),
        "leaked_samples": sum(group["group_size"] for group in leaked),
        "leaked_val_samples": sum(group["val_count"] for group in leaked),
        "leaked_val_errors": sum(group["val_error_count"] for group in leaked),
        "cross_label_leaked_groups": sum(group["cross_label"] for group in leaked),
    }


def write_csv(path, groups):
    fieldnames = [
        "group_type",
        "group_id",
        "group_size",
        "train_count",
        "val_count",
        "split_leak",
        "cross_label",
        "val_error_count",
        "labels",
        "val_paths",
        "train_paths",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            writer.writerow(group)


def markdown_summary_table(summaries):
    lines = [
        "## Duplicate Group Split Summary",
        "",
        "| Group Type | Groups | Samples | Leaked Groups | Leaked Samples | "
        "Leaked Val Samples | Leaked Val Errors | Cross-Label Leaked Groups |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            f"| `{summary['group_type']}` | "
            f"{summary['groups']} | "
            f"{summary['samples']} | "
            f"{summary['leaked_groups']} | "
            f"{summary['leaked_samples']} | "
            f"{summary['leaked_val_samples']} | "
            f"{summary['leaked_val_errors']} | "
            f"{summary['cross_label_leaked_groups']} |"
        )
    return lines


def detail_table(groups, title, top_limit):
    lines = [
        f"## {title}",
        "",
        "| Group Type | Group | Size | Train | Val | Val Errors | Cross Label | "
        "Labels | Example Val Paths | Example Train Paths |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    if not groups:
        lines.append("|  |  | 0 | 0 | 0 | 0 | 0 |  |  |  |")
        return lines
    for group in groups[:top_limit]:
        val_paths = "<br>".join(
            f"`{path}`" for path in group["val_paths"].split("; ")[:3] if path
        )
        train_paths = "<br>".join(
            f"`{path}`" for path in group["train_paths"].split("; ")[:3] if path
        )
        lines.append(
            f"| `{group['group_type']}` | "
            f"`{group['group_id']}` | "
            f"{group['group_size']} | "
            f"{group['train_count']} | "
            f"{group['val_count']} | "
            f"{group['val_error_count']} | "
            f"{group['cross_label']} | "
            f"`{group['labels']}` | "
            f"{val_paths} | "
            f"{train_paths} |"
        )
    return lines


def rank_groups(groups):
    return sorted(
        groups,
        key=lambda group: (
            -group["val_error_count"],
            -group["val_count"],
            -group["group_size"],
            group["group_id"],
        ),
    )


def write_markdown(path, groups, feature_rows, val_predictions, csv_path, args):
    summaries = [
        split_summary(groups, "exact_dhash"),
        split_summary(groups, "near_duplicate"),
    ]
    leaked_exact = rank_groups(
        [
            group
            for group in groups
            if group["group_type"] == "exact_dhash" and group["split_leak"]
        ]
    )
    leaked_near = rank_groups(
        [
            group
            for group in groups
            if group["group_type"] == "near_duplicate" and group["split_leak"]
        ]
    )
    leaked_cross_label = rank_groups(
        [group for group in groups if group["split_leak"] and group["cross_label"]]
    )
    lines = [
        "# Split Duplicate Leakage Diagnostic",
        "",
        "Duplicate-group diagnostics for an Internal Validation Split. dHash groups "
        "are coarse perceptual similarity candidates; a split leak is evidence of "
        "possible local validation optimism, not proof of duplicate source images "
        "or label errors.",
        "",
        f"- Features CSV: `{relative(resolve_path(args.features))}`",
        f"- Validation predictions: `{relative(resolve_path(args.val_predictions))}`",
        f"- CSV: `{relative(csv_path)}`",
        f"- Training Set rows: `{len(feature_rows)}`",
        f"- Internal Validation Split rows: `{len(val_predictions)}`",
        f"- Inferred training rows: `{len(feature_rows) - len(val_predictions)}`",
        "- Validation error counts are model-specific values from the provided "
        "`val_predictions.csv` file.",
        "",
    ]
    lines.extend(markdown_summary_table(summaries))
    lines.extend([""])
    lines.extend(
        detail_table(
            leaked_exact,
            "Leaked Exact dHash Groups",
            args.top_limit,
        )
    )
    lines.extend([""])
    lines.extend(
        detail_table(
            leaked_near,
            "Leaked Near-Duplicate Groups",
            args.top_limit,
        )
    )
    lines.extend([""])
    lines.extend(
        detail_table(
            leaked_cross_label,
            "Leaked Cross-Label Duplicate Groups",
            args.top_limit,
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.top_limit < 1:
        print("--top-limit must be at least 1", file=sys.stderr)
        return 2

    features_path = resolve_path(args.features)
    val_predictions_path = resolve_path(args.val_predictions)
    output_dir = resolve_path(args.output_dir)
    try:
        feature_rows = read_features(features_path)
        val_predictions = read_val_predictions(val_predictions_path)
        feature_paths = {row["path"] for row in feature_rows}
        unknown_val_paths = sorted(set(val_predictions) - feature_paths)
        if unknown_val_paths:
            example = ", ".join(unknown_val_paths[:3])
            raise ValueError(
                "validation predictions contain paths missing from features CSV: "
                + example
            )
        groups = build_group_summaries(feature_rows, val_predictions)
    except (FileNotFoundError, ValueError) as exc:
        print(f"split duplicate leakage audit failed: {exc}", file=sys.stderr)
        return 1

    csv_path = output_dir / f"{args.stem}.csv"
    md_path = output_dir / f"{args.stem}.md"
    write_csv(csv_path, groups)
    write_markdown(md_path, groups, feature_rows, val_predictions, csv_path, args)

    summaries = {
        summary["group_type"]: summary
        for summary in [
            split_summary(groups, "exact_dhash"),
            split_summary(groups, "near_duplicate"),
        ]
    }
    print("split duplicate leakage audit OK")
    for group_type in ["exact_dhash", "near_duplicate"]:
        summary = summaries[group_type]
        print(
            f"{group_type}: leaked_groups={summary['leaked_groups']} "
            f"leaked_val_samples={summary['leaked_val_samples']}"
        )
    print(f"csv: {relative(csv_path)}")
    print(f"markdown: {relative(md_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
