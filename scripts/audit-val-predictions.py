#!/usr/bin/env python3
"""Create a compact audit pack from validation prediction errors."""

import argparse
import csv
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "results/runs"


def relative(path):
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_run_dir(value):
    path = Path(value)
    if not path.is_absolute():
        direct = REPO_ROOT / path
        if direct.exists():
            return direct
        run_dir = RUNS_DIR / value
        if run_dir.exists():
            return run_dir
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit high-confidence validation errors from an Experiment Run."
    )
    parser.add_argument(
        "run",
        help="Experiment Run ID or path containing val_predictions.csv.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.8,
        help="Minimum predicted confidence for an error to enter the audit set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=80,
        help="Maximum rows to write after sorting by confidence. Use 0 for all rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <run>/audit.",
    )
    parser.add_argument(
        "--contact-sheet",
        action="store_true",
        help="Also write a JPEG contact sheet. Requires cv2 and numpy.",
    )
    parser.add_argument("--sheet-cols", type=int, default=5)
    parser.add_argument("--thumb-size", type=int, default=180)
    return parser.parse_args()


def read_predictions(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_errors(rows, min_confidence, limit):
    selected = [
        row
        for row in rows
        if row.get("correct") == "0" and float(row["confidence"]) >= min_confidence
    ]
    selected.sort(key=lambda row: float(row["confidence"]), reverse=True)
    if limit > 0:
        selected = selected[:limit]
    return selected


def write_csv(path, rows, labels):
    fieldnames = [
        "path",
        "true_label",
        "pred_label",
        "confidence",
        "review_label",
        "review_notes",
    ] + [f"logit_{label}" for label in labels]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {
                "path": row["path"],
                "true_label": row["true_label"],
                "pred_label": row["pred_label"],
                "confidence": row["confidence"],
                "review_label": "",
                "review_notes": "",
            }
            for label in labels:
                payload[f"logit_{label}"] = row.get(f"logit_{label}", "")
            writer.writerow(payload)


def write_markdown(path, run_dir, rows, min_confidence, csv_path, sheet_path):
    lines = [
        "# Validation Error Audit",
        "",
        f"- Run: `{relative(run_dir)}`",
        f"- Minimum confidence: `{min_confidence}`",
        f"- Selected errors: `{len(rows)}`",
        f"- CSV: `{relative(csv_path)}`",
    ]
    if sheet_path is not None:
        lines.append(f"- Contact sheet: `{relative(sheet_path)}`")
    lines.extend(
        [
            "",
            "| Confidence | Error | Path |",
            "| --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{float(row['confidence']):.4f} | "
            f"`{row['true_label']} -> {row['pred_label']}` | "
            f"`{row['path']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_image(path):
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image


def write_contact_sheet(path, rows, cols, thumb_size):
    import cv2
    import numpy as np

    if not rows:
        return

    label_height = 44
    gap = 8
    rows_count = math.ceil(len(rows) / cols)
    tile_height = thumb_size + label_height
    sheet_height = rows_count * tile_height + (rows_count + 1) * gap
    sheet_width = cols * thumb_size + (cols + 1) * gap
    sheet = np.full((sheet_height, sheet_width, 3), 245, dtype=np.uint8)

    for index, row in enumerate(rows):
        image_path = REPO_ROOT / row["path"]
        image = read_image(image_path)
        image = cv2.resize(
            image, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA
        )
        grid_row, grid_col = divmod(index, cols)
        y = gap + grid_row * (tile_height + gap)
        x = gap + grid_col * (thumb_size + gap)
        sheet[y : y + thumb_size, x : x + thumb_size] = image

        text_y = y + thumb_size + 16
        error_text = f"{row['true_label']}->{row['pred_label']}"
        conf_text = f"{float(row['confidence']):.3f}"
        cv2.putText(
            sheet,
            error_text,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            conf_text,
            (x, text_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"failed to write contact sheet: {path}")


def main():
    args = parse_args()
    if not 0 <= args.min_confidence <= 1:
        print("--min-confidence must be between 0 and 1", file=sys.stderr)
        return 2
    if args.limit < 0:
        print("--limit must be non-negative", file=sys.stderr)
        return 2
    if args.sheet_cols < 1 or args.thumb_size < 32:
        print("--sheet-cols must be positive and --thumb-size must be at least 32")
        return 2

    run_dir = resolve_run_dir(args.run)
    predictions_path = run_dir / "val_predictions.csv"
    if not predictions_path.is_file():
        print(
            f"missing val_predictions.csv: {relative(predictions_path)}",
            file=sys.stderr,
        )
        return 1

    output_dir = args.output_dir or run_dir / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_predictions(predictions_path)
    labels = [
        field.removeprefix("logit_") for field in rows[0] if field.startswith("logit_")
    ]
    selected = select_errors(rows, args.min_confidence, args.limit)

    csv_path = output_dir / "high_conf_errors.csv"
    md_path = output_dir / "high_conf_errors.md"
    sheet_path = output_dir / "high_conf_errors.jpg" if args.contact_sheet else None
    write_csv(csv_path, selected, labels)
    if sheet_path is not None:
        write_contact_sheet(sheet_path, selected, args.sheet_cols, args.thumb_size)
    write_markdown(
        md_path, run_dir, selected, args.min_confidence, csv_path, sheet_path
    )

    print("audit OK")
    print(f"run: {relative(run_dir)}")
    print(f"selected_errors: {len(selected)}")
    print(f"csv: {relative(csv_path)}")
    print(f"markdown: {relative(md_path)}")
    if sheet_path is not None:
        print(f"contact_sheet: {relative(sheet_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
