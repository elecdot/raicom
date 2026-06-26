#!/usr/bin/env python3
"""Create image feature diagnostics for an Internal Validation Split."""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = (
    REPO_ROOT
    / "results/runs/efficientnet-b0-seed42-bs32-w4/val_predictions.csv"
)
DEFAULT_STABLE_POOL = (
    REPO_ROOT / "docs/diagnostics/efficientnet-b0-stable-error-pool.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs/diagnostics"
DEFAULT_STEM = "efficientnet-b0-val-image-features"
FEATURE_COLUMNS = [
    "brightness_mean",
    "brightness_std",
    "saturation_mean",
    "saturation_std",
    "contrast",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
    "low_saturation_ratio",
    "edge_density",
    "laplacian_variance",
]


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value):
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
        elif self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1


def relative(path):
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit image features for a validation prediction file."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS,
        help="Validation predictions CSV for the target Internal Validation Split.",
    )
    parser.add_argument(
        "--stable-error-pool",
        type=Path,
        default=DEFAULT_STABLE_POOL,
        help="Optional stable error pool CSV to join by path.",
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
        "--near-duplicate-threshold",
        type=int,
        default=5,
        help="Maximum dHash Hamming distance for near-duplicate groups.",
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


def read_stable_pool(path):
    if not path.is_file():
        return {}
    rows = read_csv_rows(
        path,
        ["path", "error_count", "stability_tier", "diagnostic_category"],
    )
    return {
        row["path"]: {
            "error_count": row["error_count"],
            "stability_tier": row["stability_tier"],
            "diagnostic_category": row["diagnostic_category"],
        }
        for row in rows
    }


def dhash(gray, hash_size=8):
    import cv2

    resized = cv2.resize(
        gray,
        (hash_size + 1, hash_size),
        interpolation=cv2.INTER_AREA,
    )
    diff = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}", value


def image_features(image_path):
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        return {
            "read_ok": 0,
            "read_error": "failed to read image",
        }

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    brightness = gray
    edges = cv2.Canny(gray, 100, 200)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    hash_hex, hash_int = dhash(gray)
    pixels = float(height * width)

    return {
        "read_ok": 1,
        "read_error": "",
        "width": width,
        "height": height,
        "brightness_mean": float(brightness.mean()),
        "brightness_std": float(brightness.std()),
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
        "contrast": float(gray.std()),
        "dark_pixel_ratio": float((gray < 64).sum() / pixels),
        "bright_pixel_ratio": float((gray > 192).sum() / pixels),
        "low_saturation_ratio": float((saturation < 32).sum() / pixels),
        "edge_density": float((edges > 0).sum() / pixels),
        "laplacian_variance": float(laplacian.var()),
        "dhash": hash_hex,
        "dhash_int": hash_int,
    }


def hamming_distance(left, right):
    return bin(left ^ right).count("1")


def assign_duplicate_groups(rows, near_duplicate_threshold):
    hash_to_indices = defaultdict(list)
    readable_indices = []
    for index, row in enumerate(rows):
        if row["read_ok"] != 1:
            row["exact_dhash_group_size"] = 0
            row["near_duplicate_group_size"] = 0
            row["near_duplicate_group_id"] = ""
            continue
        hash_to_indices[row["dhash"]].append(index)
        readable_indices.append(index)

    for indices in hash_to_indices.values():
        size = len(indices)
        for index in indices:
            rows[index]["exact_dhash_group_size"] = size

    union_find = UnionFind(len(rows))
    for left_offset, left_index in enumerate(readable_indices):
        left_hash = rows[left_index]["dhash_int"]
        for right_index in readable_indices[left_offset + 1 :]:
            right_hash = rows[right_index]["dhash_int"]
            if hamming_distance(left_hash, right_hash) <= near_duplicate_threshold:
                union_find.union(left_index, right_index)

    groups = defaultdict(list)
    for index in readable_indices:
        groups[union_find.find(index)].append(index)

    group_id_by_root = {}
    next_group_id = 1
    for root, indices in sorted(groups.items(), key=lambda item: item[0]):
        if len(indices) > 1:
            group_id_by_root[root] = f"near_{next_group_id:03d}"
            next_group_id += 1
    for root, indices in groups.items():
        group_id = group_id_by_root.get(root, "")
        group_size = len(indices) if group_id else 1
        for index in indices:
            rows[index]["near_duplicate_group_size"] = group_size
            rows[index]["near_duplicate_group_id"] = group_id


def build_feature_rows(prediction_rows, stable_pool, near_duplicate_threshold):
    rows = []
    for prediction in prediction_rows:
        image_path = prediction["path"]
        features = image_features(REPO_ROOT / image_path)
        stable = stable_pool.get(image_path, {})
        row = {
            "path": image_path,
            "true_label": prediction["true_label"],
            "pred_label": prediction["pred_label"],
            "correct": int(prediction["correct"]),
            "confidence": float(prediction["confidence"]),
            "error_count": int(stable.get("error_count") or 0),
            "stability_tier": stable.get("stability_tier", ""),
            "diagnostic_category": stable.get("diagnostic_category", ""),
            **features,
        }
        if row["read_ok"] != 1:
            for field in [
                "width",
                "height",
                *FEATURE_COLUMNS,
                "dhash",
                "dhash_int",
            ]:
                row[field] = ""
        rows.append(row)
    assign_duplicate_groups(rows, near_duplicate_threshold)
    return rows


def numeric(value):
    if value == "":
        return None
    return float(value)


def summarize_group(rows, group_key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
    summaries = []
    for name, group_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        readable = [row for row in group_rows if row["read_ok"] == 1]
        summary = {
            "group": name,
            "rows": len(group_rows),
            "read_ok": len(readable),
        }
        for field in FEATURE_COLUMNS:
            values = [numeric(row[field]) for row in readable]
            values = [value for value in values if value is not None]
            summary[field] = sum(values) / len(values) if values else None
        summaries.append(summary)
    return summaries


def write_csv(path, rows):
    fieldnames = [
        "path",
        "true_label",
        "pred_label",
        "correct",
        "confidence",
        "error_count",
        "stability_tier",
        "diagnostic_category",
        "read_ok",
        "read_error",
        "width",
        "height",
        *FEATURE_COLUMNS,
        "dhash",
        "exact_dhash_group_size",
        "near_duplicate_group_size",
        "near_duplicate_group_id",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload.pop("dhash_int", None)
            for key in ["confidence", *FEATURE_COLUMNS]:
                if payload[key] != "":
                    payload[key] = f"{float(payload[key]):.6f}"
            writer.writerow(payload)


def feature_table(summaries, title):
    lines = [
        f"## {title}",
        "",
        "| Group | Rows | Read OK | Brightness | Saturation | Contrast | "
        "Dark Ratio | Bright Ratio | Low Sat Ratio | Edge Density | "
        "Laplacian Var |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            "| "
            f"`{summary['group']}` | "
            f"{summary['rows']} | "
            f"{summary['read_ok']} | "
            f"{format_optional(summary['brightness_mean'])} | "
            f"{format_optional(summary['saturation_mean'])} | "
            f"{format_optional(summary['contrast'])} | "
            f"{format_optional(summary['dark_pixel_ratio'])} | "
            f"{format_optional(summary['bright_pixel_ratio'])} | "
            f"{format_optional(summary['low_saturation_ratio'])} | "
            f"{format_optional(summary['edge_density'])} | "
            f"{format_optional(summary['laplacian_variance'])} |"
        )
    return lines


def format_optional(value):
    if value is None:
        return ""
    return f"{value:.4f}"


def duplicate_groups(rows, key):
    groups = defaultdict(list)
    for row in rows:
        group_id = row[key]
        if group_id:
            groups[group_id].append(row)
    return sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))


def exact_hash_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["read_ok"] == 1 and row["exact_dhash_group_size"] > 1:
            groups[row["dhash"]].append(row)
    return sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))


def group_table(groups, title, top_limit):
    lines = [
        f"## {title}",
        "",
        "| Group | Size | Labels | Correct | Example Paths |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    if not groups:
        lines.append("|  | 0 |  |  |  |")
        return lines
    for group_id, group_rows in groups[:top_limit]:
        labels = Counter(row["true_label"] for row in group_rows)
        label_text = "; ".join(
            f"{label}:{count}" for label, count in labels.most_common()
        )
        correct_count = sum(int(row["correct"]) for row in group_rows)
        paths = "<br>".join(f"`{row['path']}`" for row in group_rows[:3])
        lines.append(
            f"| `{group_id}` | {len(group_rows)} | `{label_text}` | "
            f"{correct_count} | {paths} |"
        )
    return lines


def row_table(rows, title, value_field, top_limit):
    lines = [
        f"## {title}",
        "",
        "| Value | True | Pred | Correct | Diagnostic | Path |",
        "| ---: | --- | --- | ---: | --- | --- |",
    ]
    for row in rows[:top_limit]:
        diagnostic = row["diagnostic_category"] or ""
        lines.append(
            f"| {float(row[value_field]):.4f} | "
            f"`{row['true_label']}` | "
            f"`{row['pred_label']}` | "
            f"{row['correct']} | "
            f"`{diagnostic}` | "
            f"`{row['path']}` |"
        )
    return lines


def readable_rows(rows):
    return [row for row in rows if row["read_ok"] == 1]


def write_markdown(path, rows, predictions_path, stable_pool_path, csv_path, args):
    read_failures = [row for row in rows if row["read_ok"] != 1]
    exact_groups = exact_hash_groups(rows)
    near_groups = duplicate_groups(rows, "near_duplicate_group_id")
    exact_samples = sum(len(group_rows) for _, group_rows in exact_groups)
    near_samples = sum(len(group_rows) for _, group_rows in near_groups)
    category_counts = Counter(
        row["diagnostic_category"] or "not_in_stable_pool" for row in rows
    )
    valid_rows = readable_rows(rows)
    lines = [
        "# EfficientNet-B0 Validation Image Features",
        "",
        "Image feature diagnostics for the fixed split used by the seed42 "
        "EfficientNet-B0 Experiment Run.",
        "",
        f"- Predictions: `{relative(predictions_path)}`",
        f"- Stable error pool: `{relative(stable_pool_path)}`",
        f"- CSV: `{relative(csv_path)}`",
        f"- Rows: `{len(rows)}`",
        f"- Read failures: `{len(read_failures)}`",
        f"- Near-duplicate threshold: `{args.near_duplicate_threshold}`",
        "",
        "## Diagnostic Category Counts",
        "",
        "| Category | Rows |",
        "| --- | ---: |",
    ]
    for category, count in category_counts.most_common():
        lines.append(f"| `{category}` | {count} |")

    lines.extend([""])
    lines.extend(
        feature_table(
            summarize_group(rows, lambda row: row["true_label"]),
            "Feature Means By True Label",
        )
    )
    lines.extend([""])
    lines.extend(
        feature_table(
            summarize_group(
                rows,
                lambda row: "correct" if int(row["correct"]) == 1 else "error",
            ),
            "Feature Means By Seed42 Correctness",
        )
    )
    lines.extend([""])
    lines.extend(
        feature_table(
            summarize_group(
                rows,
                lambda row: row["diagnostic_category"] or "not_in_stable_pool",
            ),
            "Feature Means By Stable Pool Category",
        )
    )
    lines.extend(
        [
            "",
            "## Duplicate Hash Summary",
            "",
            f"- Exact dHash collision groups: `{len(exact_groups)}`",
            f"- Samples in exact dHash collision groups: `{exact_samples}`",
            f"- Near-duplicate groups: `{len(near_groups)}`",
            f"- Samples in near-duplicate groups: `{near_samples}`",
            "",
        ]
    )
    lines.extend(group_table(exact_groups, "Top Exact dHash Groups", args.top_limit))
    lines.extend([""])
    lines.extend(
        group_table(near_groups, "Top Near-Duplicate Groups", args.top_limit)
    )

    darkest = sorted(valid_rows, key=lambda row: row["brightness_mean"])
    brightest = sorted(
        valid_rows,
        key=lambda row: row["brightness_mean"],
        reverse=True,
    )
    lowest_saturation = sorted(valid_rows, key=lambda row: row["saturation_mean"])
    most_blurred = sorted(valid_rows, key=lambda row: row["laplacian_variance"])
    highest_edge = sorted(
        valid_rows,
        key=lambda row: row["edge_density"],
        reverse=True,
    )
    for title, selected, field in [
        ("Darkest Rows", darkest, "brightness_mean"),
        ("Brightest Rows", brightest, "brightness_mean"),
        ("Lowest Saturation Rows", lowest_saturation, "saturation_mean"),
        ("Lowest Laplacian Variance Rows", most_blurred, "laplacian_variance"),
        ("Highest Edge Density Rows", highest_edge, "edge_density"),
    ]:
        lines.extend([""])
        lines.extend(row_table(selected, title, field, args.top_limit))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.near_duplicate_threshold < 0:
        print("--near-duplicate-threshold must be non-negative", file=sys.stderr)
        return 2
    if args.top_limit < 1:
        print("--top-limit must be at least 1", file=sys.stderr)
        return 2

    predictions_path = resolve_path(args.predictions)
    stable_pool_path = resolve_path(args.stable_error_pool)
    output_dir = resolve_path(args.output_dir)
    try:
        prediction_rows = read_csv_rows(
            predictions_path,
            ["path", "true_label", "pred_label", "correct", "confidence"],
        )
        stable_pool = read_stable_pool(stable_pool_path)
        rows = build_feature_rows(
            prediction_rows,
            stable_pool,
            args.near_duplicate_threshold,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"image feature audit failed: {exc}", file=sys.stderr)
        return 1
    except ModuleNotFoundError as exc:
        if exc.name == "cv2":
            print(
                "image feature audit failed: OpenCV is required; "
                "install the platform or training runtime dependencies",
                file=sys.stderr,
            )
            return 1
        raise

    csv_path = output_dir / f"{args.stem}.csv"
    md_path = output_dir / f"{args.stem}.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, predictions_path, stable_pool_path, csv_path, args)
    print("image feature audit OK")
    print(f"rows: {len(rows)}")
    print(f"csv: {relative(csv_path)}")
    print(f"markdown: {relative(md_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
