#!/usr/bin/env python3
"""Create image feature diagnostics for the full Training Set."""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_DIR = REPO_ROOT / "datasets/6a39ed934d7b489daf5f80a4-momodel/train"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs/diagnostics"
DEFAULT_STEM = "train-image-features"
WEATHER_CATEGORIES = ("cloudy", "rainy", "snowy", "sunny")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit image features for the full Training Set."
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help="Training Image Root containing one subdirectory per Weather Category.",
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


def relative(path):
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def discover_images(train_dir):
    if not train_dir.is_dir():
        raise FileNotFoundError(f"missing Training Image Root: {relative(train_dir)}")

    missing_categories = [
        category
        for category in WEATHER_CATEGORIES
        if not (train_dir / category).is_dir()
    ]
    if missing_categories:
        raise ValueError(
            "Training Image Root is missing Weather Category directories: "
            + ", ".join(missing_categories)
        )

    rows = []
    for category in WEATHER_CATEGORIES:
        category_dir = train_dir / category
        for image_path in sorted(category_dir.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            rows.append(
                {
                    "path": relative(image_path),
                    "true_label": category,
                    "image_path": image_path,
                }
            )
    if not rows:
        raise ValueError(f"no images found under {relative(train_dir)}")
    return rows


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
    edges = cv2.Canny(gray, 100, 200)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    hash_hex, hash_int = dhash(gray)
    pixels = float(height * width)

    return {
        "read_ok": 1,
        "read_error": "",
        "width": width,
        "height": height,
        "brightness_mean": float(gray.mean()),
        "brightness_std": float(gray.std()),
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


def build_feature_rows(image_rows, near_duplicate_threshold):
    rows = []
    for image_row in image_rows:
        features = image_features(image_row["image_path"])
        row = {
            "path": image_row["path"],
            "true_label": image_row["true_label"],
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


def readable_rows(rows):
    return [row for row in rows if row["read_ok"] == 1]


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


def format_optional(value):
    if value is None:
        return ""
    return f"{value:.4f}"


def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_quantiles(rows, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["true_label"]].append(row)
    summaries = []
    for label, label_rows in sorted(grouped.items()):
        readable = [row for row in label_rows if row["read_ok"] == 1]
        for field in fields:
            values = [numeric(row[field]) for row in readable]
            values = [value for value in values if value is not None]
            summaries.append(
                {
                    "label": label,
                    "feature": field,
                    "p05": quantile(values, 0.05),
                    "p50": quantile(values, 0.50),
                    "p95": quantile(values, 0.95),
                }
            )
    return summaries


def write_csv(path, rows):
    fieldnames = [
        "path",
        "true_label",
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
            for key in FEATURE_COLUMNS:
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
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
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


def quantile_table(summaries):
    lines = [
        "## Feature Quantiles By True Label",
        "",
        "| Label | Feature | P05 | P50 | P95 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            f"| `{summary['label']}` | "
            f"`{summary['feature']}` | "
            f"{format_optional(summary['p05'])} | "
            f"{format_optional(summary['p50'])} | "
            f"{format_optional(summary['p95'])} |"
        )
    return lines


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


def cross_label_groups(groups):
    return [
        (group_id, group_rows)
        for group_id, group_rows in groups
        if len({row["true_label"] for row in group_rows}) > 1
    ]


def group_table(groups, title, top_limit):
    lines = [
        f"## {title}",
        "",
        "| Group | Size | Labels | Example Paths |",
        "| --- | ---: | --- | --- |",
    ]
    if not groups:
        lines.append("|  | 0 |  |  |")
        return lines
    for group_id, group_rows in groups[:top_limit]:
        labels = Counter(row["true_label"] for row in group_rows)
        label_text = "; ".join(
            f"{label}:{count}" for label, count in labels.most_common()
        )
        paths = "<br>".join(f"`{row['path']}`" for row in group_rows[:3])
        lines.append(f"| `{group_id}` | {len(group_rows)} | `{label_text}` | {paths} |")
    return lines


def row_table(rows, title, value_field, top_limit):
    lines = [
        f"## {title}",
        "",
        "| Value | True | Path |",
        "| ---: | --- | --- |",
    ]
    for row in rows[:top_limit]:
        lines.append(
            f"| {float(row[value_field]):.4f} | "
            f"`{row['true_label']}` | "
            f"`{row['path']}` |"
        )
    return lines


def dimension_table(rows, top_limit):
    counts = Counter(
        (row["width"], row["height"]) for row in rows if row["read_ok"] == 1
    )
    lines = [
        "## Image Dimensions",
        "",
        "| Width | Height | Rows |",
        "| ---: | ---: | ---: |",
    ]
    for (width, height), count in counts.most_common(top_limit):
        lines.append(f"| {width} | {height} | {count} |")
    return lines


def write_markdown(path, rows, train_dir, csv_path, args):
    read_failures = [row for row in rows if row["read_ok"] != 1]
    valid_rows = readable_rows(rows)
    exact_groups = exact_hash_groups(rows)
    near_groups = duplicate_groups(rows, "near_duplicate_group_id")
    exact_cross_label = cross_label_groups(exact_groups)
    near_cross_label = cross_label_groups(near_groups)
    exact_samples = sum(len(group_rows) for _, group_rows in exact_groups)
    near_samples = sum(len(group_rows) for _, group_rows in near_groups)
    label_counts = Counter(row["true_label"] for row in rows)

    lines = [
        "# Training Set Image Features",
        "",
        "Image feature diagnostics for the full official Training Set. This report "
        "summarizes observed data properties and does not propose automatic "
        "relabeling rules. dHash groups are coarse perceptual similarity "
        "candidates, not proof of duplicate source images or label errors.",
        "",
        f"- Training Image Root: `{relative(train_dir)}`",
        f"- CSV: `{relative(csv_path)}`",
        f"- Rows: `{len(rows)}`",
        f"- Read failures: `{len(read_failures)}`",
        f"- Near-duplicate threshold: `{args.near_duplicate_threshold}`",
        "",
        "## Weather Category Counts",
        "",
        "| Weather Category | Rows |",
        "| --- | ---: |",
    ]
    for label in WEATHER_CATEGORIES:
        lines.append(f"| `{label}` | {label_counts[label]} |")

    lines.extend([""])
    lines.extend(dimension_table(rows, args.top_limit))
    lines.extend([""])
    lines.extend(
        feature_table(
            summarize_group(rows, lambda row: row["true_label"]),
            "Feature Means By True Label",
        )
    )
    lines.extend([""])
    lines.extend(
        quantile_table(
            summarize_quantiles(
                rows,
                [
                    "brightness_mean",
                    "saturation_mean",
                    "contrast",
                    "edge_density",
                    "laplacian_variance",
                ],
            )
        )
    )
    lines.extend(
        [
            "",
            "## Duplicate Hash Summary",
            "",
            f"- Exact dHash collision groups: `{len(exact_groups)}`",
            f"- Samples in exact dHash collision groups: `{exact_samples}`",
            f"- Cross-label exact dHash groups: `{len(exact_cross_label)}`",
            f"- Near-duplicate groups: `{len(near_groups)}`",
            f"- Samples in near-duplicate groups: `{near_samples}`",
            f"- Cross-label near-duplicate groups: `{len(near_cross_label)}`",
            "",
        ]
    )
    lines.extend(group_table(exact_groups, "Top Exact dHash Groups", args.top_limit))
    lines.extend([""])
    lines.extend(
        group_table(
            exact_cross_label,
            "Top Cross-Label Exact dHash Groups",
            args.top_limit,
        )
    )
    lines.extend([""])
    lines.extend(group_table(near_groups, "Top Near-Duplicate Groups", args.top_limit))
    lines.extend([""])
    lines.extend(
        group_table(
            near_cross_label,
            "Top Cross-Label Near-Duplicate Groups",
            args.top_limit,
        )
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

    train_dir = resolve_path(args.train_dir)
    output_dir = resolve_path(args.output_dir)
    try:
        image_rows = discover_images(train_dir)
        rows = build_feature_rows(image_rows, args.near_duplicate_threshold)
    except (FileNotFoundError, ValueError) as exc:
        print(f"training image feature audit failed: {exc}", file=sys.stderr)
        return 1
    except ModuleNotFoundError as exc:
        if exc.name == "cv2":
            print(
                "training image feature audit failed: OpenCV is required; "
                "install the platform or training runtime dependencies",
                file=sys.stderr,
            )
            return 1
        raise

    csv_path = output_dir / f"{args.stem}.csv"
    md_path = output_dir / f"{args.stem}.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, train_dir, csv_path, args)
    print("training image feature audit OK")
    print(f"rows: {len(rows)}")
    print(f"csv: {relative(csv_path)}")
    print(f"markdown: {relative(md_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
