"""Compute class distribution and a small dataset overview."""

import csv
import os
import statistics
import collections

IN_PATH = os.path.join("data", "processed", "train_clean.csv")
ART_DIR = "artifacts"
os.makedirs(ART_DIR, exist_ok=True)

OUT_COUNTS = os.path.join(ART_DIR, "class_counts.csv")
OUT_OVERVIEW = os.path.join(ART_DIR, "overview.txt")


def main():
    counts = collections.Counter()
    lengths = []

    with open(IN_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        total = 0
        for row in reader:
            total += 1
            label = (row.get("label") or "").strip()
            text = (row.get("text") or "")
            counts[label] += 1
            lengths.append(len(text.split()))

    labels_sorted = sorted(counts.keys())
    with open(OUT_COUNTS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "count"])
        for lbl in labels_sorted:
            w.writerow([lbl, counts[lbl]])

    num_classes = len(labels_sorted)
    min_len = min(lengths) if lengths else 0
    mean_len = round(statistics.mean(lengths), 1) if lengths else 0
    median_len = statistics.median(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0

    lines = [
        "=== DATASET OVERVIEW (train_clean.csv) ===",
        f"Total rows: {total}",
        f"Number of classes: {num_classes}",
        f"Text length (words): min={min_len}, mean={mean_len}, median={median_len}, max={max_len}",
        "",
        "Per-class counts and percentages:",
    ]
    for lbl in labels_sorted:
        pct = (counts[lbl] / total * 100) if total else 0
        lines.append(f"  {lbl:10s} : {counts[lbl]}  ({pct:.1f}%)")

    with open(OUT_OVERVIEW, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Saved:")
    print(f" - {OUT_COUNTS}")
    print(f" - {OUT_OVERVIEW}")


if __name__ == "__main__":
    main()
