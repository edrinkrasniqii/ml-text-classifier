import csv
import statistics
import collections
import os

TRAIN_PATH = os.path.join("data", "raw", "train.csv")


def main():
    counts = collections.Counter()
    lengths = []
    samples = collections.defaultdict(list)

    with open(TRAIN_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        total = 0
        for row in reader:
            total += 1
            cat = (row.get("category") or "").strip()
            text = (row.get("text") or "")
            counts[cat] += 1
            lengths.append(len(text.split()))
            if len(samples[cat]) < 2:
                samples[cat].append(text.replace("\n", " ")[:120])

    print("\n=== BASIC OVERVIEW ===")
    print(f"Total rows: {total}")
    print("Classes:", ", ".join(sorted(counts.keys())))
    for k in sorted(counts):
        print(f"  {k:12s} : {counts[k]}")

    print("\n=== TEXT LENGTH (words) ===")
    print(f"min={min(lengths)}, mean={round(statistics.mean(lengths),1)}, median={statistics.median(lengths)}, max={max(lengths)}")

    print("\n=== SAMPLE SNIPPETS (per class) ===")
    for k in sorted(samples):
        print(f"\n[{k}]")
        for s in samples[k]:
            print(" -", s)


if __name__ == "__main__":
    main()