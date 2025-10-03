"""Extract top words from the cleaned training set."""

import csv
import os
import collections

IN_PATH = os.path.join("data", "processed", "train_clean.csv")
ART_DIR = "artifacts"
OUT_CSV = os.path.join(ART_DIR, "top_words_global.csv")
os.makedirs(ART_DIR, exist_ok=True)


def main():
    counts = collections.Counter()
    with open(IN_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "")
            for tok in text.split():
                if len(tok) < 3:
                    continue
                counts[tok] += 1

    top = counts.most_common(30)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["word", "count"])
        w.writerows(top)

    print(f"Saved: {OUT_CSV}")
    print("Top 10 preview:")
    for w, c in top[:10]:
        print(f"{w:15s} {c}")


if __name__ == "__main__":
    main()
