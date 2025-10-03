"""Data cleaning: map categories, drop 'verify_code', and clean texts."""

import csv
import json
import os
import re
import collections

RAW_DIR = os.path.join("data", "raw")
PROC_DIR = os.path.join("data", "processed")
ART_DIR = "artifacts"
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)

RAW_TRAIN = os.path.join(RAW_DIR, "train.csv")
RAW_TEST = os.path.join(RAW_DIR, "test.csv")
OUT_TRAIN = os.path.join(PROC_DIR, "train_clean.csv")
OUT_TEST = os.path.join(PROC_DIR, "test_clean.csv")

# Source categories -> final labels. None means drop the row.
SOURCE_TO_FINAL = {
    "promotions": "Promotions",
    "social_media": "Social",
    "updates": "Updates",
    "spam": "Spam",
    "forum": "Forums",
    "verify_code": None,
}


def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"\S+@\S+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def process(in_path: str, out_path: str) -> collections.Counter:
    counts = collections.Counter()
    kept = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=["id", "text", "label"])
        writer.writeheader()

        for row in reader:
            src_cat = (row.get("category") or "").strip()
            final = SOURCE_TO_FINAL.get(src_cat, None)
            if final is None:
                continue

            text = row.get("text") or ""
            text = clean_text(text)
            writer.writerow({"id": row.get("id"), "text": text, "label": final})
            counts[final] += 1
            kept += 1

    print(f"[{os.path.basename(in_path)}] kept rows: {kept}")
    for k in sorted(counts):
        print(f"  {k:10s}: {counts[k]}")
    return counts


def main():
    print(">> Cleaning TRAIN ...")
    process(RAW_TRAIN, OUT_TRAIN)
    print("\n>> Cleaning TEST ...")
    process(RAW_TEST, OUT_TEST)

    record = {
        "kept": ["Promotions", "Social", "Updates", "Spam", "Forums"],
        "dropped": ["Verify Code"],
        "mapping": {"social_media": "Social", "forum": "Forums"},
    }
    with open(os.path.join(ART_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print("\nSaved:")
    print(f" - {OUT_TRAIN}")
    print(f" - {OUT_TEST}")
    print(f" - {os.path.join(ART_DIR, 'label_map.json')}")


if __name__ == "__main__":
    main()
