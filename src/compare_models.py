"""Compare saved metrics for baseline and Naive Bayes models."""

import os
import json
import csv

ART = "artifacts"
BASE = os.path.join(ART, "metrics.json")
NB = os.path.join(ART, "metrics_nb.json")
OUT = os.path.join(ART, "results_compare.csv")


def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    a = load(BASE)
    b = load(NB)

    rows = [
        ["Model", a.get("model"), b.get("model")],
        ["Accuracy", f'{a["accuracy"]:.4f}', f'{b["accuracy"]:.4f}'],
        ["Precision (macro)", f'{a["precision_macro"]:.4f}', f'{b["precision_macro"]:.4f}'],
        ["Recall (macro)", f'{a["recall_macro"]:.4f}', f'{b["recall_macro"]:.4f}'],
        ["F1 (macro)", f'{a["f1_macro"]:.4f}', f'{b["f1_macro"]:.4f}'],
    ]

    print("\n=== Model Comparison ===")
    headers = ["Metric", "LogReg", "Naive Bayes"]
    colw = [max(len(h), max(len(r[i]) for r in ([headers] + rows))) for i, h in enumerate(headers)]
    print(f'{headers[0]:<{colw[0]}}  {headers[1]:<{colw[1]}}  {headers[2]:<{colw[2]}}')
    print("-" * (sum(colw) + 4))
    for r in rows:
        print(f'{r[0]:<{colw[0]}}  {r[1]:<{colw[1]}}  {r[2]:<{colw[2]}}')

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "logreg", "naive_bayes"])
        for r in rows:
            w.writerow([r[0], r[1], r[2]])

    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
