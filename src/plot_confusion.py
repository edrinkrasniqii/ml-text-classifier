"""Generate and save a confusion matrix image for the baseline model."""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

TEST_PATH = os.path.join("data", "processed", "test_clean.csv")
MODEL_PATH = os.path.join("artifacts", "baseline_model.joblib")
OUT_PATH = os.path.join("artifacts", "confusion_matrix.png")


def main():
    df = pd.read_csv(TEST_PATH)
    texts = df["text"].astype(str).tolist()
    y_true = df["label"].tolist()

    pipe = joblib.load(MODEL_PATH)
    y_pred = pipe.predict(texts)

    labels = sorted(df["label"].unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True, values_format="d")
    ax.set_title("Confusion Matrix (TF-IDF + Logistic Regression)")
    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)

    print(f"Saved: {OUT_PATH}")
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
