"""Train and evaluate TF-IDF + Multinomial Naive Bayes for comparison."""

import os
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

TRAIN = os.path.join("data", "processed", "train_clean.csv")
TEST = os.path.join("data", "processed", "test_clean.csv")
ART = "artifacts"
os.makedirs(ART, exist_ok=True)


def main():
    df_tr = pd.read_csv(TRAIN)
    df_te = pd.read_csv(TEST)
    X_train, y_train = df_tr["text"].astype(str).tolist(), df_tr["label"].tolist()
    X_test, y_test = df_te["text"].astype(str).tolist(), df_te["label"].tolist()

    # TF-IDF followed by Multinomial Naive Bayes
    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)),
            ("clf", MultinomialNB()),
        ]
    )

    print("Training model (TF-IDF + Multinomial Naive Bayes)...")
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
    report = classification_report(y_test, preds, zero_division=0)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Macro avg - Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print("\nClassification report:\n")
    print(report)

    joblib.dump(pipe, os.path.join(ART, "nb_model.joblib"))
    with open(os.path.join(ART, "metrics_nb.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "TFIDF + MultinomialNB",
                "accuracy": acc,
                "precision_macro": prec,
                "recall_macro": rec,
                "f1_macro": f1,
                "labels": sorted(set(y_train) | set(y_test)),
            },
            f,
            indent=2,
        )
    with open(os.path.join(ART, "classification_report_nb.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    print("\nSaved:")
    print(" - artifacts/nb_model.joblib")
    print(" - artifacts/metrics_nb.json")
    print(" - artifacts/classification_report_nb.txt")


if __name__ == "__main__":
    main()
