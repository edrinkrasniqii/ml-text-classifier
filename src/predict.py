"""Simple interactive predictor.

Run: python src\predict.py
"""

import os
import joblib

MODEL_PATH = os.path.join("artifacts", "baseline_model.joblib")


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run: python src\\train_baseline.py")
        raise SystemExit(1)
    return joblib.load(MODEL_PATH)


def main():
    pipe = load_model()
    print("Email classifier (Promotions / Social / Updates / Spam / Forums)")
    print("Type a sentence and press Enter. Type 'q' to quit.")

    while True:
        try:
            text = input("\nEnter email text> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if text.lower() in {"q", "quit", "exit"}:
            print("Bye.")
            break
        if not text:
            print("Please type something (or 'q' to quit).")
            continue

        label = pipe.predict([text])[0]
        print(f"→ Predicted: {label}")


if __name__ == "__main__":
    main()
