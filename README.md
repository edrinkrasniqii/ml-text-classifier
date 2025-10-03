
# Gmail-like Email Text Classifier

## Overview

This repository contains a small, reproducible text-classification pipeline that assigns email messages to Gmail-like tabs (Promotions, Social, Updates, Spam, Forums). The emphasis is on clear data handling, a simple and explainable baseline model (TF-IDF + Logistic Regression), fair evaluation using the official train/test split, and a small interactive prediction utility. A second model (Multinomial Naive Bayes) and a basic FastAPI endpoint are included as bonuses.

## Contents / Quick links

- `data/` – source and processed CSVs
- `src/` – preprocessing, EDA, training, prediction, and a small API
- `artifacts/` – generated models, reports, metrics, and plots
- `README.md` – this file

## Dataset

- Source: High-Accuracy Email Classification Dataset (Hugging Face)
  - https://huggingface.co/datasets/jason23322/high-accuracy-email-classifier
- License: Apache-2.0

Original labels: `promotions`, `social_media`, `updates`, `spam`, `forum`, `verify_code`.

For this project we keep 5 classes and drop `verify_code`:

- Kept: Promotions, Social, Updates, Spam, Forums
- Dropped: Verify Code
- Mapping: `social_media` → `Social`, `forum` → `Forums`

Label mapping file: `artifacts/label_map.json` (example below):

```json
{
  "kept": ["Promotions", "Social", "Updates", "Spam", "Forums"],
  "dropped": ["Verify Code"],
  "mapping": { "social_media": "Social", "forum": "Forums" }
}
```

Why this dataset: ~12.5k labeled emails, accessible license, and clear categories that make it suitable for a teaching/demo project.

## Approach

High-level plan and design decisions:

- Data preparation: load CSVs, lowercase, remove URLs and email addresses, drop punctuation/numbers, collapse spaces. Keep preprocessing minimal so the baseline is easy to understand and reproduce.
- EDA: verify class counts, show a few examples per class, and extract top frequent tokens.
- Modeling: baseline TF-IDF features + Logistic Regression. A second model uses TF-IDF + Multinomial Naive Bayes for comparison.
- Evaluation: use the provided train/test split and compute accuracy, precision, recall, and macro F1. Produce a classification report and a confusion matrix image.
- Prediction: interactive command-line utility (`src/predict.py`) and an optional FastAPI endpoint (`src/api.py`).

### Load & Inspect (plan)

- Verify the dataset contains `text` and `label` fields.
- Confirm the 5-class set after dropping `verify_code`.
- Check per-class counts and inspect a few examples per class.
- Note very short or missing texts if present.

### Preprocessing rules

- Lowercase all text.
- Remove URLs and email addresses.
- Keep only letters and spaces (strip punctuation and digits).
- Collapse multiple spaces and trim leading/trailing spaces.
- Remove English stopwords inside the TF-IDF vectorizer (keep cleaning simple in code).

## Produced EDA outputs

- `artifacts/class_counts.csv` — label, count
- `artifacts/overview.txt` — dataset size, number of classes, text length (min/mean/median/max), per-class percentage
- `artifacts/top_words_global.csv` — top frequent words after simple cleaning
- `artifacts/label_map.json` — kept/dropped/mapping

## Results (examples)

Model 1 — TF-IDF + Logistic Regression (baseline)

- Accuracy: 0.9920
- Macro Precision / Recall / F1: 0.9920 / 0.9920 / 0.9920
- Report: `artifacts/classification_report.txt`
- Confusion matrix: `artifacts/confusion_matrix.png`

Model 2 — TF-IDF + Multinomial Naive Bayes

- Accuracy: 0.9880
- Macro Precision / Recall / F1: 0.9880 / 0.9880 / 0.9880
- Report: `artifacts/classification_report_nb.txt`

Comparison table: `artifacts/results_compare.csv`

Best performer used by `src/predict.py`: TF-IDF + Logistic Regression.

Notes: rare confusions occur (for example, Updates ↔ Spam), which is realistic for short email text.

## How to run (Windows / PowerShell)

Open a PowerShell terminal in the repository root and keep a virtual environment active.

### 1) Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # or use .\.venv\Scripts\activate for cmd.exe
python -m pip install --upgrade pip
pip install scikit-learn pandas numpy joblib matplotlib
```

Optional (for API):

```powershell
pip install fastapi "uvicorn[standard]"
```

### 2) Get data

Download `train.csv` and `test.csv` from the dataset page and place them at:

```
data/raw/train.csv
data/raw/test.csv
```

### 3) Preprocess (drop Verify Code and clean text)

```powershell
python src\clean_data.py
```

Outputs:

- `data/processed/train_clean.csv`
- `data/processed/test_clean.csv`
- `artifacts/label_map.json`

### 4) EDA

```powershell
python src\eda_counts.py
python src\eda_top_words.py
```

Outputs:

- `artifacts/class_counts.csv`
- `artifacts/overview.txt`
- `artifacts/top_words_global.csv`

### 5) Train & evaluate (baseline)

```powershell
python src\train_baseline.py
```

Outputs:

- `artifacts/baseline_model.joblib`
- `artifacts/metrics.json`
- `artifacts/classification_report.txt`

### 6) Confusion matrix (bonus)

```powershell
python src\plot_confusion.py
```

Output: `artifacts/confusion_matrix.png`

### 7) Predict (interactive)

```powershell
python src\predict.py
```

Type a sentence to get one of: Promotions / Social / Updates / Spam / Forums. Type `q` to quit.

### 8) (Bonus) Second model + comparison

```powershell
python src\train_nb.py
python src\compare_models.py
```

Output: `artifacts/results_compare.csv`

### 9) (Bonus) Simple API (FastAPI)

Start the API:

```powershell
uvicorn src.api:app --reload
```

Swagger UI: http://127.0.0.1:8000/docs

Health check: http://127.0.0.1:8000/health

Example request body (POST /predict in the docs):

```json
{ "text": "Your invoice for October is ready." }
```

(If `http://127.0.0.1:8000/` returns `{"detail":"Not Found"}`, use `/docs` or add a simple root route in `src/api.py`.)

## Repository structure

```
ml-text-classifier/
  ├─ src/
  │   ├─ clean_data.py
  │   ├─ eda_counts.py
  │   ├─ eda_top_words.py
  │   ├─ train_baseline.py
  │   ├─ train_nb.py
  │   ├─ compare_models.py
  │   ├─ plot_confusion.py
  │   └─ predict.py
  ├─ data/
  │   ├─ raw/          # train.csv, test.csv (downloaded)
  │   └─ processed/    # *_clean.csv (generated)
  ├─ artifacts/        # metrics, reports, confusion_matrix, model, etc.
  └─ README.md
```

## Scope & assumptions

- This project approximates Gmail tabs for demonstration; it is not affiliated with Google or Gmail.
- English emails only. No handling of attachments, threaded conversations, or multi-part MIME.
- Focus is on clarity and reproducibility over heavy hyperparameter tuning.

## Next steps / possible improvements

- Add richer preprocessing (lemmatization, custom tokenizer).
- Experiment with transformer-based models and fine-tuning.
- Add CI to run quick smoke tests and linting.
- Add unit tests for preprocessing and prediction logic.

---

If you want, I can also: add a short contribution section, include an example curl request for the API, or generate a `requirements.txt` for the project. Tell me which and I'll update the README.


