# Symptom Triage Journal — NLP-Based Symptom Checker with Severity Analysis

MSc AI coursework prototype. Classifies free-text symptom descriptions into 24 disease
categories, assigns a clinically-informed severity level (Mild / Moderate / Urgent), and
tracks symptoms longitudinally to warn when a session is escalating.

Two NLP techniques are benchmarked: **calibrated TF-IDF + LinearSVC** and a
**word + character n-gram Logistic Regression** model. An optional third technique,
fine-tuned BERT, runs in `notebook/bert_finetune_colab.ipynb` on Colab GPU.

---

## What's novel

Most public symptom-checker prototypes return a single disease prediction and stop there.
This project adds three layers on top:

1. **Severity-aware triage** — deterministic disease→severity mapping plus a red-flag
   phrase detector (chest pain, shortness of breath, blood, …) that overrides severity
   to Urgent regardless of the model output.
2. **Longitudinal escalation detection** — the app remembers past entries within a
   session and flags worsening trends (Mild → Moderate → Urgent) over time.
3. **Transparent explainability** — top driver tokens, top-3 differential probabilities,
   and a clinician-shareable Markdown export of the whole session.

---

## Project structure

```
project/
├── dataset/                              # Shared by notebook and app
│   └── Symptom2Disease.csv               # Kaggle dataset (1,200 rows · 24 classes)
│
├── notebook/                             # All Jupyter notebooks
│   ├── coursework_main.ipynb             # Final consolidated notebook (run this)
│   └── bert_finetune_colab.ipynb         # Optional Technique 3: BERT (Colab GPU)
│
├── streamlit_web-app/                    # Multi-page Streamlit application
│   ├── streamlit_app.py                  # Home — Live Triage page
│   ├── triage_engine.py                  # Public facade re-exporting the engine
│   ├── constants.py                      # Severity map, colours, red-flag phrases
│   ├── schemas.py                        # Dataclasses (Prediction, JournalEntry, …)
│   ├── inference.py                      # Train / cache / load + predict()
│   ├── escalation.py                     # Longitudinal escalation + Markdown export
│   ├── ui.py                             # Shared Streamlit helpers
│   └── pages/
│       ├── 1_📈_Journal_Timeline.py      # Longitudinal session view
│       └── 2_📋_Clinician_Report.py      # Markdown export for teleconsults
│
├── src/                                  # Reusable training + comparison modules
│   ├── data.py                           # Canonical loader / preprocessing
│   ├── train_models.py                   # Trains both AI techniques + saves artifacts
│   └── compare_models.py                 # Renders comparison plots from saved metrics
│
├── examples/
│   ├── sample_inputs.json                # Curated input/output pairs (replayable test set)
│   └── run_examples.py                   # Replays them through the trained model
│
├── artifacts/                            # Saved models + metrics (auto-generated)
│   ├── calibrated_svm_pipeline.joblib    # Technique 1 — calibrated TF-IDF + LinearSVC
│   ├── logreg_pipeline.joblib            # Technique 2 — Word+Char TF-IDF + LogReg
│   ├── bert_meta.json                    # Optional Technique 3 (drop in from Colab)
│   └── model_comparison.json             # Combined report consumed by app + paper
│
├── figures/                              # Auto-generated plots
│   ├── eda_distribution.png
│   ├── svm_confusion_matrix.png
│   ├── logreg_confusion_matrix.png
│   ├── severity_confusion_matrix.png
│   ├── model_comparison.png
│   └── per_class_f1.png
│
├── requirements.txt
└── README.md
```

> `dataset/` is shared — both `notebook/` and `streamlit_web-app/` resolve paths relative
> to the project root so no copies are needed.

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Run the notebook

```bash
jupyter notebook notebook/coursework_main.ipynb
```

Select **Kernel → Restart & Run All** to execute end-to-end. The notebook resolves the
project root automatically whether Jupyter is launched from `project/` or `project/notebook/`.

| # | Section | Output |
|---|---------|--------|
| 1 | Severity Triage Mapping | — |
| 2 | Dataset Loading & EDA | `figures/eda_distribution.png` |
| 3–5 | Preprocessing + Split | — |
| 6 | Technique 1 — Calibrated SVM (94.7% acc) | `figures/svm_confusion_matrix.png` · `artifacts/calibrated_svm_pipeline.joblib` |
| 7 | Technique 2 — Word+Char LogReg (97.2% acc) | `figures/logreg_confusion_matrix.png` · `artifacts/logreg_pipeline.joblib` |
| 8 | Model comparison | `figures/model_comparison.png` · `figures/per_class_f1.png` |
| 9 | Severity triage layer | `figures/severity_confusion_matrix.png` |
| 10 | Longitudinal escalation detector | — |
| 11 | Explainability (top drivers + top-3 differential) | — |
| 12–13 | Demo inference + escalation walk-through | — |
| 14 | BERT (gated runtime check, GPU) | See `notebook/bert_finetune_colab.ipynb` |

---

## Train both AI techniques (standalone)

```bash
python -m src.train_models      # trains Technique 1 (SVM) and Technique 2 (LogReg)
python -m src.compare_models    # renders figures/model_comparison.png + per_class_f1.png
```

Current results on the held-out 30% split (n=360):

| Technique | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|-----------|----------|-------------------|----------------|------------|
| **Technique 1** — TF-IDF (word 1-2g) + Calibrated LinearSVC | 94.72% | 0.949 | 0.947 | 0.946 |
| **Technique 2** — TF-IDF (word + char 3-5g) + Logistic Regression | **97.22%** | **0.974** | **0.972** | **0.972** |
| _Optional_ Technique 3 — Fine-tuned BERT (Colab) | _populated when_ `artifacts/bert_meta.json` _is present_ |


## Run the Streamlit app

```bash
streamlit run streamlit_web-app/streamlit_app.py
```

Opens at **http://localhost:8501**. On first launch the app trains the calibrated
TF-IDF + LinearSVC pipeline (a few seconds) and caches it under `artifacts/`.
Subsequent launches load the cache instantly.

### App pages

| Page | Purpose |
|------|---------|
| **🩺 Live Triage** (home) | Type symptoms → top-3 differential, severity badge, driver tokens, red-flag detection. |
| **📈 Journal Timeline** | Per-session timeline of all logged entries; severity-over-time line chart; escalation verdict. |
| **📋 Clinician Report** | One-click Markdown / plain-text export of the session, ready to paste into a teleconsult. |

---

## Optional Technique 3 — BERT fine-tuning (Colab)

```
notebook/bert_finetune_colab.ipynb
```

1. Open in Google Colab.
2. **Runtime → Change runtime type → GPU**.
3. Upload `dataset/Symptom2Disease.csv` to the Colab session.
4. **Runtime → Run all** (≈ 2-3 min on a T4).
5. Download `bert_meta.json` and place it under `artifacts/bert_meta.json`.
6. Re-run `python -m src.train_models` to refresh `model_comparison.json`.

---

## Dataset

**Symptom2Disease** — Niyarr Barman, Kaggle.
1,200 samples across 24 disease categories (50 per class). Located at `dataset/Symptom2Disease.csv`.

---

## Disclaimer

This is an academic prototype. It does **not** provide medical advice and must not be
used to diagnose or treat any condition. If you are unwell, contact a qualified clinician.
