# Thyroid Disease Prediction

A comparative case study on **Random Forest** vs. **Artificial Neural Network (ANN)** for three-class thyroid disease classification, using the UCI ANN-thyroid dataset. Built as a course project for Nepal College of Information Technology (NCIT), Pokhara University.

## Overview

Thyroid function abnormalities are commonly classified as hyperthyroid, hypothyroid, or normal based on demographic data, treatment history, and lab measurements (TSH, T3, TT4, T4U, FTI). This project frames the task as supervised multiclass classification and compares a classical machine-learning model against a deep-learning model under identical data conditions, with a focus on macro-averaged metrics due to significant class imbalance (~92.6% normal class).

## Dataset

- **Source:** [UCI Machine Learning Repository — Thyroid Disease](https://archive.ics.uci.edu/dataset/102/thyroid+disease)
- **Records:** 7,200 total (ann-train.data + ann-test.data combined)
- **Features:** 21 numeric predictors — 6 continuous (age, TSH, T3, TT4, T4U, FTI) + 15 binary clinical flags
- **Classes:** Hyperthyroid (166), Hypothyroid (368), Normal (6,666)
- **Split:** Stratified 80:20 → 5,760 train / 1,440 test (`random_state=42`)

## Methodology

1. Combine raw UCI files, add documented column names, extract class codes
2. Stratified train/test split (shared across both models)
3. **Random Forest** — 100 trees, Gini impurity, `class_weight='balanced'`, `max_features='sqrt'`
4. **ANN (MLP)** — 21 → Dense(64, ReLU) + BatchNorm + Dropout(0.3) → Dense(32, ReLU) + Dropout(0.2) → Softmax(3)
5. Class-weighted loss for both models to counter imbalance
6. Keras Tuner RandomSearch (12 trials) for ANN hyperparameter optimization
7. Evaluation via accuracy, balanced accuracy, macro precision/recall/F1, weighted F1, and confusion matrices

## Results

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| Random Forest (baseline) | **99.58%** | 96.91% | 99.85% | **98.34%** |
| ANN (baseline) | 75.97% | 54.37% | 74.64% | 57.71% |
| ANN (tuned) | 80.49% | 60.86% | 72.73% | 62.84% |

**Key finding:** Random Forest substantially outperformed the ANN on this structured, imbalanced tabular dataset. Hyperparameter tuning improved the ANN's accuracy and macro F1, but minority-class precision (especially hypothyroid) remained a persistent weakness for the neural model.

See the full report (`ML_Thyroid_Detection.pdf`) for detailed confusion matrices, training curves, and discussion.

## Repository Structure

```
├── data/                          # Raw and processed datasets (train.csv, test.csv)
├── models/                        # Exported trained models (Random Forest, ANN)
├── results/                       # Metrics, plots, confusion matrices
├── 00_prepare_data.ipynb          # Combine UCI files, stratified split
├── 01_random_forest_model.ipynb   # Random Forest baseline
├── 02_random_forest_tuning.ipynb  # Random Forest hyperparameter tuning
├── 02_deep_learning.ipynb         # ANN baseline + Keras Tuner search
├── app.py                         # Demo/inference app
├── requirements.txt
├── README.md
└── SETUP.md
```

## Getting Started

See [SETUP.md](./SETUP.md) for environment setup and installation instructions.

## Running the App

`app.py` provides a simple interface for running predictions with the trained models. After setup, launch it with:

```bash
python app.py
```

## Limitations

- Original UCI train/test files were combined and re-split, so results aren't directly comparable to studies preserving the official split
- Single random split; no repeated cross-validation or confidence intervals
- No exact-duplicate audit performed before splitting
- Educational prototype only — **not a clinical diagnostic tool**

## References

Full citation list available in the report. Key sources: UCI Thyroid Disease dataset, Breiman (2001) Random Forests, Temurtas (2009) neural network thyroid diagnosis comparison.
