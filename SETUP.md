# Setup Guide

Instructions to set up the environment and run the notebooks for the Thyroid Disease Prediction project.

## Prerequisites

- Python 3.10 or later
- pip
- Jupyter Notebook or JupyterLab

## 1. Clone the Repository

```bash
git clone https://github.com/adhikari-arpan/Thyroid_Disease_Prediction.git
cd Thyroid_Disease_Prediction
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

## 3. Activate the Virtual Environment

**Windows (Command Prompt)**
```bash
.venv\Scripts\activate
```

**Windows (PowerShell)**
```bash
.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

Core libraries used in this project: `scikit-learn`, `tensorflow`, `keras-tuner`, `pandas`, `numpy`, `matplotlib`, `seaborn`.

## 5. Get the Data

Raw and processed dataset files live in `data/`. If it's empty, download the ANN-thyroid dataset files (`ann-train.data`, `ann-test.data`) from the [UCI Repository](https://archive.ics.uci.edu/dataset/102/thyroid+disease) and place them there, then run `00_prepare_data.ipynb` to regenerate `train.csv` / `test.csv`.

## 6. Launch Jupyter

```bash
jupyter notebook
```
or
```bash
jupyter lab
```

Run notebooks in order:
1. `00_prepare_data.ipynb` — combines raw files, builds the stratified train/test split
2. `01_random_forest_model.ipynb` — trains and evaluates the Random Forest baseline
3. `02_random_forest_tuning.ipynb` — hyperparameter tuning for Random Forest
4. `02_deep_learning.ipynb` — trains baseline ANN, runs Keras Tuner search, exports tuned model

Trained models are saved to `models/`; metrics and plots are saved to `results/`.

## Deactivating

When finished:

```bash
deactivate
```

## Troubleshooting

- **TensorFlow install issues on Apple Silicon:** use `tensorflow-macos` instead of `tensorflow` in `requirements.txt`.
- **Keras Tuner not found:** install separately with `pip install keras-tuner`.
- **Kernel not showing in Jupyter:** register the venv with `python -m ipykernel install --user --name=.venv`.
