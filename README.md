# Project Setup

Follow the steps below to set up the project and install the required dependencies.

## Prerequisites

- Python 3.10 or later
- pip

## 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-folder>
```

## 2. Create a Virtual Environment

Create a virtual environment named `.venv`:

```bash
python -m venv .venv
```

## 3. Activate the Virtual Environment

### Windows (Command Prompt)

```bash
.venv\Scripts\activate
```

### Windows (PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
source .venv/bin/activate
```

## 4. Install Required Packages

Install all required dependencies using:

```bash
pip install -r requirements.txt
```

## 5. Launch Jupyter Notebook

If Jupyter Notebook is installed:

```bash
jupyter notebook
```

Or, if using JupyterLab:

```bash
jupyter lab
```

Open the notebook file and run the cells.

## Deactivating the Virtual Environment

When you're finished working:

```bash
deactivate
```