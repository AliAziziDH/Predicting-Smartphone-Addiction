import os
import json
import re

def create_notebook_cell(cell_type, source):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")]
    }

def clean_code(filepath):
    with open(filepath, 'r') as f:
        code = f.read()

    # Strip out local relative imports
    code = re.sub(r'from src\.model\.formulation import .*\n', '', code)
    code = re.sub(r'from src\.model\.neural_tabular import .*\n', '', code)
    code = re.sub(r'from src\.model\.solver import .*\n', '', code)
    code = re.sub(r'from src\.train import .*\n', '', code)
    code = re.sub(r'from src\.predict import .*\n', '', code)
    code = re.sub(r'import src\.model\.formulation\n', '', code)
    code = re.sub(r'import src\.model\.neural_tabular\n', '', code)
    code = re.sub(r'import src\.model\.solver\n', '', code)
    code = re.sub(r'import src\.train\n', '', code)
    code = re.sub(r'import src\.predict\n', '', code)

    # Strip out root dir and sys.path manipulation for clean notebook execution
    code = re.sub(r'# Dynamic path resolution to handle running from subfolders, root, or notebook\n', '', code)
    code = re.sub(r'# Dynamic path resolution to handle running from subfolders or root\n', '', code)
    code = re.sub(r'try:\n\s+ROOT_DIR = os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\n\s+if os\.path\.basename\(ROOT_DIR\) in \["src", "tests"\]:\n\s+ROOT_DIR = os\.path\.dirname\(ROOT_DIR\)\n\s+if ROOT_DIR not in sys\.path:\n\s+sys\.path\.insert\(0, ROOT_DIR\)\nexcept NameError:\n\s+ROOT_DIR = os\.getcwd\(\)\n', '', code)
    code = re.sub(r'ROOT_DIR = os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\n', '', code)
    code = re.sub(r'if os\.path\.basename\(ROOT_DIR\) in \["src", "tests"\]:\n\s+ROOT_DIR = os\.path\.dirname\(ROOT_DIR\)\n', '', code)
    code = re.sub(r'if ROOT_DIR not in sys\.path:\n\s+sys\.path\.insert\(0, ROOT_DIR\)\n', '', code)
    return code.strip()

def compile_notebook(output_path):
    cells = []

    # 1. Header Markdown & Setup
    cells.append(create_notebook_cell("markdown", """# Smartphone Addiction Prediction - Elite 4-Way Ensemble
## High-Capacity GBDTs + PyTorch Deep Tabular MLP + Nested Logistic Stacking on GPU

This notebook implements an automated pipeline designed for the Kaggle Playground Series s6e8 competition.
It combines high-capacity gradient boosting (LightGBM, XGBoost, CatBoost) with a Deep Tabular Neural Network and a Nested Logistic Stacker on rank percentiles to maximize Out-of-Fold (OOF) ROC AUC."""))

    setup_code = """import os
import gc
import warnings
import numpy as np
import pandas as pd
import optuna
from typing import Dict, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from pydantic import BaseModel, Field, ValidationError
from scipy.optimize import minimize
from scipy.stats import rankdata, ks_2samp

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

# Dynamic Kaggle vs Local Path Resolution
def resolve_data_path(filename):
    paths_to_check = [
        f"/kaggle/input/playground-series-s6e8/{filename}",
        f"/kaggle/input/competitions/playground-series-s6e8/{filename}",
        f"../input/playground-series-s6e8/{filename}",
        f"data/{filename}",
        f"./{filename}",
        f"../{filename}"
    ]
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"[INFO] Successfully resolved {filename} to: {path}")
            return path

    search_roots = ["/kaggle/input", "../input", "data", "."]
    for root_dir in search_roots:
        if os.path.exists(root_dir):
            for root, dirs, files in os.walk(root_dir):
                if filename in files:
                    found = os.path.join(root, filename)
                    print(f"[INFO] Found {filename} via walk: {found}")
                    return found

    raise FileNotFoundError(f"Could not find {filename} anywhere in {search_roots}")

# Ensure reproducibility
def seed_everything(seed=42):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(42)"""
    cells.append(create_notebook_cell("code", setup_code))

    # 2. Predictive Layer & Features
    cells.append(create_notebook_cell("markdown", """### Predictive Layer & Features (Native NaN Propagation + Grid Frequency)"""))
    cells.append(create_notebook_cell("code", clean_code('src/model/formulation.py')))

    # 3. Deep Tabular PyTorch Neural Network
    cells.append(create_notebook_cell("markdown", """### Deep Tabular PyTorch Neural Network (Entity Embeddings + Residual Blocks)"""))
    cells.append(create_notebook_cell("code", clean_code('src/model/neural_tabular.py')))

    # 4. Solver & Modeling Core
    cells.append(create_notebook_cell("markdown", """### Solver Core & Nested Logistic Stacker"""))
    cells.append(create_notebook_cell("code", clean_code('src/model/solver.py')))

    # 5. 10-Fold Stratified CV training logic
    cells.append(create_notebook_cell("markdown", """### Training Loop (10-Fold Stratified CV)"""))
    cells.append(create_notebook_cell("code", clean_code('src/train.py') + "\n\nif __name__ == '__main__':\n    main()"))

    # 6. Inference, rank percentile normalization, and submission formatting
    cells.append(create_notebook_cell("markdown", """### Inference, KS Drift Screening and Submission Formatting"""))
    cells.append(create_notebook_cell("code", clean_code('src/predict.py') + "\n\nif __name__ == '__main__':\n    main()"))

    # 7. Direct Cloud-to-Competition Auto-Submission
    cells.append(create_notebook_cell("markdown", """### Direct Cloud-to-Competition Submission (Zero Local Roundtrips)"""))
    cells.append(create_notebook_cell("code", """# Auto-submit directly from cloud environment to Kaggle
import subprocess
sub_file = "outputs/submission.csv"
if os.path.exists(sub_file):
    print("🚀 Submitting directly from Cloud to Kaggle Competition...", flush=True)
    res = subprocess.run([
        "kaggle", "competitions", "submit",
        "-c", "playground-series-s6e8",
        "-f", sub_file,
        "-m", "V10: 10-Fold Ensemble + Optuna Tuned + ValueLevel Target Enc + Productive Shield"
    ], capture_output=True, text=True)
    print(res.stdout, flush=True)
    if res.stderr:
        print("Kaggle CLI response:", res.stderr, flush=True)
else:
    print(f"Submission file not found at {sub_file}")"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=1)

if __name__ == '__main__':
    compile_notebook('predicting-smartphone-addiction-elite.ipynb')
    print("Notebook compiled successfully at predicting-smartphone-addiction-elite.ipynb")
