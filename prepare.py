"""
FROZEN -- Do not modify this file.
Data loading, train/val split, evaluation metric, and plotting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import csv
import os

# ── Constants ──────────────────────────────────────────────
DATA_PATH = 'data/cleaned/exp_train.pkl'
RANDOM_STATE = 42
VAL_FRACTION = 0.2

# ── Data ───────────────────────────────────────────────────
def load_data():
    """
    Load and split data
    """

    data = pd.read_pickle(DATA_PATH)
    X, y = data.drop(columns=['chromosome', 'group']), data['group'].astype(int)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VAL_FRACTION,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, y_train, X_val, y_val


# ── Evaluation (frozen metric) ─────────────────────────────
def evaluate(model, X_val, y_val):
    """
    Compute validation balanced accuracy (higher is better).
    """
    y_pred = model.predict(X_val)
    acc = balanced_accuracy_score(y_val, y_pred)
    return acc


# ── Logging ────────────────────────────────────────────────
def log_result(results_file, experiment_id, val_acc, status, description):
    """
    Append one row to results.tsv.
    """
    file_exists = os.path.exists(results_file)
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        if not file_exists:
            writer.writerow(['experiment', 'val_acc', 'status', 'description'])
        writer.writerow([experiment_id, f'{val_acc:.6f}', status, description])
    return

if __name__ == '__main__':
    load_data()