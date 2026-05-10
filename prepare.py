"""
FROZEN -- Do not modify this file.
Data loading, train/val split, evaluation metric
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, silhouette_score
import csv
import os

# constants
DATA_PATH = 'data/cleaned/exp_train.pkl'
RANDOM_STATE = 42
VAL_FRACTION = 0.2
RESULTS_PATH = 'results/results.tsv'

# load data
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

# precision @ k (k=10)
def precision_at_k(X_emb, labels, k=10):
    """"
    Compute precision@k (k=10) on the embddings
    """
    from sklearn.neighbors import NearestNeighbors

    labels = np.asarray(labels)
    neighbors = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(X_emb)
    indices = neighbors.kneighbors(X_emb, return_distance=False)

    precision_scores = []
    
    for i in range(X_emb.shape[0]):
        neighbors = indices[i, 1:]  # exclude self
        precision = np.mean(labels[neighbors] == labels[i])
        precision_scores.append(precision)

    return np.mean(precision_scores)

# evaluation
def evaluate(dr_model, cls_model, X_val, y_val):
    """
    Compute validation metrics: silhouette score, precision@10, and balanced accuracy
    """
    X_val_dr = dr_model.transform(X_val)
    y_pred = cls_model.predict(X_val_dr)

    sil = silhouette_score(X_val_dr, y_val)
    prec_10 = precision_at_k(X_val_dr, y_val)
    acc = balanced_accuracy_score(y_val, y_pred)
    return sil, prec_10, acc

# logging
def log_result(run_id, train_time, val_metrics, description, baseline=False):
    """
    Append one row to results.tsv
    """
    sil, prec_10, acc = val_metrics

    if baseline:
        status = 'baseline'
    else:
        # determine keep/discard using val_sil and val_prec_10 rules
        status = 'keep'
        best_sil = float('-inf')
        best_prec = float('-inf')

        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                for parts in reader:
                    if len(parts) <= 3:
                        continue
                    try:
                        vs = float(parts[2])
                        vp = float(parts[3])
                    except Exception:
                        continue
                    if vs > best_sil:
                        best_sil = vs
                    if vp > best_prec:
                        best_prec = vp

        if best_sil == float('-inf'):
            status = 'keep'
        else:
            delta_sil = sil - best_sil
            if delta_sil >= 0.005:
                status = 'keep'
            elif 0 < delta_sil < 0.005 and prec_10 > best_prec:
                status = 'keep'
            else:
                status = 'discard'

    file_exists = os.path.exists(RESULTS_PATH)
    with open(RESULTS_PATH, 'a', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        if not file_exists:
            writer.writerow(['experiment', 'train_time', 'val_sil', 'val_prec_10', 'val_acc', 'status', 'description'])
        writer.writerow([run_id, f'{train_time:.2f}', f'{sil:.6f}', f'{prec_10:.6f}', f'{acc:.6f}', status, description])
    return

if __name__ == '__main__':
    load_data()
