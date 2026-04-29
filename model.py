"""
Run one experiment: build model, train, evaluate, log result.

Usage:
    python model.py "description"              # logs as status=keep
    python model.py "description" --baseline   # logs as status=baseline
    python model.py "description" --discard    # logs as status=discard
"""

import subprocess
import sys
import time
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from prepare import load_data, evaluate, log_result

def get_git_hash():
    """
    FROZEN: Track the experiment version
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "no-git"

def build_model():
    """
    EDITABLE: Return an sklearn Pipeline. This is what the agent improves.
    """
    return Pipeline([
        ('pca', PCA(n_components=250)),
        ('knn', KNeighborsClassifier(n_neighbors=8))
    ])

def run_model():
    """
    FROZEN: Run and evaluate model, and log results
    """

    # setup
    args = sys.argv[1:]
    status = "keep"
    description_parts = []
    for a in args:
        if a == "--baseline":
            status = "baseline"
        elif a == "--discard":
            status = "discard"
        else:
            description_parts.append(a)
    description = " ".join(description_parts) if description_parts else "experiment"

    # load data
    X_train, X_val, y_train, y_val = load_data()

    # train model
    model = build_model()
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.2f} seconds")

    # evaluate model and log results
    acc = evaluate(model, X_val, y_val)
    commit = get_git_hash()
    log_result(
        experiment_id='baseline',
        val_acc=acc,
        status=status,
        description=description
    )
    print(f"Result logged to results.tsv (status={status})")

if __name__ == '__main__':
    run_model()
