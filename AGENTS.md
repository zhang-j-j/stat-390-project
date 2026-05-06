# AI Agent Instructions for STAT 390 Autoresearch Project

## Project Overview

This is an **autoresearch framework** for optimizing dimension reduction in gene expression data. The goal is to maximize **balanced accuracy** on a probe classification task by iteratively improving the machine learning pipeline.

**Key metric:** Balanced accuracy (higher is better, target > 0.24)  
**Dataset:** Allen Brain Atlas microarray data (Donor 9861)  
**Runtime budget:** Training + evaluation must complete in <60 seconds on CPU

See [README.md](README.md) for data setup and [program.md](program.md) for detailed agent instructions.

---

## Repository Structure & Constraints

### FROZEN Directories (Do Not Modify)
- **`prepare.py`** — Data loading, train/val split (80/20), evaluation metric, and logging. Contains `load_data()` and `evaluate()` functions.
- **`manual/`** — Initial exploration, baseline models, and analysis notebooks. Reference only.
- **`data/`** — Raw and cleaned datasets. Reference only.
- **`report/`** — LaTeX documents and final writeup.

### EDITABLE Files (Agent-Improved)
- **`model.py`** — Contains the `build_model()` function that agents modify. This is the only place to change the ML pipeline. The wrapper code for running, training, and logging is frozen.

---

## The Build Model Function

### Current State
```python
def build_model():
    """EDITABLE: Return an sklearn Pipeline. This is what the agent improves."""
    return Pipeline([
        ('pca', PCA(n_components=250)),
        ('knn', KNeighborsClassifier(n_neighbors=8))
    ])
```

### Constraints
1. **Must return an sklearn-compatible estimator** (Pipeline or single estimator)
2. **Allowed preprocessing:** `sklearn.decomposition` (PCA, FastICA, etc.) and `umap` (UMAP)
3. **Allowed classifiers:** Any from `sklearn` (LogisticRegression, KNeighborsClassifier, SVC, etc.)
4. **Hyperparameter tuning:** Allowed within the pipeline
5. **Function signature:** Must match `def build_model(): return ...` (no arguments)
6. **No hard-coding validation data or external downloads**
7. **Runtime constraint:** Full training + evaluation must be <60 seconds

---

## The Autoresearch Workflow

### Step 1: Propose a Change
Before modifying `model.py`, understand the current model and propose a specific change (e.g., "try UMAP instead of PCA", "increase KNN neighbors to 10", "add StandardScaler preprocessing").

### Step 2: Edit `model.py`
Modify only the `build_model()` function. Keep all other code unchanged.

### Step 3: Run the Experiment
```powershell
python model.py "description of change"
```

The output will show:
- Training time (must be <60s)
- `val_acc` — The balanced accuracy score
- Status message confirming the log was recorded

### Step 4: Check the Result
Look at the `val_acc` value and compare to the current best in `results.tsv`.

### Step 5: Commit or Revert
- **If improved:** `git add model.py && git commit -m "feat: description of change"`
- **If worse or equal:** `git checkout model.py` to revert

### Step 6: Repeat
Go back to Step 1 and propose the next change.

---

## Baseline Results

Current performance benchmarks from initial exploration:

| Model Specification | CV Performance |
|---|---|
| Baseline (KNN k=1) | 0.242 |
| Baseline (KNN k=9) | 0.240 |
| PCA (n=250) + KNN (k=8) | 0.241 |
| PCA (n=100) + KNN (k=9) | 0.241 |

The current model in `model.py` achieves **~0.241 balanced accuracy**. The goal is to exceed this through dimension reduction and classifier improvements.

---

## Ideas to Explore

See [program.md](program.md) for the full research plan. High-impact areas:

- **Dimension Reduction:** UMAP, FastICA, FactorAnalysis, try different `n_components`
- **Classifiers:** GradientBoostingClassifier, RandomForestClassifier, SVC with RBF, HistGradientBoostingClassifier
- **Preprocessing:** StandardScaler, RobustScaler, QuantileTransformer
- **Ensemble methods:** VotingClassifier combining multiple approaches
- **Pipeline combinations:** Different orderings of scaling → reduction → classification

---

## Key Files for Reference

- [README.md](README.md) — Data setup, project structure, how to run
- [program.md](program.md) — Full agent instructions and research objectives
- `model.py` — The editable pipeline (contains only `build_model()` and FROZEN runner code)
- `prepare.py` — Data loading and evaluation (reference only, FROZEN)
- `results.tsv` — Experiment log (auto-generated, tracks all runs)
- `manual/initial_exploration.ipynb` — Baseline and data exploration (reference only)

---

## Environment Setup

The project uses:
- **Package manager:** `uv` (recommended) or `pip`
- **Python:** ≥3.12
- **Key dependencies:** numpy, pandas, scikit-learn, umap-learn, matplotlib, seaborn, jupyter

Setup commands:
```powershell
uv venv          # Create virtual environment
uv sync          # Install dependencies
# Or with pip:
pip install -r requirements.txt  # (if available)
```

The virtual environment is typically activated at `.venv/Scripts/Activate.ps1` (Windows).

---

## Common Pitfalls

1. **Modifying frozen files** — Only edit `build_model()` in `model.py`
2. **Changing function signature** — `build_model()` must take no arguments and return an estimator
3. **Adding dependencies** — Stick to packages already in `pyproject.toml`
4. **Exceeding runtime budget** — Complex models may exceed 60s; monitor carefully
5. **Hard-coding data splits** — `load_data()` is frozen; let it handle the split
6. **Forgetting git commits** — Commit successful experiments to track progress

---

## Quick Reference: Run a New Experiment

```powershell
# 1. Edit model.py build_model() function
# 2. Save the file
# 3. Run the experiment
python model.py "my experiment description"

# 4. Check output (val_acc) in results.tsv
# 5. If good: commit
git add model.py
git commit -m "feat: my experiment description"
# 6. If bad: revert
git checkout model.py
```

---

## Next Steps for Agents

1. **Understand the current model** by reading `model.py` and the baseline results
2. **Review past experiments** in `results.tsv` to see what has been tried
3. **Propose a single focused change** to the pipeline
4. **Run, evaluate, and commit or revert** before moving to the next idea
5. **Iterate** on the most promising research directions

Good luck optimizing the dimension reduction pipeline!
