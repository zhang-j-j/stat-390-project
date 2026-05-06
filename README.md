## Autoresearch for Dimension Reduction in Gene Expression

high level description (fill in later)

### Overview

- research goal: compute a low-dimensional representation of gene expression data that optimizes classification performance
- metric: balanced accuracy
- data: from [Allen Brain Atlas](https://human.brain-map.org/static/download) (Dataset H0351.2001)

### Repository structure

- key files shown

```
stat-390-project/
├── data/                               # microarray dataset from one brain (Donor 9861)
    └── raw/                            # raw data files
        └── ...                         
    └── cleaned/                        # possible additional datasets from other brains
        └── ...                         # cleaned and split datasets
├── manual/                             # FROZEN - code for manual tasks
    ├── initial_exploration.ipynb       # initial data exploration and preprocessing
    ├── baseline.ipynb                  # manually implemented baseline model
    ├── final_analysis.ipynb            # analysis code of final results
    └── ...                             # additional files as needed
├── model.py                            # EDITABLE - agent-improved model code
├── prepare.py                          # FROZEN - initial model setup
├── program.md                          # FROZEN - user-defined agent instructions
└── results.tsv                         # GENERATED - logged experiment results
```

### How to run the project

#### Setup

Download the dataset from the [Allen Brain Atlas](https://human.brain-map.org/static/download) (Dataset H0351.2001). Extract the zip file and directly copy into `data/raw/`.

Using `uv` package manager (recommended):

1. Install `uv` (PowerShell on Windows or curl on macOS/Linux):

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# or with pip / pipx
python -m pip install --upgrade pip
pip install uv
# or: pipx install uv
```

2. Quick `uv` project workflow (run these in the project root):

```powershell
# initialize or register the current directory as a uv project
uv init .

# create a virtual environment (uv will create `.venv`)
uv venv

# reproduce the environment from `pyproject.toml`
# create a virtual environment (uv will create `.venv`)
uv venv

# create a reproducible lockfile from `pyproject.toml`
uv lock

# install from the lockfile / sync the environment
uv sync
```

#### Baseline

Ensure that the file directories are correctly orgnized for the `data/raw/` folder. Then, run the notebook `manual/initial_exploration.ipynb` to obtain the following baseline results.

| Model specification | Fitting time (seconds) | Cross-validation performance |
| --- | ---: | ---: |
| Baseline (KNN with k=1) | 0.207 | 0.242 |
| Baseline (KNN with k=9) | 0.265 | 0.240 |
| PCA KNN (n=250, k=8) | 4.72 | 0.241 |
| PCA KNN (n=100, k=9) | 3.81 | 0.241 |



#### Autoresearch Loop

The folllowing results were obtained from the first autoresearch loop trial.

| Exp | Model Configuration | Val Acc | Status | Change vs Prev | Notes |
|-----|---------------------|---------:|--------|---------------:|-------|
| 0 | Baseline — PCA(250) + KNN(k=8) | 0.240397 | KEPT | — | Initial baseline |
| 1 | PCA(250) + KNN(k=10) | 0.243506 | KEPT | +0.003109 | Increased KNN neighbors |
| 2 | PCA(100) + KNN(k=10) | 0.242960 | REVERTED | -0.000546 | Fewer components worse |
| 3 | PCA(250) + StandardScaler + LogisticRegression | 0.264977 | KEPT | +0.021 (vs baseline) | Major improvement |
| 4 | PCA(250) + RandomForest(n=30) | 0.238411 | REVERTED | -0.026566 | Worse than baseline |
| 5 | PCA(150) + LogisticRegression | 0.263351 | REVERTED | -0.001626 | Fewer components worse |
| 6 | PCA(250) + StandardScaler + LogisticRegression(C=0.1) | 0.265090 | KEPT | +0.000113 | Best so far |
| 7 | PCA(250) + StandardScaler + LogisticRegression(C=0.01) | 0.264630 | REVERTED | -0.000460 | Stronger regularization hurt |
| 8 | PCA(250) + RobustScaler + LogisticRegression(C=0.1) | 0.264764 | REVERTED | -0.000326 | StandardScaler slightly better |
| 9 | PCA(200) + StandardScaler + LogisticRegression(C=0.1) | 0.264355 | REVERTED | -0.000735 | 250 components preferred |
| 10 | PCA(250) + StandardScaler + SVC(RBF) | TIMEOUT | REVERTED | — | Exceeded 60s runtime budget |


