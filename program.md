## AutoResearch Agent Instructions

---

## Objective

Maximize **balanced accuracy** on the gene probe classification task.

## Rules

1. You may **ONLY** modify sections of `model.py` that are marked as `EDITABLE`
2. `prepare.py` is **FROZEN** - do not touch it
3. All other subdirectories are **FROZEN** - do not touch them
4. `build_model()` must return an sklearn-compatible estimator (Pipeline preferred)
5. Training + evaluation must complete in **under 60 seconds** on CPU
6. No additional data sources or external downloads

## Workflow

1. Read current `model.py`
2. Propose a modification
3. Edit `model.py`
4. Run: `python model.py "description of change"`
5. Check `val_acc` in output
6. If improved: `git add model.py && git commit -m "feat: <description>"`
7. If worse: `git checkout model.py` (revert)
8. Repeat from step 1

## Ideas to explore

- Different classifiers: LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, SVC, MLPClassifier
- Ensemble methods: RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
- Dimension reduction methods **within the `sklearn.decomposition` module** and **UMAP**: PCA, FastICA, UMAP
- Preprocessing: StandardScalar, RobustScaler, QuantileTransformer
- Hyperparameter tuning within the pipeline

## What NOT to do

- Do not modify `prepare.py` (data split, metric)
- Do not add new files or dependencies
- Do not hard-code validation data into the model
- Do not change the function signature of `build_model()`
- Do not try dimension reduction techniques outside of `sklearn.decomposition` and `umap` (DO NOT use autoencoders)
