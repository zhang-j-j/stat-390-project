## AutoResearch Agent Instructions

---

## Objective

Maximize **balanced accuracy** on the gene probe classification task.

## Rules

1. You may **ONLY** modify sections of `model.py` that are marked as `EDITABLE`
2. `prepare.py` is **FROZEN** - do not touch it
3. All other subdirectories are **FROZEN** - do not touch them
4. `build_model()` must return an sklearn-compatible estimator (Pipeline preferred)
5. Training + evaluation must complete in **under 3 minutes (180 seconds)** on CPU
6. No additional data sources or external downloads

## Workflow

At the start of each run, set the environment variables for stable CPU timing

```
$env:TF_CPP_MIN_LOG_LEVEL='2'
$env:TF_ENABLE_ONEDNN_OPTS='0'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
```

1. Set the user-specified results file path (within the `results/` subdirectory) in `model.py`
2. Read current `model.py`
3. Propose a modification and briefly describe your reasoning
4. Edit `model.py`
5. Run: `python model.py "description of change"`
6. Check `val_acc` in output and log the results
7. If improved: Keep the modifications to `model.py`
8. If worse: Revert `model.py` to the initial state before the modification
9. If time limit exceeded: Record the experiment as a failure and revert
10. Repeat from step 2 for the number of iterations that the user specifies

## Ideas to explore

- Different classifiers: LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, SVC, MLPClassifier
- Ensemble methods: RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
- Preprocessing: StandardScalar, RobustScaler, QuantileTransformer
- Hyperparameter tuning within the pipeline
- Dimension reduction methods within the `sklearn.decomposition` module and UMAP: PCA, KernelPCA, FastICA, UMAP
- Autoencoder dimension reduction: Hyperparameter tuning within the `AutoencoderTransformer` class

## What NOT to do

- Do not modify `prepare.py` (data split, metric)
- Do not modify the `AutoencoderTransformer` class in `model.py`
- Do not run an experiment without logging it, even if it is too slow or fails
- Do not import unnecessary modules into `model.py`
- Do not add new files or dependencies
- Do not hard-code validation data into the model
- Do not change the function signature of `build_model()` or `run_model()`
