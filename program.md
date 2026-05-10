## Autoresearch Agent Instructions

---

## Objective

You are an autoresearch agent. Your goal is to produce the **best dimension reduction** model for the gene probe data.

- Focus on identifying a low-dimensional representation of the gene expression data that maximizes **silhouette score** and **precision@10**.
- Ultimately, I will assess overall performance using **balanced accuracy** in the classification task.

## Rules

1. You may **ONLY** modify sections of `model.py` that are marked as `EDITABLE`
2. `prepare.py` is **FROZEN** - do not touch it
3. All other subdirectories are **FROZEN** - do not touch them
4. `build_model()` must return a tuple of 2 sklearn Pipelines - `(dr_model, cls_model)`
   1. `dr_model` is a dimensionality reduction model
   2. `cls_model` is a classifier
5. Training + evaluation must complete in **under 10 minutes (600 seconds)** on CPU
6. No additional data sources or external downloads
7. Document **all** of your  in `results/agent_experiment_notes.md`

## Workflow

At the start of each run, set the environment variables for stable CPU timing.

```
$env:TF_CPP_MIN_LOG_LEVEL='2'
$env:TF_ENABLE_ONEDNN_OPTS='0'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
```

1. Set the user-specified run ID in `model.py`
2. Read current `model.py` and propose a modification to `build_model()`
3. Edit `model.py`
   1. After making your modifications to `build_model()`, remove any unnecessary imports from the top of the `model.py` script
4. Run `python model.py "description of change"` with `mode='sync'`  and `timeout=600000` (10 minutes = 600 seconds)
   1. If the runtime exceeds the allotment, terminate the experiment and manually log the entry to `results/results.tsv` (in TSV format): `[run_id], 600.00+, ---, ---, ---, fail, [description]`
5. Evaluate the experiment success according to silhouette score and precision@10
   1. If the silhouette score improves by at least 0.005, **keep** the modifications as a successful experiment
   2. If the silhouette score improves by between 0-0.005 and precision@10 improves, **keep** the modifications as a successful experiment
   3. Otherwise, **discard** the experiment and revert `model.py` to the initial state before the modification
   4. If the time limit is exceeded, record the experiment as a **failure** (according to step 4.1 above) and revert `model.py` to the initial state before the modification
   5. **Note:** If the only experiment modification was to the classification model (`cls_model`), silhouette score and precision@10 will not change; in this case, **keep** the modifications as a successful experiment if balanced accuracy increases by at least 0.005, otherwise **discard** and revert
6. Log the experiment results according to the procedure in the **Documentation** section below
7. Repeat from step 2 for the number of iterations that the user specifies

### Documentation

Create a new section in `results/agent_experiment_notes.md` labeled with the run ID for every new autoresearch loop. Then, for each experiment, complete the following steps:

1. Clearly describe what is being changed from the prior experiment
2. Briefly explain why this change is being made
3. After running the experiment, log the results in `results/agent_experiment_notes.md`
   1. Performance metrics: silhouette score, precision@10, balanced accuracy
   2. Model runtime (if the model exceeds the 10 minute runtime, clearly note this as a **failed experiment**)
   3. Experiment status (**keep/revert/fail**)

At the end of each autoresearch loop, create a summary table of all experiments that you ran. Include a short description of the experiment, performance metrics, model runtime, and experiment status.

## Ideas to explore

### Dimension Reduction

- Preprocessing: StandardScalar, RobustScaler, QuantileTransformer
- Dimension reduction methods within `sklearn.decomposition`: PCA, KernelPCA, FastICA, DictionaryLearning
- Dimension reduction using UMAP: UMAP, Parametric UMAP, Supervised UMAP
- Autoencoder dimension reduction: Test different architectures using the built-in `AutoencoderTransformer` class
- Hyperparameter tuning within each step
- Combinations of the above steps

### Classifier

- Preprocessing: StandardScalar, RobustScaler, QuantileTransformer
- Hyperparameter tuning for the LogisticRegression classifier

## What NOT to do

- Do not modify `prepare.py` (data split, evaluation, result logging)
- Do not modify the `AutoencoderTransformer` class in `model.py`
- Do not change the classifier model from LogisticRegression
- Do not proceed to a new experiment until the previous experiment was logged, even if it failed
- Do not run an experiment for longer than the allotted time
- Do not import unnecessary modules into `model.py`
- Do not add new files or dependencies
- Do not hard-code validation data into the model
- Do not change the function signature of `build_model()` or `run_model()`
