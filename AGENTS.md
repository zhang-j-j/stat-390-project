# AGENTS.md — Agent instructions for `stat-390-project`

Purpose: give AI coding agents concise, actionable guidance so they can be productive immediately.

Quick links
- Repository README: [README.md](README.md)
- Primary editable entrypoint: [model.py](model.py)
- Data loader and evaluation (FROZEN): [prepare.py](prepare.py)
- Research plan / agent instructions: [program.md](program.md)
- Data folder: [data/](data)
- Manual scripts (FROZEN): [manual/](manual)
- Results: [results/results.tsv](results/results.tsv)

Quick start (recommended)
- Preferred: use `uv` (see [README.md](README.md) for uv steps).
- Fallback (Windows PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# install project deps listed in pyproject.toml (manually or via a lockfile)
```

How to run an experiment
- Run: `python model.py "short description"` — this runs one experiment and appends to [results/results.tsv](results/results.tsv).
- Baseline run: `python model.py "desc" --baseline`.
- `model.py` contains a top-level `RUN_ID` (edit as appropriate) and an editable `build_model()` function.

Conventions & important notes
- Files marked FROZEN: do not change `prepare.py` or files in `manual/` unless explicitly asked.
- Editable model code: change `model.py` (implement new `dr_model` / `cls_model` in `build_model()`).
- Data: `prepare.py` expects `data/cleaned/exp_train.pkl`. If missing, run the preprocessing notebook/script in [manual/preprocess.py](manual/preprocess.py).
- Results are appended to [results/results.tsv](results/results.tsv). Agents should read and update that file atomically (append-only behavior is expected).
- Evaluation: `prepare.evaluate()` returns `(silhouette, prec@10, balanced_accuracy)`. Use these metrics to decide whether to `keep` or `discard` as the project logic does.

Agent behavior guidelines
- Link, don't duplicate: reference files above rather than copying large docs into this file.
- Minimal edits: prefer iterative changes to `model.py` (small, testable commits) rather than big rewrites.
- Reproducible environment: prefer the `uv` workflow from [README.md](README.md). If changes add new deps, update `pyproject.toml`.
- Data safety: do not commit large raw datasets; keep them in `data/raw/` and document steps to reproduce cleaned data in `manual/preprocess.py`.

Suggested next customizations (optional)
- Create a small `agent-runner` skill that:
  - Runs `python model.py` with a provided description and captures `results/results.tsv` output.
  - Validates presence of `data/cleaned/exp_train.pkl` and runs the preprocessing step if missing.

Feedback
If this looks good, I can: create an automated `agent-runner` skill, or add a `.github/copilot-instructions.md` with the same content for GitHub-native agents.
