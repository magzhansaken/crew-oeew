# CREW: cohort-relative early warning for courses without outcome labels

Code, tests and result files for the paper *"[title withheld for double-blind review]"* (submitted to IJACSA, 2026).

The task is early warning of course non-success (Fail or Withdrawn) for a course that has **no outcome labels yet**,
for example the first run of a course on an open-education platform. The model is trained on other
courses only, and on prediction day *d* it may use the activity logged so far by the cohort of the new course, without any of its labels.

CREW has three label-free parts:

1. **Design-agnostic behaviour features**: the assessment calendar (deadlines due, deadlines missed, days to the next
   deadline), activity trajectory (trend, gaps, longest inactive spell) and activity before the official start.
2. **Cohort-relative encoding**: each behavioural feature is re-expressed as a percentile rank and a z-score within the
   active cohort (risk set) of the same course presentation on day *d*. CREW applies it in LightGBM (feature set and
   hyperparameters chosen by nested course-disjoint selection inside the training modules); CREW-GRU applies it to
   daily activity sequences. Platt scaling fitted on learner-grouped out-of-fold scores from the training modules calibrates every model,
   CREW included.
3. **Gated cohort prior** for any model: a ridge regression from five unlabelled cohort statistics to the outcome rate,
   fitted on training presentations, shifts calibrated probabilities on the logit scale. It is applied to a target only
   when a nested check on course-disjoint groups of the training modules lowers the per-presentation calibration error.

All comparisons use the protocol of Lim (IJACSA 17(9), 2026): leave-one-module-out with removal of every learner of the
held-out module from training (P1). A forward protocol (P2, 2013 → 2014) and a random learner-disjoint protocol (P3)
are reported as boundary conditions.

## Data

Open University Learning Analytics Dataset (OULAD): Kuzilek, Hlosta and Zdrahal, *Scientific Data* 4:170171 (2017),
distributed by the UCI Machine Learning Repository (dataset 349, DOI [10.24432/C5KK69](https://doi.org/10.24432/C5KK69))
under CC BY 4.0. The data are **not** redistributed here. `python -m oeew.data` downloads the official archive and
verifies its SHA-256 (`f2ed1902…c6d3e4`); the experiments stop if the hash differs.

| Item | Value |
|---|---|
| Enrolments / learners | 32,593 / 28,785 |
| Modules / presentations | 7 / 22 (2013B–2014J) |
| VLE interaction rows | 10,655,280 |
| Target | 1 = Fail or Withdrawn, 0 = Pass or Distinction |
| Risk set on day *d* | enrolments not unregistered before *d*; features use records dated before *d* only |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test,seq]"           # exact versions: requirements-lock.txt
python -m oeew.data                    # download + SHA-256 check (about 46 MB)
pytest                                 # unit tests and leakage tests (about 20 s)
./run_all.sh                           # every number, table and figure (about 10 h on 2 CPU cores, 8 GB RAM)
```

No GPU is needed. The GRU baselines run on CPU with PyTorch.

## Repository layout

| Path | Content |
|---|---|
| `src/oeew/data.py` | download, hash check, loading, target |
| `src/oeew/features.py` | leakage-safe features on day *d*: REG, VLE, SUB (Lim, 2026), CAL, TRD, PRE, REL (ours); daily sequences |
| `src/oeew/protocols.py` | P1 leave-one-module-out, P2 forward, P3 random; all learner-disjoint; inner course-disjoint folds |
| `src/oeew/models.py` | LR, RF, HGB with Lim's settings; LightGBM and CatBoost grids |
| `src/oeew/calibration.py` | Platt, isotonic, EM prior correction, cohort statistics, cohort prior |
| `src/oeew/metrics.py` | AUROC, AUPRC, Brier, ECE, recall at 20%, learner-cluster bootstrap, Holm |
| `src/oeew/seqmodel.py` | GRU baseline over daily activity sequences |
| `experiments/run_tabular.py` | tabular models × protocols × days → row-level predictions (`results/preds`) and training OOF scores (`results/oof`) |
| `experiments/run_gru.py` | first GRU runner; its `cohort_normalise` is reused by `run_gru2.py` (its own runs are not used in the paper) |
| `experiments/run_gru2.py` | GRU models over daily sequences, absolute and cohort-normalised (CREW-GRU), with the same OOF calibration |
| `experiments/evaluate.py` | calibration and base-rate schemes (Platt, cohort prior, gated prior, EM, ACC/BBSE, oracle), metrics (`results/eval`) |
| `experiments/analysis.py` | paired tests over presentations, learner-bootstrap intervals, prior effect, cost, `numbers.json` |
| `experiments/cost_bench.py` | training and inference time and peak memory per new course, one process per model |
| `experiments/ablation.py` | ablation contrasts with learner-bootstrap intervals |
| `experiments/subgroups.py` | results by gender, disability, age band and IMD band with bootstrap intervals |
| `experiments/explain.py` | SHAP attributions and their stability across held-out modules |
| `experiments/dataset_stats.py`, `experiments/derived_numbers.py` | data description and derived numbers quoted in the paper |
| `experiments/figures.py`, `experiments/tables.py` | figures (vector PDF, each with a CSV table view) and LaTeX tables |
| `tests/` | learner overlap, module hold-out, forward split, future-record and label-permutation leakage tests, metric formulas |

## What prevents leakage

* Every split removes all enrolments of test learners from training (`tests/test_protocols.py`).
* Features on day *d* are unchanged when all records dated on or after *d* are altered (`tests/test_leakage.py`).
* Features, cohort-relative transforms and cohort statistics are unchanged when outcome labels are permuted
  (`tests/test_leakage.py`): the target course's labels are never read before evaluation.
* Hyperparameters, feature-set choice, calibrators, the cohort prior and the calibration scheme are fitted inside the
  training modules of each outer split (nested course-disjoint folds).
* The oracle prevalence shift (`+oracle`) uses target labels and is reported only as an upper bound.

## Results

Held-out-module protocol P1, day 28, Platt calibration for every model (27,553 target enrolments; mean of five seeds):

| Model | AUROC | ECE | Module-mean ECE |
|---|---|---|---|
| LR, benchmark of Lim (2026), reimplemented | 0.660 | 0.065 | 0.143 |
| CatBoost on all absolute features (strongest baseline) | 0.706 | 0.049 | 0.103 |
| LightGBM on all absolute features (ablation) | 0.709 | 0.042 | 0.093 |
| CREW-GRU | 0.711 | 0.016 | 0.086 |
| **CREW** | **0.724** | 0.018 | **0.078** |

Every number, table and figure of the paper is generated from the files in `results/analysis/` and
`results/eval/metrics.parquet`; `results/analysis/numbers.json` and `results/analysis/extra_numbers.json` hold each
quoted number under a key. Row-level predictions and out-of-fold scores (`results/preds/*.parquet`, `results/oof/`,
`results/eval/rows/`, about 1 GB) are not stored in the repository because of their size; `run_all.sh` regenerates them.

## Licence and citation

Code: MIT (see `LICENSE`). Data: CC BY 4.0, by the OULAD authors; please cite Kuzilek et al. (2017).
Citation metadata for this repository: `CITATION.cff`.
