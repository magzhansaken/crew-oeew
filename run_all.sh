#!/usr/bin/env bash
# Reproduce every number, table and figure of the paper (CPU only, 2 cores, 8 GB RAM; about 10 h in total).
# Run the steps one after another: two LightGBM jobs at once oversubscribe the cores and slow both down.
set -euo pipefail
cd "$(dirname "$0")"
python -m oeew.data                                    # 1. download OULAD (UCI 349) and verify SHA-256
pytest                                                 # 2. unit and leakage tests
R="python experiments/run_tabular.py --seeds 5"        # 3. tabular models x protocols x prediction days
ALLDAYS="7 14 28 42"
$R --days $ALLDAYS --protocols P1 --methods lim_lr lim_rf lim_hgb lgbm_lim lgbm_iw lgbm_abs_bal cat_abs lgbm_abs crew crew_lim
$R --days $ALLDAYS --protocols P2 --methods lim_lr lim_rf lim_hgb lgbm_lim lgbm_abs crew
$R --days 28 --protocols P3 --methods lim_lr lgbm_abs crew
$R --days 28 14 --protocols P1 --methods crew_rel
$R --days 28 --protocols P1 --methods lgbm_abs_noCAL lgbm_abs_noTRD lgbm_abs_noPRE crew_noCAL lim_lr_vle
python experiments/run_gru2.py --days $ALLDAYS --protocols P1 --seeds 5   # 4. GRU models (PyTorch, CPU)
python experiments/evaluate.py                         # 5. calibration and base-rate schemes, metrics
python experiments/analysis.py --boot 1000             # 6. paired tests, bootstrap intervals, prior effect
python experiments/subgroups.py --boot 1000            # 7. learner subgroups
python experiments/ablation.py --boot 1000             #    ablation contrasts (day 28)
python experiments/cost_bench.py --all                 #    cost per new course (run with no other job active)
python experiments/explain.py --day 28 --method crew   # 8. SHAP attributions and their stability across folds
python experiments/explain.py --day 14 --method crew
python experiments/explain.py --day 28 --method lgbm_abs
python experiments/explain.py --day 28 --compare crew lgbm_abs
python experiments/dataset_stats.py                    # 9. data description numbers
python experiments/derived_numbers.py                  # 10. derived numbers quoted in the text
python experiments/figures.py                          # 11. figures (vector PDF) with a CSV table view each
python experiments/tables.py                           # 12. LaTeX tables
