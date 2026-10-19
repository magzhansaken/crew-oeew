"""Figure style matched to the SAI/IJACSA template: Times-like serif 8 pt, single column 3.5 in, vector PDF."""
from __future__ import annotations

import matplotlib as mpl
from matplotlib import font_manager

for f in font_manager.findSystemFonts():
    if "texgyretermes" in f.lower():
        font_manager.fontManager.addfont(f)

COL_W = 3.5    # inches, one column of the two-column template
FULL_W = 7.16  # inches, two columns

# Okabe-Ito palette (colour-blind safe); every series also differs by marker or line style for grey-scale print
C = {"crew": "#D55E00", "lgbm_abs": "#E69F00", "lgbm_lim": "#0072B2", "lim_lr": "#000000", "lim_hgb": "#56B4E9",
     "lim_rf": "#999999", "cat_abs": "#009E73", "gru_abs": "#CC79A7", "gru_rel": "#CC79A7", "lgbm_iw": "#F0E442"}
MK = {"crew": "o", "lgbm_abs": "s", "lgbm_lim": "^", "lim_lr": "D", "lim_hgb": "v", "lim_rf": "x", "cat_abs": "P",
      "gru_abs": "*", "gru_rel": "h", "lgbm_iw": "<"}
LS = {"crew": "-", "lgbm_abs": "--", "lgbm_lim": "-.", "lim_lr": ":", "lim_hgb": (0, (3, 1, 1, 1)), "lim_rf": ":",
      "cat_abs": "--", "gru_abs": (0, (5, 2)), "gru_rel": (0, (1, 1)), "lgbm_iw": "-."}
NAME = {"crew": "CREW (ours)", "lgbm_abs": "LightGBM, absolute (ours)", "lgbm_lim": "LightGBM, Lim features",
        "lim_lr": "LR [Lim]", "lim_hgb": "HGB [Lim]", "lim_rf": "RF [Lim]", "cat_abs": "CatBoost, absolute",
        "gru_abs": "GRU, sequences", "gru_rel": "GRU, cohort-normalised", "lgbm_iw": "LightGBM, importance-weighted",
        "lgbm_abs_bal": "LightGBM, module-balanced", "crew_lim": "Relative, Lim features", "crew_rel": "Relative only"}


def apply():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
        "lines.linewidth": 1.0, "lines.markersize": 3.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
        "axes.grid": True, "grid.linewidth": 0.3, "grid.color": "#d0d0d0", "axes.axisbelow": True,
    })
