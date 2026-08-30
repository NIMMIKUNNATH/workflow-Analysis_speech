"""
match_conditions.py — resolve ambiguous condition <-> clinical_*.csv pairings.

Both study_log.csv and the clinical_*.csv files record a per-case `wer` derived
from the same transcription runs. So a correct pairing is one where the per-case
WER vectors agree over the shared cases. This matches on the data itself rather
than on filenames, which is the only safe way to resolve names like
'baseline' vs 'clinical_baseline40.csv' vs 'clinical_full.csv'.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = Path(os.environ.get("ASR_RESULTS_ROOT", REPO_ROOT / "results"))
R = Path(os.environ.get("ASR_CLINICAL_RESULTS_ROOT", RESULTS_ROOT / "clinical"))
STUDY_LOG = Path(os.environ.get("ASR_STUDY_LOG", RESULTS_ROOT / "study_log.csv"))

UNMATCHED_CONDITIONS = ["baseline", "compute_fp16", "model_medium"]
ORPHAN_FILES = ["clinical_272", "clinical_baseline40", "clinical_full", "clinical_medium40"]

study = pd.read_csv(STUDY_LOG)

print("=== case counts per condition in study_log.csv ===")
print(study.groupby("condition")["case"].nunique().to_string())

print("\n=== orphan clinical files ===")
clin = {}
for name in ORPHAN_FILES:
    path = R / f"{name}.csv"
    if not path.exists():
        print(f"{name}: MISSING")
        continue
    d = pd.read_csv(path)
    clin[name] = d
    has_neg = "negation_n" in d.columns
    print(f"{name:26s} rows={len(d):4d}  mean_wer={d['wer'].mean():.4f}  "
          f"negation_col={has_neg}")

print("\n=== fingerprint match (per-case WER agreement) ===")
print(f"{'condition':18s} {'file':26s} {'n_shared':>8s} {'max_abs_diff':>13s} {'verdict':>10s}")

for cond in UNMATCHED_CONDITIONS:
    s = study[study["condition"] == cond][["case", "wer"]].dropna()
    if s.empty:
        print(f"{cond:18s} -- no rows in study_log --")
        continue
    s = s.set_index("case")["wer"]

    best = None
    for name, d in clin.items():
        if "case" not in d.columns:
            continue
        c = d[["case", "wer"]].dropna().set_index("case")["wer"]
        shared = s.index.intersection(c.index)
        if len(shared) == 0:
            continue
        diff = float(np.abs(s.loc[shared].values - c.loc[shared].values).max())
        verdict = "MATCH" if diff < 1e-4 else ("close" if diff < 0.01 else "")
        print(f"{cond:18s} {name:26s} {len(shared):8d} {diff:13.6f} {verdict:>10s}")
        if best is None or diff < best[1]:
            best = (name, diff)
    if best and best[1] < 1e-4:
        print(f"  -> {cond} maps to {best[0]}")
    else:
        print(f"  -> {cond}: NO CONFIDENT MATCH (best {best[0]} at {best[1]:.4f})"
              if best else f"  -> {cond}: no comparable file")
    print()
