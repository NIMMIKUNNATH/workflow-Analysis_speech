#!/usr/bin/env python3
"""
selection_validity_bootstrap.py
-------------------------------
Answers the winner's-curse objection: Rules A and C each searched 42 candidates
using confidence bounds on the same 40 recordings, so "the sole passing
candidate" may be an artefact of which recordings happened to be sampled.

Method. Resample the 40 recordings WITH REPLACEMENT, jointly across all 42
candidates (the same recording identifiers for every configuration, preserving
the paired design). Within each outer draw, recompute the paired confidence
bounds by an inner bootstrap and REAPPLY THE COMPLETE SELECTION RULE. Report
how often each candidate is selected, and how often the rule abstains.

This is the analysis your supervisor listed first. It requires no ASR decoding.

Reads:
  sa/sa_<cond>.csv        per-recording wer, wder, sa_wer
  clin/clin_<cond>.csv    per-recording negation_n, negation_err,
                          medication_n, medication_err

Rules reproduced from the manuscript:
  Rule A  argmin WER over candidates with U(negation) <= 0
          and U(WDER) <= 4.0 and U(SA-WER) <= 3.0
  Rule C  argmin WER over candidates with L(k) <= 0 for
          k in {negation, WDER, SA-WER}

Usage:
  python3 selection_validity_bootstrap.py \
      --root /path/to/derived-outcomes \
      --subset /path/to/development-split.json \
      --outer 2000 --inner 400 \
      --out selection_validity.csv
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

SEED = 20260902
REFERENCE = "baseline"
DELTA_WDER = 4.0
DELTA_SAWER = 3.0


def load_all(root, subset_ids):
    """Return dict cond -> DataFrame indexed by recording with the outcomes."""
    sa_files = sorted(glob.glob(os.path.join(root, "sa", "sa_*.csv")))
    if not sa_files:
        sys.exit(f"no sa/sa_*.csv under {root}")

    data = {}
    for p in sa_files:
        cond = os.path.basename(p)[3:-4]          # strip "sa_" and ".csv"
        sa = pd.read_csv(p)
        clin_p = os.path.join(root, "clin", f"clin_{cond}.csv")
        if not os.path.exists(clin_p):
            print(f"  skip {cond}: no clin_{cond}.csv")
            continue
        clin = pd.read_csv(clin_p)

        sa = sa.set_index("case")
        clin = clin.set_index("case")
        idx = sa.index.intersection(clin.index)
        if subset_ids:
            idx = idx.intersection(subset_ids)
        if len(idx) == 0:
            print(f"  skip {cond}: no overlapping recordings")
            continue

        df = pd.DataFrame(index=idx)
        # sa/ stores rates as fractions; the manuscript reports percentages
        df["wer"] = sa.loc[idx, "wer"].astype(float) * 100.0
        df["wder"] = sa.loc[idx, "wder"].astype(float) * 100.0
        df["sa_wer"] = sa.loc[idx, "sa_wer"].astype(float) * 100.0
        # counts, not rates: resampling must aggregate numerator and denominator
        df["neg_err"] = clin.loc[idx, "negation_err"].astype(float)
        df["neg_n"] = clin.loc[idx, "negation_n"].astype(float)
        data[cond] = df
    return data


def common_recordings(data):
    idx = None
    for df in data.values():
        idx = df.index if idx is None else idx.intersection(df.index)
    return sorted(idx)


def paired_bounds(diff_matrix, draws):
    """
    diff_matrix: (n_cond, n_rec) paired differences vs reference.
    draws: (B, n_rec) index array.
    Returns (lo, hi) each (n_cond,) percentile bounds of the bootstrap mean.
    """
    # (B, n_cond) means
    means = diff_matrix[:, draws].mean(axis=2).T
    lo = np.percentile(means, 2.5, axis=0)
    hi = np.percentile(means, 97.5, axis=0)
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--subset", default=None)
    ap.add_argument("--outer", type=int, default=2000)
    ap.add_argument("--inner", type=int, default=400)
    ap.add_argument("--out", default="selection_validity.csv")
    args = ap.parse_args()

    subset_ids = None
    if args.subset:
        sub = json.load(open(args.subset))
        subset_ids = set(sub if isinstance(sub, list)
                         else (sub.get("dev") or sub.get("ids") or []))

    print("[load] reading per-recording outcomes")
    data = load_all(args.root, subset_ids)
    if REFERENCE not in data:
        sys.exit(f"reference condition '{REFERENCE}' not found")

    recs = common_recordings(data)
    conds = sorted(data.keys())
    print(f"[load] {len(conds)} configurations, {len(recs)} common recordings")

    n_rec = len(recs)
    n_cond = len(conds)

    W = np.zeros((n_cond, n_rec))
    D_neg = np.zeros((n_cond, n_rec))
    D_wder = np.zeros((n_cond, n_rec))
    D_sawer = np.zeros((n_cond, n_rec))

    ref = data[REFERENCE].loc[recs]
    ref_neg_rate = np.where(ref["neg_n"].to_numpy() > 0,
                            100.0 * ref["neg_err"].to_numpy()
                            / np.maximum(ref["neg_n"].to_numpy(), 1), np.nan)

    for i, c in enumerate(conds):
        d = data[c].loc[recs]
        W[i] = d["wer"].to_numpy()
        cur = np.where(d["neg_n"].to_numpy() > 0,
                       100.0 * d["neg_err"].to_numpy()
                       / np.maximum(d["neg_n"].to_numpy(), 1), np.nan)
        D_neg[i] = np.nan_to_num(cur - ref_neg_rate, nan=0.0)
        D_wder[i] = d["wder"].to_numpy() - ref["wder"].to_numpy()
        D_sawer[i] = d["sa_wer"].to_numpy() - ref["sa_wer"].to_numpy()

    rng = np.random.default_rng(SEED)

    # ---- point-estimate selection on the observed sample, for reference
    inner0 = rng.integers(0, n_rec, size=(args.inner, n_rec))
    lo_n, hi_n = paired_bounds(D_neg, inner0)
    lo_w, hi_w = paired_bounds(D_wder, inner0)
    lo_s, hi_s = paired_bounds(D_sawer, inner0)
    wer_mean = W.mean(axis=1)

    okA = (hi_n <= 0) & (hi_w <= DELTA_WDER) & (hi_s <= DELTA_SAWER)
    okC = (lo_n <= 0) & (lo_w <= 0) & (lo_s <= 0)
    pointA = conds[int(np.argmin(np.where(okA, wer_mean, np.inf)))] if okA.any() else None
    pointC = conds[int(np.argmin(np.where(okC, wer_mean, np.inf)))] if okC.any() else None
    print(f"[point] Rule A -> {pointA}   ({okA.sum()} candidates passed)")
    print(f"[point] Rule C -> {pointC}   ({okC.sum()} candidates passed)")

    # ---- outer bootstrap: reapply the whole rule in each resample
    winsA, winsC = Counter(), Counter()
    passA_count, passC_count = [], []

    print(f"[boot] {args.outer} outer x {args.inner} inner draws")
    for b in range(args.outer):
        outer = rng.integers(0, n_rec, size=n_rec)
        Wb = W[:, outer]
        Nb, WDb, SAb = D_neg[:, outer], D_wder[:, outer], D_sawer[:, outer]
        inner = rng.integers(0, n_rec, size=(args.inner, n_rec))

        lo_n, hi_n = paired_bounds(Nb, inner)
        lo_w, hi_w = paired_bounds(WDb, inner)
        lo_s, hi_s = paired_bounds(SAb, inner)
        wm = Wb.mean(axis=1)

        a = (hi_n <= 0) & (hi_w <= DELTA_WDER) & (hi_s <= DELTA_SAWER)
        c = (lo_n <= 0) & (lo_w <= 0) & (lo_s <= 0)
        passA_count.append(int(a.sum()))
        passC_count.append(int(c.sum()))

        winsA[conds[int(np.argmin(np.where(a, wm, np.inf)))] if a.any()
              else "ABSTAIN"] += 1
        winsC[conds[int(np.argmin(np.where(c, wm, np.inf)))] if c.any()
              else "ABSTAIN"] += 1

        if (b + 1) % 250 == 0:
            print(f"       {b+1}/{args.outer}")

    rows = []
    for rule, wins, point in (("A", winsA, pointA), ("C", winsC, pointC)):
        for name, n in wins.most_common():
            rows.append(dict(rule=rule, selection=name, times=n,
                             frequency_pct=round(100.0 * n / args.outer, 2),
                             is_point_selection=(name == point)))
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

    pd.set_option("display.width", 160)
    print("\n=== SELECTION FREQUENCY ACROSS RESAMPLES ===")
    for rule in ("A", "C"):
        sub = out[out.rule == rule].head(8)
        print(f"\nRule {rule}   (point selection: "
              f"{pointA if rule == 'A' else pointC})")
        print(sub[["selection", "times", "frequency_pct",
                   "is_point_selection"]].to_string(index=False))
    print(f"\nmean candidates passing Rule A: {np.mean(passA_count):.1f}")
    print(f"mean candidates passing Rule C: {np.mean(passC_count):.1f}")
    print(f"\n[write] {args.out}")
    print("\nReport the frequency, not just the point winner. A candidate "
          "selected in a minority of resamples is not a stable selection.")


if __name__ == "__main__":
    main()
