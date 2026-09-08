#!/usr/bin/env python3
"""
equivalence_sensitivity.py
--------------------------
Quantifies how much the negation-equivalence table suppresses reported error.

clinical_errors.py treats a substitution as NOT an error when the reference and
hypothesis tokens fall in the same predefined equivalence class:

    if equivalent(rw, hw):
        normalised[c] += 1
        continue          # <- not counted in errors[c]

Those suppressed events are already recorded per recording in the *_norm
columns. The equivalence-disabled counterfactual is therefore computable from
the retained outputs with no re-transcription and no rescoring:

    reported rate    = err            / n
    equivalence-off  = (err + norm)   / n

If a content screen flags equivalence mappings as potentially conflating
auxiliary, temporal, modal or grammatical distinctions, this analysis states
the maximum consequence: how much error the table removes, and whether any
configuration comparison changes direction when it is disabled.

    python3 equivalence_sensitivity.py \
        --clin /path/to/clinical-results \
        --reference baseline \
        --out equivalence_sensitivity.csv

Reports pooled and recording-paired macro rates under both conventions, with
recording-clustered bootstrap intervals on the paired difference.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

SEED = 20260902
B = 10000


def load(clin_dir, cue="negation"):
    """cond -> DataFrame indexed by case with n, err, norm for the cue class."""
    files = sorted(glob.glob(os.path.join(clin_dir, "clin_*.csv")))
    if not files:
        sys.exit(f"no clin_*.csv under {clin_dir}")
    need = [f"{cue}_n", f"{cue}_err", f"{cue}_norm"]
    out = {}
    for p in files:
        cond = os.path.basename(p)[5:-4]
        df = pd.read_csv(p)
        missing = [c for c in need if c not in df.columns]
        if missing:
            print(f"  skip {cond}: missing {missing}")
            continue
        df = df.set_index("case")
        out[cond] = df[need].astype(float).rename(
            columns={need[0]: "n", need[1]: "err", need[2]: "norm"})
    return out


def rates(d):
    """Recording-level rates under both conventions; NaN where n == 0."""
    n = d["n"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rep = np.where(n > 0, 100.0 * d["err"].to_numpy() / np.maximum(n, 1),
                       np.nan)
        off = np.where(n > 0,
                       100.0 * (d["err"].to_numpy() + d["norm"].to_numpy())
                       / np.maximum(n, 1), np.nan)
    return rep, off


def paired_ci(diff, rng, b=B):
    d = diff[~np.isnan(diff)]
    if len(d) < 2:
        return np.nan, np.nan, np.nan
    draws = rng.integers(0, len(d), size=(b, len(d)))
    means = d[draws].mean(axis=1)
    return float(np.mean(d)), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clin", required=True)
    ap.add_argument("--cue", default="negation",
                    choices=["negation", "medication"])
    ap.add_argument("--reference", default="baseline")
    ap.add_argument("--out", default="equivalence_sensitivity.csv")
    args = ap.parse_args()

    data = load(args.clin, args.cue)
    if args.reference not in data:
        sys.exit(f"reference '{args.reference}' not found")
    conds = sorted(data)
    print(f"[load] {len(conds)} configurations, cue class '{args.cue}'")

    idx = None
    for df in data.values():
        idx = df.index if idx is None else idx.intersection(df.index)
    idx = sorted(idx)
    print(f"[load] {len(idx)} common recordings")

    rng = np.random.default_rng(SEED)
    rows = []
    for c in conds:
        d = data[c].loc[idx]
        n_tot = d["n"].sum()
        err_tot = d["err"].sum()
        norm_tot = d["norm"].sum()
        rep, off = rates(d)
        m, lo, hi = paired_ci(off - rep, rng)
        rows.append(dict(
            condition=c,
            ref_tokens=int(n_tot),
            errors_reported=int(err_tot),
            suppressed_by_equivalence=int(norm_tot),
            pooled_reported_pct=round(100 * err_tot / n_tot, 4) if n_tot else None,
            pooled_equiv_off_pct=round(100 * (err_tot + norm_tot) / n_tot, 4)
            if n_tot else None,
            pooled_increase_pp=round(100 * norm_tot / n_tot, 4) if n_tot else None,
            macro_reported_pct=round(float(np.nanmean(rep)), 4),
            macro_equiv_off_pct=round(float(np.nanmean(off)), 4),
            macro_increase_pp=round(m, 4),
            increase_ci_low=round(lo, 4), increase_ci_high=round(hi, 4),
        ))

    df = pd.DataFrame(rows).sort_values("pooled_reported_pct")
    df.to_csv(args.out, index=False)

    total_norm = df["suppressed_by_equivalence"].sum()
    total_err = df["errors_reported"].sum()

    pd.set_option("display.width", 220)
    print("\n=== EQUIVALENCE SUPPRESSION BY CONFIGURATION ===")
    print(df[["condition", "ref_tokens", "errors_reported",
              "suppressed_by_equivalence", "pooled_reported_pct",
              "pooled_equiv_off_pct", "pooled_increase_pp"]]
          .to_string(index=False))

    print(f"\nacross all configurations: {total_err:,} reported errors, "
          f"{total_norm:,} suppressed by the equivalence table "
          f"({100*total_norm/max(total_err+total_norm,1):.1f}% of all "
          f"matched cue events)")

    # does disabling equivalence change any comparison against the reference?
    print("\n=== EFFECT ON PAIRED COMPARISONS vs "
          f"{args.reference} ===")
    ref = data[args.reference].loc[idx]
    r_rep, r_off = rates(ref)
    flips, checked = [], 0
    for c in conds:
        if c == args.reference:
            continue
        d = data[c].loc[idx]
        c_rep, c_off = rates(d)
        m1, lo1, hi1 = paired_ci(c_rep - r_rep, rng)
        m2, lo2, hi2 = paired_ci(c_off - r_off, rng)
        checked += 1
        sig1 = (lo1 > 0) or (hi1 < 0)
        sig2 = (lo2 > 0) or (hi2 < 0)
        if np.sign(m1) != np.sign(m2) or sig1 != sig2:
            flips.append((c, m1, lo1, hi1, sig1, m2, lo2, hi2, sig2))

    if not flips:
        print(f"  {checked} comparisons: no change in sign or in whether the "
              f"interval excludes zero.")
        print("  Disabling the equivalence table does not alter the direction "
              "or detectability of any comparison against the reference.")
    else:
        print(f"  {len(flips)} of {checked} comparisons change:")
        for c, m1, lo1, hi1, s1, m2, lo2, hi2, s2 in flips:
            print(f"    {c:<22} reported {m1:+.3f} [{lo1:+.3f},{hi1:+.3f}]"
                  f"{' *' if s1 else '  '}   "
                  f"equiv-off {m2:+.3f} [{lo2:+.3f},{hi2:+.3f}]"
                  f"{' *' if s2 else ''}")

    print(f"\n[write] {args.out}")
    print("\nThis bounds the consequence of the equivalence table: it is the "
          "maximum error that could be restored if every mapping were judged "
          "lossy. It is not evidence that any particular mapping is wrong; "
          "that requires clinician adjudication of the mappings themselves.")


if __name__ == "__main__":
    main()
