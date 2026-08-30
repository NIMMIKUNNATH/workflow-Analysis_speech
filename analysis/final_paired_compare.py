#!/usr/bin/env python3
"""
final_paired_compare.py - paired comparison between any TWO conditions.

WHY THIS EXISTS
    study.py --analyse compares every condition against BASELINE. That answers
    "does this parameter do anything", which was the question the parameter study
    asked. It does not answer "is best_combo better than large-v2", because
    neither of those is the baseline.

    Comparing them through the baseline is not the same thing: two differences
    each with their own CI cannot be subtracted to get the difference between
    them. The recordings are the same in both arms, so the direct paired
    comparison is available and is much tighter.

WHY PAIRED
    WER varies by ~3.96 points (SD) BETWEEN recordings and only ~0.3 points
    between two runs of an identical configuration. Comparing group means throws
    away the pairing and buries a 1-point effect under 4 points of noise. Taking
    the per-recording difference cancels the between-recording variance entirely.

    This is the same reasoning that made the 40-recording paired design work
    where 2 unpaired recordings could not.

WHAT IT REPORTS, PER METRIC
    mean paired difference, bootstrap 95% CI (resampling RECORDINGS),
    paired t-test, Wilcoxon signed-rank, Cohen's dz, and the direction split.
    Holm correction is applied ACROSS METRICS, because testing WER, WDER,
    medication and negation on one comparison is four tests, not one.

INPUTS
    Two per-recording CSVs with a case/recording identifier column. Only
    recordings present in BOTH are used, and the script says how many were
    dropped - a silent inner join is how you end up comparing 31 recordings
    while reporting n=40.

USAGE
    python final_paired_compare.py --a results/model_largev2.csv \\
                                   --b results/best_combo.csv \\
                                   --name-a large-v2 --name-b best_combo

    python final_paired_compare.py --a A.csv --b B.csv --metrics wer wder
"""
import argparse
import sys

import numpy as np
import pandas as pd

try:
    from scipy import stats as st
except ImportError:
    st = None

# Columns that identify a recording, in order of preference.
ID_CANDIDATES = ["case", "recording", "file", "name", "id", "stem"]

# Metrics scored automatically if present. Lower is better for all of them.
DEFAULT_METRICS = ["wer", "wder", "sa_wer", "der", "medication", "negation",
                   "clinical", "missed", "confusion"]


def find_id_column(df, label):
    for c in ID_CANDIDATES:
        if c in df.columns:
            return c
    sys.exit(f"{label}: no recording identifier column found. Looked for "
             f"{ID_CANDIDATES}, saw {list(df.columns)}.")


def bootstrap_ci(x, n_boot, seed, alpha=0.05):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="first condition CSV (reference)")
    ap.add_argument("--b", required=True, help="second condition CSV")
    ap.add_argument("--name-a", default=None)
    ap.add_argument("--name-b", default=None)
    ap.add_argument("--metrics", nargs="*", default=None,
                    help="columns to compare; default: any of "
                         + ", ".join(DEFAULT_METRICS) + " that are present")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--out", default=None, help="per-recording differences")
    args = ap.parse_args()

    name_a = args.name_a or args.a
    name_b = args.name_b or args.b

    da, db = pd.read_csv(args.a), pd.read_csv(args.b)
    ida, idb = find_id_column(da, name_a), find_id_column(db, name_b)
    da = da.rename(columns={ida: "case"})
    db = db.rename(columns={idb: "case"})

    # Which metrics can actually be compared?
    if args.metrics:
        metrics = args.metrics
        for m in metrics:
            if m not in da.columns or m not in db.columns:
                sys.exit(f"'{m}' is not in both files. "
                         f"{name_a}: {list(da.columns)}. "
                         f"{name_b}: {list(db.columns)}.")
    else:
        metrics = [m for m in DEFAULT_METRICS
                   if m in da.columns and m in db.columns]
        if not metrics:
            sys.exit("No shared metric columns. Pass --metrics explicitly.")

    merged = da[["case"] + metrics].merge(
        db[["case"] + metrics], on="case", suffixes=("_a", "_b"))

    only_a = len(set(da.case) - set(db.case))
    only_b = len(set(db.case) - set(da.case))
    n = len(merged)
    if n < 3:
        sys.exit(f"Only {n} recordings in common - nothing to test.")

    print(f"\n{'='*72}")
    print(f"PAIRED COMPARISON   {name_b}  vs  {name_a}")
    print(f"{'='*72}")
    print(f"{n} recordings in both files")
    if only_a or only_b:
        print(f"  DROPPED: {only_a} only in {name_a}, {only_b} only in {name_b}.")
        print(f"  A paired test uses the intersection. If these numbers are not")
        print(f"  zero, say n={n} in the write-up, not 40.")
    print(f"\nnegative difference = {name_b} is BETTER (lower is better "
          f"for every metric here)\n")

    results, pvals = [], []
    for m in metrics:
        a = merged[f"{m}_a"].astype(float)
        b = merged[f"{m}_b"].astype(float)
        ok = a.notna() & b.notna()
        if ok.sum() < 3:
            print(f"{m:<12} too few paired values - skipped")
            continue
        d = (b[ok] - a[ok]).to_numpy()
        # Report in percentage points if the values look like proportions.
        scale = 100.0 if max(a[ok].max(), b[ok].max()) <= 1.5 else 1.0
        d = d * scale
        lo, hi = bootstrap_ci(d, args.boot, args.seed)

        if st is not None:
            t, p = st.ttest_rel(b[ok], a[ok])
            try:
                _, p_w = st.wilcoxon(d)
            except ValueError:
                p_w = float("nan")
        else:
            t, p, p_w = float("nan"), float("nan"), float("nan")

        results.append({
            "metric": m,
            "mean_a": a[ok].mean() * scale,
            "mean_b": b[ok].mean() * scale,
            "diff": d.mean(),
            "lo": lo, "hi": hi,
            "t": t, "p": p, "p_wilcoxon": p_w,
            "dz": d.mean() / d.std(ddof=1) if d.std(ddof=1) else float("nan"),
            "better": int((d < 0).sum()), "worse": int((d > 0).sum()),
            "n": int(ok.sum()),
        })
        pvals.append(p if p == p else 1.0)
        merged[f"diff_{m}"] = (merged[f"{m}_b"] - merged[f"{m}_a"]) * scale

    if not results:
        sys.exit("nothing comparable")

    adj = holm(pvals)
    for r, pa in zip(results, adj):
        r["p_holm"] = pa

    hdr = (f"{'metric':<12}{name_a[:10]:>11}{name_b[:10]:>11}"
           f"{'diff':>9}{'95% CI':>20}{'p':>11}{'p Holm':>10}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        ci = f"[{r['lo']:+.2f}, {r['hi']:+.2f}]"
        crosses = r["lo"] <= 0 <= r["hi"]
        if crosses:
            verdict = "no difference"
        elif r["p_holm"] >= 0.05:
            verdict = "exploratory"
        else:
            verdict = "BETTER" if r["diff"] < 0 else "WORSE"
        print(f"{r['metric']:<12}{r['mean_a']:>11.2f}{r['mean_b']:>11.2f}"
              f"{r['diff']:>+9.2f}{ci:>20}{r['p']:>11.4g}"
              f"{r['p_holm']:>10.4g}  {verdict}")

    print(f"\ndirection, per recording")
    for r in results:
        print(f"  {r['metric']:<12} {name_b} better in {r['better']:3d}/{r['n']}"
              f"  ({r['better']/r['n']*100:5.1f}%),  worse in {r['worse']:3d}"
              f"   dz {r['dz']:+.3f}")

    print(f"\nHolm correction applied across {len(results)} metrics on this")
    print("comparison. A metric whose CI crosses zero is reported as no")
    print("difference regardless of its p-value - with n=40 and a difference-SD")
    print("of 1-2 points, an interval spanning zero is the honest answer.")
    print("\nThis compares TWO CONDITIONS ONLY. It does not correct across the")
    print("29-condition parameter table; that correction is separate and still")
    print("outstanding.")

    if args.out:
        cols = ["case"] + [c for c in merged.columns if c.startswith("diff_")]
        merged[cols].to_csv(args.out, index=False)
        print(f"\nSaved -> {args.out}")
    print()


if __name__ == "__main__":
    main()
