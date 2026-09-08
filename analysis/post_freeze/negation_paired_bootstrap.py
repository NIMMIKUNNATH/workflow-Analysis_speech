#!/usr/bin/env python3
"""
negation_paired_bootstrap.py
----------------------------
Recording-paired bootstrap CIs for the negation decomposition, including the
new hallucinated class. Matches the manuscript's existing convention:
B = 10,000 draws, recording-clustered, seed 20260902, Holm across the family.

Input:  negation_insertion_per_recording.csv  (from negation_insertion_audit.py)
Output: negation_paired_bootstrap.csv

  python negation_paired_bootstrap.py \
      --per-recording negation_insertion_per_recording.csv \
      --reference baseline \
      --out negation_paired_bootstrap.csv
"""

import argparse
import numpy as np
import pandas as pd

SEED = 20260902
B = 10000


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def rate(err, ref):
    """Recording-level rate, undefined when the recording has no reference cue."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(ref > 0, 100.0 * err / np.maximum(ref, 1), np.nan)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-recording", required=True)
    ap.add_argument("--reference", default="baseline")
    ap.add_argument("--metrics", nargs="+",
                    default=["scored", "hallucinated", "total"])
    ap.add_argument("--out", default="negation_paired_bootstrap.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.per_recording)
    df["scored"] = df["deleted"] + df["substituted"]
    df["total"] = df["scored"] + df["hallucinated"]

    ref_name = args.reference
    conds = [c for c in df.condition.unique() if c != ref_name]
    if ref_name not in set(df.condition):
        raise SystemExit(f"reference condition '{ref_name}' not present")

    rng = np.random.default_rng(SEED)
    rows = []

    for metric in args.metrics:
        wide_e = df.pivot(index="recording", columns="condition", values=metric)
        wide_n = df.pivot(index="recording", columns="condition", values="ref_tokens")
        # keep recordings scoreable under every condition compared
        keep = wide_e[[ref_name] + conds].notna().all(axis=1)
        wide_e, wide_n = wide_e[keep], wide_n[keep]

        # hallucination has no reference-token denominator of its own; express it
        # per 100 reference negation tokens so it is on the same scale as the rest
        denom = wide_n[ref_name].to_numpy(float)
        valid = denom > 0
        idx = np.where(valid)[0]

        base = rate(wide_e[ref_name].to_numpy(float), denom)

        draws = rng.integers(0, len(idx), size=(B, len(idx)))
        pvals, cache = [], {}

        for cond in conds:
            cur = rate(wide_e[cond].to_numpy(float), denom)
            d = (cur - base)[idx]
            d = d[~np.isnan(d)]
            if len(d) == 0:
                continue
            obs = float(np.mean(d))
            samp = d[draws[:, :len(d)] % len(d)]
            bmeans = samp.mean(axis=1)
            lo, hi = np.percentile(bmeans, [2.5, 97.5])
            # two-sided bootstrap p by interval inversion
            p = 2 * min((bmeans <= 0).mean(), (bmeans >= 0).mean())
            p = max(p, 1.0 / B)
            sd = d.std(ddof=1)
            cache[cond] = dict(
                metric=metric, condition=cond, n_recordings=len(d),
                mean_diff_pp=round(obs, 4),
                ci_low=round(float(lo), 4), ci_high=round(float(hi), 4),
                dz=round(obs / sd, 4) if sd > 0 else np.nan,
                better_n=int((d < 0).sum()), worse_n=int((d > 0).sum()),
                raw_p=round(p, 6),
            )
            pvals.append(p)

        if cache:
            adj = holm(pvals)
            for (cond, rec), a in zip(cache.items(), adj):
                rec["holm_p"] = round(float(a), 6)
                rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    pd.set_option("display.width", 220)
    print(out.to_string(index=False))
    print(f"\n[write] {args.out}")
    print("\nRead the 'hallucinated' block first: if the reference condition "
          "hallucinates more cues than the alternatives, the frozen scorer "
          "understates reference-condition negation error.")


if __name__ == "__main__":
    main()
