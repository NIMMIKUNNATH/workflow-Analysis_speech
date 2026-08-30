#!/usr/bin/env python3
"""
final_objectives.py - evaluate every candidate selection objective on the real data.

WHY
    Six selection rules have been proposed. Rather than argue about them, this
    computes what each one actually selects from the scored conditions, side by
    side, so the choice is made against evidence.

THE SIX RULES

    A  Pareto front over (WER, negation)            no thresholds, returns a SET
    B  demonstrated improvement on negation          U95(neg) <= 0, then min WER
    C  non-degradation screen                        L95 <= 0 on neg/wder/sa, then min WER
    D  technical-first weighted sum                  normalised, weights on WER/WDER/SA
    E  safety-first weighted sum                     normalised, weight on negation
    F  Chebyshev min-max                             minimise the worst normalised metric

    A-C use paired confidence intervals and respect uncertainty.
    D-F rank point estimates only.

TWO WARNINGS ABOUT D-F, PRINTED IN THE OUTPUT TOO

    Min-max normalisation is computed across the candidate set, so a metric's
    scale is set by whichever condition happens to be worst. Removing one
    unrelated condition changes every score. The script reports how much by
    recomputing with the negation outlier dropped.

    D-F ignore confidence intervals entirely. A configuration can win on a
    difference indistinguishable from zero. The script flags when the winner's
    advantage is not statistically significant.

USAGE
    python final_objectives.py --results RESULTS --clinrate clinrate --sa sa
    python final_objectives.py ... --wer-weight 0.5 --exclude-outlier
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID = ["case", "recording", "file", "name", "id", "stem"]
METRICS = ["wer", "wder", "sa_wer", "negation", "medication"]
LOWER_BETTER = True  # every metric here is an error rate


def idcol(df, label):
    for c in ID:
        if c in df.columns:
            return c
    sys.exit(f"{label}: no identifier column. Saw {list(df.columns)}")


def load(results, clinrate, sa):
    conds = {}
    for p in sorted(Path(results).glob("*.csv")):
        name = p.stem
        d = pd.read_csv(p)
        d = d.rename(columns={idcol(d, name): "case"})
        conds[name] = d[["case"] + [c for c in ("wer", "wder") if c in d.columns]]
    for name in list(conds):
        for folder, prefix, cols in ((clinrate, "", ["negation", "medication"]),
                                     (sa, "sa_", ["sa_wer"])):
            p = Path(folder) / f"{prefix}{name}.csv"
            if not p.exists():
                continue
            e = pd.read_csv(p)
            e = e.rename(columns={idcol(e, name): "case"})
            take = [c for c in cols if c in e.columns]
            if take:
                conds[name] = conds[name].merge(e[["case"] + take], on="case", how="left")
    return conds


def pct(s):
    v = pd.Series(s).dropna().astype(float)
    if not len(v):
        return np.nan
    return v.mean() * (100.0 if v.max() <= 1.5 else 1.0)


def paired_ci(cand, base, metric, n_boot, seed):
    """Paired difference candidate - baseline, with a bootstrap 95% interval."""
    if metric not in cand.columns or metric not in base.columns:
        return None
    m = base[["case", metric]].merge(cand[["case", metric]], on="case",
                                     suffixes=("_b", "_c")).dropna()
    if len(m) < 3:
        return None
    a = m[f"{metric}_b"].astype(float)
    c = m[f"{metric}_c"].astype(float)
    scale = 100.0 if max(a.max(), c.max()) <= 1.5 else 1.0
    d = (c - a).to_numpy() * scale
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return dict(diff=d.mean(),
                lo=float(np.percentile(means, 2.5)),
                hi=float(np.percentile(means, 97.5)),
                n=len(d))


def minmax(values):
    """Normalise to [0,1] across the candidate set. 0 = best, 1 = worst."""
    v = np.array(values, dtype=float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if hi - lo < 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--clinrate", required=True)
    ap.add_argument("--sa", required=True)
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--delta-wder", type=float, default=4.0)
    ap.add_argument("--delta-sa", type=float, default=3.0)
    ap.add_argument("--wer-weight", type=float, default=0.4)
    ap.add_argument("--exclude-outlier", action="store_true",
                    help="drop the worst-negation condition and recompute D-F")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    conds = load(args.results, args.clinrate, args.sa)
    if args.baseline not in conds:
        sys.exit(f"baseline '{args.baseline}' not found in {args.results}")
    base = conds[args.baseline]

    # ---- means and paired intervals -------------------------------------
    rows = {}
    for name, d in conds.items():
        r = {m: pct(d[m]) if m in d.columns else np.nan for m in METRICS}
        for m in ("negation", "wder", "sa_wer", "medication"):
            ci = paired_ci(d, base, m, args.boot, args.seed) if name != args.baseline else None
            r[f"{m}_lo"] = ci["lo"] if ci else 0.0
            r[f"{m}_hi"] = ci["hi"] if ci else 0.0
            r[f"{m}_d"] = ci["diff"] if ci else 0.0
        rows[name] = r
    E = pd.DataFrame(rows).T
    E = E.dropna(subset=["wer", "wder", "sa_wer", "negation"])

    print(f"\n{'='*92}")
    print(f"SELECTION OBJECTIVES  ·  {len(E)} configurations  ·  baseline "
          f"'{args.baseline}'  ·  {args.boot:,} bootstrap resamples")
    print(f"{'='*92}")

    # ---- A: Pareto front -------------------------------------------------
    pts = {i: (E.loc[i, "wer"], E.loc[i, "negation"]) for i in E.index}
    front = [i for i, (w, n) in pts.items()
             if not any(w2 <= w and n2 <= n and (w2 < w or n2 < n)
                        for j, (w2, n2) in pts.items() if j != i)]
    front.sort(key=lambda i: pts[i][0])
    print("\nRULE A  ·  Pareto front over (WER, negation) — no thresholds, returns a set")
    for i in front:
        print(f"    {i:22s} WER {pts[i][0]:6.2f}   negation {pts[i][1]:5.2f}")
    if len(front) > 1:
        print("    exchange rate along the front:")
        for a, b in zip(front, front[1:]):
            dw, dn = pts[b][0] - pts[a][0], pts[a][1] - pts[b][1]
            print(f"      {a} -> {b}:  +{dw:.2f} WER buys {dn:.2f} negation")

    # ---- B: demonstrated improvement on negation -------------------------
    okB = [i for i in E.index if i != args.baseline
           and E.loc[i, "negation_hi"] <= 0
           and E.loc[i, "wder_hi"] <= args.delta_wder
           and E.loc[i, "sa_wer_hi"] <= args.delta_sa]
    print(f"\nRULE B  ·  U95(negation) <= 0, non-inferior on WDER/SA-WER, then min WER")
    if not okB:
        print("    no configuration qualifies -> abstention")
    else:
        for i in sorted(okB, key=lambda i: E.loc[i, "wer"]):
            print(f"    {i:22s} WER {E.loc[i,'wer']:6.2f}   U95(neg) "
                  f"{E.loc[i,'negation_hi']:+.2f}")
        print(f"    -> selects {min(okB, key=lambda i: E.loc[i,'wer'])}")

    # ---- C: non-degradation screen ---------------------------------------
    okC = [i for i in E.index
           if E.loc[i, "negation_lo"] <= 0 and E.loc[i, "wder_lo"] <= 0
           and E.loc[i, "sa_wer_lo"] <= 0]
    print(f"\nRULE C  ·  non-degradation screen (L95 <= 0 on negation, WDER, SA-WER)")
    print(f"    {len(okC)} of {len(E)} configurations pass")
    excl = sorted(set(E.index) - set(okC), key=lambda i: E.loc[i, "wer"])
    if excl:
        print(f"    excluded ({len(excl)}): {', '.join(excl[:10])}"
              + (" ..." if len(excl) > 10 else ""))
    if okC:
        selC = min(okC, key=lambda i: E.loc[i, "wer"])
        print(f"    -> selects {selC}  (WER {E.loc[selC,'wer']:.2f})")

    # ---- D/E/F: normalised scalarisations --------------------------------
    def scalarised(frame, label):
        n = pd.DataFrame({m: minmax(frame[m].values) for m in
                          ["wer", "wder", "sa_wer", "negation", "medication"]},
                         index=frame.index)
        w = args.wer_weight
        rest = (1.0 - w) / 2.0
        D = w * n.wer + rest * n.wder + rest * n.sa_wer + 0.1 * n.negation
        Es_ = n.negation + 0.3 * n.medication.fillna(0)
        Fs = n[["wer", "wder", "sa_wer", "negation"]].max(axis=1)
        out = {}
        for nm, S in (("D technical-first", D), ("E safety-first", Es_),
                      ("F Chebyshev balanced", Fs)):
            best = S.idxmin()
            out[nm] = best
            print(f"\nRULE {nm}{label}")
            for i in S.nsmallest(4).index:
                print(f"    {i:22s} score {S[i]:.3f}   WER {frame.loc[i,'wer']:6.2f}"
                      f"   negation {frame.loc[i,'negation']:5.2f}")
            sig = (frame.loc[best, "negation_hi"] <= 0 or
                   frame.loc[best, "negation_lo"] >= 0)
            if best != args.baseline and not sig:
                print(f"    WARNING: {best}'s negation difference is "
                      f"{frame.loc[best,'negation_d']:+.2f} "
                      f"[{frame.loc[best,'negation_lo']:+.2f}, "
                      f"{frame.loc[best,'negation_hi']:+.2f}] — not distinguishable")
                print(f"    from zero. This rule ranks point estimates and ignores that.")
        return out

    print(f"\n{'-'*92}")
    print("NORMALISED SCALARISATIONS  ·  min-max over the candidate set, point estimates only")
    print(f"{'-'*92}")
    sel_full = scalarised(E, "")

    # sensitivity: drop the worst-negation condition and see what moves
    worst = E["negation"].idxmax()
    E2 = E.drop(index=worst)
    print(f"\n{'-'*92}")
    print(f"SCALE SENSITIVITY  ·  recomputed without '{worst}' "
          f"(negation {E.loc[worst,'negation']:.2f}, the range-setting condition)")
    print(f"{'-'*92}")
    sel_drop = scalarised(E2, "  [outlier dropped]")

    print(f"\n{'-'*92}")
    changed = [k for k in sel_full if sel_full[k] != sel_drop.get(k)]
    if changed:
        print("SELECTION CHANGED when one unrelated condition was removed:")
        for k in changed:
            print(f"    {k}: {sel_full[k]} -> {sel_drop[k]}")
        print("\n  Min-max normalisation sets each metric's scale from whichever")
        print("  condition happens to be worst. A rule whose answer depends on")
        print("  which candidates were run is difficult to defend.")
    else:
        print("Scalarised selections were unchanged when the range-setting")
        print("condition was removed.")

    print(f"\n{'='*92}")
    print("SUMMARY")
    print(f"{'='*92}")
    print(f"  A  Pareto front        {len(front)} configurations: {', '.join(front)}")
    print(f"  B  demonstrated better {min(okB, key=lambda i: E.loc[i,'wer']) if okB else 'ABSTAIN'}")
    print(f"  C  non-degradation     {min(okC, key=lambda i: E.loc[i,'wer']) if okC else 'ABSTAIN'}")
    for k, v in sel_full.items():
        print(f"  {k:22s} {v}")
    print("\n  A-C respect the paired confidence intervals. D-F rank point")
    print("  estimates and are reported as a sensitivity analysis, not as")
    print("  primary selection rules.\n")

    if args.out:
        E.to_csv(args.out)
        print(f"Saved -> {args.out}\n")


if __name__ == "__main__":
    main()
