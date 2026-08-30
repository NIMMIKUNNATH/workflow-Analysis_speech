#!/usr/bin/env python3
"""
final_parameter_table.py - build the parameter-study table programmatically.

WHY THIS EXISTS
    The parameter table was hand-assembled once and the counts were wrong: the
    text said "seven significant, twenty-two null" while the table listed eight
    rows and the null list contained twenty-one entries. 8 + 21 = 29, so both
    numbers in the sentence were incorrect. Hand-counting a family that has
    grown from 29 to 35 comparisons will produce the same class of error again.

    This script reads the per-condition result CSVs and regenerates the whole
    table - every comparison, paired against baseline, with Holm correction
    applied across the family. Nothing is typed by hand.

FAMILIES
    Comparisons are NOT all one family. Holm is applied within a family, and
    mixing them either inflates or deflates every adjusted p-value:

      primary   one-factor conditions vs baseline        (--family primary)
      transfer  large-v2 + one parameter change          (--family transfer)
                these change TWO factors, so they cannot sit in the one-factor
                table; they are their own family
      focused   best_combo vs large-v2                   run separately with
                                                         final_paired_compare.py

    The excluded quality-control runs (temperature 0, disabled thresholds,
    initial prompt) belong to NEITHER family. They are pipeline-failure checks,
    not parameter comparisons, and including them would corrupt the correction.

USAGE
    python final_parameter_table.py --results ${ASR_STUDY_ROOT}/results
    python final_parameter_table.py --results results --family transfer
    python final_parameter_table.py --results results --latex --out table_params.tex
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as st
except ImportError:
    st = None

ID_CANDIDATES = ["case", "recording", "file", "name", "id", "stem"]

# Conditions that change ONE factor relative to baseline.
PRIMARY = [
    "model_small", "model_medium", "model_largev2", "model_turbo",
    "compute_fp16",
    "beam1", "beam3", "beam8",
    "patience_150", "patience_200",
    "length_pen_080", "length_pen_120",
    "rep_penalty_110", "no_repeat_3",
    "nospeech_040", "nospeech_075", "nospeech_090",
    "logprob_-05", "logprob_-15",
    "compression_20", "compression_30",
    "cond_prev_off",
    "vad_off", "vad_thresh_030", "vad_thresh_070",
    "vad_silence_500", "vad_silence_1000",
    "vad_pad_200", "vad_pad_600",
    "merge_gap_030", "merge_gap_150", "merge_gap_200",
    "min_interval_00", "min_interval_25", "min_interval_40",
]

# Two-factor: a model change AND a parameter change. Separate family.
TRANSFER = ["v2_beam1", "v2_beam3", "v2_vad_500", "v2_vad_off"]

# Neither family - reported as measurements, never inferentially.
EXCLUDED = ["best_combo"]

METRICS = ["wer", "wder"]


def find_id(df, label):
    for c in ID_CANDIDATES:
        if c in df.columns:
            return c
    sys.exit(f"{label}: no identifier column. Saw {list(df.columns)}")


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def boot_ci(x, n_boot, seed, alpha=0.05):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="directory of per-condition CSVs")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--family", choices=["primary", "transfer"], default="primary")
    ap.add_argument("--metric", default="wer", choices=METRICS)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rdir = Path(args.results)
    base_path = rdir / f"{args.baseline}.csv"
    if not base_path.exists():
        sys.exit(f"baseline not found: {base_path}")
    base = pd.read_csv(base_path)
    base = base.rename(columns={find_id(base, "baseline"): "case"})

    wanted = PRIMARY if args.family == "primary" else TRANSFER

    rows, pvals, missing, incomplete = [], [], [], []
    for cond in wanted:
        p = rdir / f"{cond}.csv"
        if not p.exists():
            missing.append(cond)
            continue
        d = pd.read_csv(p)
        d = d.rename(columns={find_id(d, cond): "case"})
        if args.metric not in d.columns:
            missing.append(f"{cond} (no '{args.metric}' column)")
            continue

        m = base[["case", args.metric]].merge(
            d[["case", args.metric]], on="case", suffixes=("_b", "_c"))
        if len(m) < len(base):
            incomplete.append((cond, len(m), len(base)))
        if len(m) < 3:
            continue

        a = m[f"{args.metric}_b"].astype(float)
        c = m[f"{args.metric}_c"].astype(float)
        scale = 100.0 if max(a.max(), c.max()) <= 1.5 else 1.0
        diff = (c - a).to_numpy() * scale

        lo, hi = boot_ci(diff, args.boot, args.seed)
        if st is not None:
            _, pv = st.ttest_rel(c, a)
        else:
            pv = float("nan")

        rows.append({
            "condition": cond, "n": len(m),
            "mean": c.mean() * scale,
            "diff": diff.mean(), "lo": lo, "hi": hi,
            "p": pv,
            "dz": diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) else np.nan,
            "better": int((diff < 0).sum()),
        })
        pvals.append(pv if pv == pv else 1.0)

    if not rows:
        sys.exit("no conditions scored")

    for r, pa in zip(rows, holm(pvals)):
        r["p_holm"] = pa
        crosses = r["lo"] <= 0 <= r["hi"]
        if crosses:
            r["verdict"] = "no difference"
        elif pa < 0.05:
            r["verdict"] = "better" if r["diff"] < 0 else "worse"
        else:
            r["verdict"] = "exploratory"

    rows.sort(key=lambda r: r["diff"])
    base_mean = base[args.metric].astype(float)
    bscale = 100.0 if base_mean.max() <= 1.5 else 1.0

    print(f"\n{'='*94}")
    print(f"{args.family.upper()} FAMILY  ·  {args.metric.upper()}  ·  "
          f"{len(rows)} comparisons vs {args.baseline} "
          f"({base_mean.mean()*bscale:.2f}%)")
    print(f"{'='*94}")

    if missing:
        print(f"\nNOT YET AVAILABLE ({len(missing)}): {', '.join(missing)}")
        print("The family is INCOMPLETE. Holm below corrects across only the")
        print("comparisons present, which is NOT the prespecified correction.")
        print("Rerun once every condition has finished.\n")
    if incomplete:
        print("PARTIAL CONDITIONS - fewer recordings than baseline:")
        for c, got, exp in incomplete:
            print(f"  {c}: {got}/{exp}")
        print()

    hdr = (f"{'condition':<20}{'mean':>8}{'diff':>8}{'95% CI':>20}"
           f"{'p':>11}{'p Holm':>10}{'dz':>7}{'n<0':>6}  verdict")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        ci = f"[{r['lo']:+.2f}, {r['hi']:+.2f}]"
        print(f"{r['condition']:<20}{r['mean']:>8.2f}{r['diff']:>+8.2f}{ci:>20}"
              f"{r['p']:>11.4g}{r['p_holm']:>10.4g}{r['dz']:>+7.2f}"
              f"{r['better']:>4d}/{r['n']}  {r['verdict']}")

    n_sig = sum(1 for r in rows if r["verdict"] in ("better", "worse"))
    n_exp = sum(1 for r in rows if r["verdict"] == "exploratory")
    n_null = sum(1 for r in rows if r["verdict"] == "no difference")
    print(f"\n{n_sig} significant after Holm · {n_exp} exploratory "
          f"(nominal only) · {n_null} no difference · {len(rows)} total")
    print("\nQuote these three counts directly. Do not recount by hand -")
    print("that is how the earlier 'seven significant, twenty-two null'")
    print("error was introduced against a table of eight and a list of 21.")

    if args.latex:
        lines = [
            r"\begin{table}[t]", r"\centering",
            r"\caption{" + f"{args.family.capitalize()} family: paired "
            f"{args.metric.upper()} differences against baseline "
            f"({base_mean.mean()*bscale:.2f}\\%), $n$ recordings paired. "
            r"Holm correction applied across the family.}",
            r"\begin{tabular}{lrrrrrl}", r"\hline",
            r"Condition & Mean & Diff. & 95\% CI & $p$ & $p_{\mathrm{Holm}}$ & Verdict \\",
            r"\hline",
        ]
        for r in rows:
            cond = r["condition"].replace("_", r"\_")
            lines.append(
                f"{cond} & {r['mean']:.2f} & {r['diff']:+.2f} & "
                f"[{r['lo']:+.2f}, {r['hi']:+.2f}] & {r['p']:.4g} & "
                f"{r['p_holm']:.4g} & {r['verdict']} \\\\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        tex = "\n".join(lines)
        if args.out:
            Path(args.out).write_text(tex)
            print(f"\nSaved -> {args.out}")
        else:
            print("\n" + tex)
    print()


if __name__ == "__main__":
    main()
