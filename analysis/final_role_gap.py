#!/usr/bin/env python3
"""
final_role_gap.py - is the doctor/patient transcription gap real?

WHY THIS IS NEEDED
    final_score_fareez_speaker.py reports a corpus-level (micro) gap:

        doctor   20790 / 201709 = 10.31%
        patient  12693 / 156118 =  8.13%
        difference -2.18 points

    That is a single ratio over 357,827 words. It carries no uncertainty, and
    words within a recording are correlated, so it cannot be tested as though
    the words were independent draws. Every other comparison in this study is
    reported paired on recordings with a confidence interval; this one must be
    too, or a reviewer will ask why the standard slipped for one table.

WHAT IT DOES
    The two roles occur in the SAME recording, so the comparison is naturally
    paired: for each recording, doctor rate minus patient rate. Between-recording
    variance - which is large, SD ~3.96 points on WER - cancels out, exactly as
    in the parameter study.

        1. paired t-test on the per-recording differences
        2. Wilcoxon signed-rank, in case the differences are not normal
        3. bootstrap 95% CI, resampling RECORDINGS (10,000 draws)
        4. how many recordings run each way, and how many are near zero
        5. macro mean alongside the micro rate, since they answer different
           questions and must not be mixed between tables

USAGE
    python final_role_gap.py --csv fareez_272_roles.csv
    python final_role_gap.py --csv fareez_272_roles.csv --out role_gap.csv
"""
import argparse
import sys

import numpy as np
import pandas as pd

try:
    from scipy import stats as st
except ImportError:
    st = None


def bootstrap_ci(x, n_boot=10000, seed=20260819, alpha=0.05):
    """Percentile bootstrap CI for the mean, resampling recordings."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="output of final_score_fareez_speaker.py")
    ap.add_argument("--out", default=None,
                    help="optional per-recording difference table")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260819)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    need = {"doctor_n", "doctor_err", "patient_n", "patient_err"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"{args.csv} is missing {sorted(missing)}. This CSV was "
                 f"probably written by the old scorer - re-run with "
                 f"final_score_fareez_speaker.py.")

    # Drop recordings where either role has no words. A recording with a silent
    # patient carries no information about the gap and would divide by zero.
    before = len(df)
    df = df[(df.doctor_n > 0) & (df.patient_n > 0)].copy()
    dropped = before - len(df)

    df["doctor_rate"] = df.doctor_err / df.doctor_n * 100
    df["patient_rate"] = df.patient_err / df.patient_n * 100
    df["diff"] = df.doctor_rate - df.patient_rate      # + = doctor harder

    d = df["diff"].to_numpy()
    n = len(d)

    # ---- micro, for comparison with the scorer's own printout ----------
    micro_doc = df.doctor_err.sum() / df.doctor_n.sum() * 100
    micro_pat = df.patient_err.sum() / df.patient_n.sum() * 100

    print(f"\n{'='*68}")
    print(f"DOCTOR vs PATIENT TRANSCRIPTION GAP - {n} recordings")
    print(f"{'='*68}")
    if dropped:
        print(f"{dropped} recording(s) dropped: one role had no reference words.\n")

    print("corpus-level (micro) - weights by words, matches the scorer")
    print(f"  doctor   {micro_doc:6.2f}%")
    print(f"  patient  {micro_pat:6.2f}%")
    print(f"  gap      {micro_doc - micro_pat:+6.2f} points")

    print("\nper-recording (macro) - weights every consultation equally")
    print(f"  doctor   {df.doctor_rate.mean():6.2f}%  (SD {df.doctor_rate.std():.2f})")
    print(f"  patient  {df.patient_rate.mean():6.2f}%  (SD {df.patient_rate.std():.2f})")
    print(f"  gap      {d.mean():+6.2f} points  (SD {d.std(ddof=1):.2f})")

    print("\n  The two answer different questions and must not be mixed between")
    print("  tables. Micro asks how much of the transcribed content is wrong;")
    print("  macro asks how the typical consultation behaves. State which.")

    # ---- is the gap real? ----------------------------------------------
    lo, hi = bootstrap_ci(d, n_boot=args.boot, seed=args.seed)
    print(f"\nbootstrap 95% CI on the paired difference "
          f"({args.boot:,} resamples of recordings)")
    print(f"  {d.mean():+.2f}  [{lo:+.2f}, {hi:+.2f}]")
    crosses = lo <= 0 <= hi
    print(f"  interval {'INCLUDES' if crosses else 'excludes'} zero")

    if st is not None:
        t, p_t = st.ttest_rel(df.doctor_rate, df.patient_rate)
        try:
            w, p_w = st.wilcoxon(d)
        except ValueError:
            w, p_w = float("nan"), float("nan")
        print(f"\npaired t-test        t = {t:7.3f}   p = {p_t:.4g}")
        print(f"Wilcoxon signed-rank W = {w:9.1f}   p = {p_w:.4g}")
        # Cohen's dz for paired designs
        print(f"effect size (dz)     {d.mean() / d.std(ddof=1):+.3f}")
    else:
        print("\nscipy not installed - CI only. pip install scipy for the tests.")

    # ---- consistency, not just the average ------------------------------
    harder_doc = int((d > 0).sum())
    harder_pat = int((d < 0).sum())
    near_zero = int((np.abs(d) < 1.0).sum())
    print(f"\ndirection, per recording")
    print(f"  doctor harder    {harder_doc:4d}  ({harder_doc/n*100:.1f}%)")
    print(f"  patient harder   {harder_pat:4d}  ({harder_pat/n*100:.1f}%)")
    print(f"  within 1 point   {near_zero:4d}  ({near_zero/n*100:.1f}%)")
    print("\n  A mean gap carried by a consistent majority is a property of the")
    print("  task. One carried by a few extreme recordings is not - check the")
    print("  split above before writing 'the doctor is transcribed less well'.")

    ext = df.reindex(df["diff"].abs().sort_values(ascending=False).index).head(5)
    print(f"\nfive largest gaps in either direction")
    for _, r in ext.iterrows():
        who = "doctor" if r["diff"] > 0 else "patient"
        print(f"  {r['case']:<10} {r['diff']:+7.2f}  ({who} harder)   "
              f"doc {r['doctor_rate']:5.2f}% n={int(r['doctor_n']):5d}   "
              f"pat {r['patient_rate']:5.2f}% n={int(r['patient_n']):5d}")

    # ---- share of speech, in case the gap is really about talk time -----
    df["doctor_share"] = df.doctor_n / (df.doctor_n + df.patient_n) * 100
    print(f"\ndoctor's share of reference words: "
          f"{df.doctor_share.mean():.1f}% (SD {df.doctor_share.std():.1f})")
    if st is not None and n > 2:
        rho, p_rho = st.spearmanr(df.doctor_share, df["diff"])
        print(f"correlation with the gap: rho = {rho:+.3f}, p = {p_rho:.4g}")
        print("  If this is strong, the gap may be about how much each person")
        print("  speaks rather than how they speak, and should be reported as")
        print("  such. If it is weak, the asymmetry stands on its own.")

    if args.out:
        cols = ["case", "doctor_n", "doctor_err", "doctor_rate",
                "patient_n", "patient_err", "patient_rate", "diff",
                "doctor_share"]
        df[[c for c in cols if c in df.columns]].to_csv(args.out, index=False)
        print(f"\nSaved -> {args.out}")

    print("\nNOTE. These are substitution-plus-deletion rates. Insertions have no")
    print("reference speaker and cannot be attributed to a role, so they are")
    print("excluded here but INCLUDED in the headline WER. Do not present these")
    print("figures alongside 18.05% as though they were the same measurement.\n")


if __name__ == "__main__":
    main()
