#!/usr/bin/env python3
"""
final_split_holdout.py - separate the 40 development recordings from the 232
internal holdout, and report every metric on both splits.

WHY THIS IS NEEDED
    The 40-recording development subset was drawn from the same 272 Fareez
    recordings that the headline figures are computed over. Reporting
    "WER 18.05% on 272 recordings" therefore describes a pool that includes the
    data used to choose the configuration. It is a descriptive corpus figure,
    NOT independent validation, and a reviewer will say so.

    The fix costs nothing: the split already exists in subset.json. Report the
    40 and the 232 separately, label the 232 as the internal holdout, and the
    problem disappears.

WHAT IT DOES
    1. Reads the development subset from subset.json.
    2. Splits any per-recording results CSV into development and holdout.
    3. Reports mean, SD, median, p90, p95 and max for every metric, on each
       split and on the full corpus, so all three can be quoted correctly.
    4. Bootstraps a 95% CI for each holdout metric - the holdout figures are
       the ones that carry inferential weight, so they need intervals.
    5. Tests whether the two splits differ (Mann-Whitney U, UNPAIRED - these
       are different recordings, not the same ones under two conditions).
       If they do not differ, the subset was representative and that is worth
       stating. If they do, say so plainly rather than hoping nobody checks.
    6. Writes both splits to separate CSVs for use in tables.

USAGE
    python final_split_holdout.py --csv fareez_272_roles.csv \\
        --subset ${ASR_STUDY_ROOT}/subset.json

    python final_split_holdout.py --csv fareez_272_roles.csv \\
        --subset subset.json --out-prefix fareez --metrics wer wder sa_wer
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as st
except ImportError:
    st = None

ID_CANDIDATES = ["case", "recording", "file", "name", "id", "stem"]
DEFAULT_METRICS = ["wer", "wder", "sa_wer", "der", "doctor_rate", "patient_rate",
                   "medication", "negation", "clinical"]


def load_subset(path):
    """subset.json may be a bare list, or a dict with the list under a key."""
    raw = json.load(open(path))
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        for key in ("subset", "recordings", "cases", "files", "development"):
            if key in raw and isinstance(raw[key], list):
                items = raw[key]
                break
        else:
            # Fall back to the first list-valued entry.
            lists = [v for v in raw.values() if isinstance(v, list)]
            if not lists:
                sys.exit(f"{path}: no list of recordings found. Keys: {list(raw)}")
            items = lists[0]
    else:
        sys.exit(f"{path}: unexpected JSON type {type(raw)}")

    # Entries may be plain IDs, paths, or dicts.
    out = []
    for it in items:
        if isinstance(it, dict):
            for k in ID_CANDIDATES + ["path", "audio"]:
                if k in it:
                    it = it[k]
                    break
            else:
                sys.exit(f"{path}: cannot find an identifier in entry {it}")
        out.append(Path(str(it)).stem)
    return set(out)


def find_id_column(df):
    for c in ID_CANDIDATES:
        if c in df.columns:
            return c
    sys.exit(f"No recording identifier column. Saw: {list(df.columns)}")


def bootstrap_ci(x, n_boot, seed, alpha=0.05):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def describe(series, scale):
    v = series.dropna().to_numpy() * scale
    if not len(v):
        return None
    return {
        "n": len(v), "mean": v.mean(), "sd": v.std(ddof=1) if len(v) > 1 else 0.0,
        "median": np.median(v), "p90": np.percentile(v, 90),
        "p95": np.percentile(v, 95), "max": v.max(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="per-recording results CSV")
    ap.add_argument("--subset", required=True, help="subset.json")
    ap.add_argument("--metrics", nargs="*", default=None)
    ap.add_argument("--out-prefix", default=None,
                    help="write <prefix>_dev40.csv and <prefix>_holdout232.csv")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260819)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    idcol = find_id_column(df)
    df["_id"] = df[idcol].astype(str).map(lambda s: Path(s).stem)

    dev_ids = load_subset(args.subset)

    metrics = args.metrics or [m for m in DEFAULT_METRICS if m in df.columns]
    if not metrics:
        sys.exit("No metric columns found. Pass --metrics explicitly.")

    df["_split"] = np.where(df["_id"].isin(dev_ids), "dev", "holdout")
    dev = df[df._split == "dev"]
    hold = df[df._split == "holdout"]

    matched = len(dev)
    missing = len(dev_ids) - matched

    print(f"\n{'='*74}")
    print(f"DEVELOPMENT / HOLDOUT SPLIT - {Path(args.csv).name}")
    print(f"{'='*74}")
    print(f"subset.json lists {len(dev_ids)} recordings")
    print(f"  development : {matched:4d}")
    print(f"  holdout     : {len(hold):4d}")
    print(f"  total       : {len(df):4d}")
    if missing:
        print(f"\n  WARNING: {missing} subset recording(s) are NOT in this CSV.")
        print(f"  Identifiers may not match between subset.json and the results.")
        print(f"  Examples from subset: {sorted(dev_ids)[:3]}")
        print(f"  Examples from CSV   : {sorted(df['_id'])[:3]}")
        print(f"  Fix this before quoting anything below.")

    for m in metrics:
        col = df[m].astype(float)
        scale = 100.0 if col.max() <= 1.5 else 1.0
        unit = "%" if scale == 100.0 else ""

        d_dev = describe(dev[m].astype(float), scale)
        d_hold = describe(hold[m].astype(float), scale)
        d_all = describe(col, scale)
        if not (d_dev and d_hold):
            continue

        print(f"\n{'-'*74}")
        print(f"{m.upper()}")
        print(f"{'-'*74}")
        hdr = f"{'split':<22}{'n':>5}{'mean':>9}{'SD':>8}{'median':>9}{'p90':>8}{'p95':>8}{'max':>8}"
        print(hdr)
        for label, d in (("development (tuned on)", d_dev),
                         ("HOLDOUT (untouched)", d_hold),
                         ("full corpus (descriptive)", d_all)):
            print(f"{label:<22}{d['n']:>5}{d['mean']:>9.2f}{d['sd']:>8.2f}"
                  f"{d['median']:>9.2f}{d['p90']:>8.2f}{d['p95']:>8.2f}{d['max']:>8.2f}")

        hv = hold[m].dropna().to_numpy() * scale
        lo, hi = bootstrap_ci(hv, args.boot, args.seed)
        print(f"\n  holdout mean 95% CI: {hv.mean():.2f}{unit} [{lo:.2f}, {hi:.2f}]")

        if st is not None:
            dv = dev[m].dropna().to_numpy() * scale
            u, p = st.mannwhitneyu(dv, hv, alternative="two-sided")
            verdict = ("subset looks representative" if p >= 0.05
                       else "SPLITS DIFFER - state this explicitly")
            print(f"  development vs holdout: Mann-Whitney U = {u:.0f}, "
                  f"p = {p:.4g}  ({verdict})")
            print(f"  (unpaired by design - different recordings, not the same "
                  f"ones twice)")

    print(f"\n{'='*74}")
    print("HOW TO REPORT")
    print(f"{'='*74}")
    print("Quote the HOLDOUT figure as the validated result. Quote the full")
    print("corpus figure only as a descriptive corpus characteristic, and say")
    print("so in the same sentence. Never present the 272-recording aggregate")
    print("as independent validation: 40 of those recordings selected the")
    print("configuration being evaluated.")
    print("\nIf development and holdout do not differ significantly, that is")
    print("worth one sentence - it shows the subset was representative and")
    print("that the parameter study was not tuned on unusual recordings. It")
    print("does NOT make the full-corpus figure independent.")

    if args.out_prefix:
        dcols = [c for c in df.columns if not c.startswith("_")]
        dev[dcols].to_csv(f"{args.out_prefix}_dev40.csv", index=False)
        hold[dcols].to_csv(f"{args.out_prefix}_holdout232.csv", index=False)
        print(f"\nSaved -> {args.out_prefix}_dev40.csv ({len(dev)} rows)")
        print(f"Saved -> {args.out_prefix}_holdout232.csv ({len(hold)} rows)")
    print()


if __name__ == "__main__":
    main()
