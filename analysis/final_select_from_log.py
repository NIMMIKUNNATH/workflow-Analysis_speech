"""
final_select_from_log.py
========================
WER-based parameter selection over the parameter search that has ALREADY been
run, reading:

  study_log.csv      per-condition, per-case: wer, wder, sub, del, ins, ref_words
  clinical_*.csv     per-case clinical metrics incl. negation_n / negation_err

No transcription. Runs in about a second.

Three things this script is careful about, because each is a way to get a
wrong answer from correct-looking data:

1.  CORPUS WER IS MICRO-AVERAGED.  total(sub+del+ins) / total(ref_words), not
    the mean of per-case WER. The macro mean weights a 300-word case the same
    as a 1500-word case. Both are reported so you can see whether the ranking
    is stable across the two; if it is not, say which you used in the paper.

2.  CONDITIONS ARE COMPARED ON A COMMON CASE SET.  Conditions covering
    different subsets (e.g. a 40-case pilot vs the full 272) are not
    comparable on WER. --common-cases restricts every condition to the
    intersection before ranking.

3.  PRIMOCK CONDITIONS ARE EXCLUDED FROM THE FAREEZ SELECTION POOL.
    Cross-corpus transfer conditions are evaluation targets, not candidates.

Usage
-----
    python final_select_from_log.py --report-join      # inspect the mapping first
    python final_select_from_log.py --common-cases
    python final_select_from_log.py --common-cases --baseline model_large-v2
"""

from __future__ import annotations

import argparse
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = Path(os.environ.get("ASR_RESULTS_ROOT", REPO_ROOT / "results"))
ROOT = Path(os.environ.get("ASR_CLINICAL_RESULTS_ROOT", RESULTS_ROOT / "clinical"))
STUDY_LOG = Path(os.environ.get("ASR_STUDY_LOG", RESULTS_ROOT / "study_log.csv"))


# --------------------------------------------------------------------------- #
# Condition-name matching between study_log.csv and the clinical_*.csv files
# --------------------------------------------------------------------------- #

# Pairings that canon() cannot resolve from names alone. Confirmed by matching
# per-case WER vectors against study_log.csv (match_conditions.py):
#   model_medium -> clinical_medium40   max abs diff 4.9e-5  (4-dp rounding)
#   baseline     -> clinical_baseline40 max abs diff 9.6e-4, from a single
#                   one-word discrepancy on CAR0002; ref_words identical
#                   across all 40 cases, so this is the same run.
#   compute_fp16 -> None. No clinical file exists (nearest candidate differs by
#                   0.084). This condition therefore has no negation
#                   measurement and is excluded from the screened pool.
# clinical_272.csv and clinical_full.csv are byte-equivalent 270-case
# full-corpus runs, not search conditions, and are not candidates.
CONDITION_OVERRIDES: dict[str, str | None] = {
    "baseline": "clinical_baseline40",
    "model_medium": "clinical_medium40",
    "compute_fp16": None,
}

NON_CONDITION_FILES = {"clinical_errors", "clinical_272", "clinical_full"}


def canon(name: str) -> str:
    """
    'model_small' -> 'small';  'clinical_beam8' -> 'beam8';  'large-v2' -> 'largev2'

    Strips a leading 'model'/'clinical' tag and all non-alphanumerics so the two
    naming conventions collapse onto the same key.
    """
    s = name.lower().strip()
    s = re.sub(r"\.csv$", "", s)
    s = re.sub(r"^(clinical|model|cond|condition)[_\-]?", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def is_primock(name: str) -> bool:
    return "primock" in name.lower()


def find_clinical_files() -> dict[str, Path]:
    """Map canonical condition key -> clinical_*.csv path (Fareez only)."""
    out: dict[str, Path] = {}
    for path in sorted(ROOT.glob("clinical_*.csv")):
        if is_primock(path.name):
            continue
        if path.stem in NON_CONDITION_FILES:
            continue
        out[canon(path.stem)] = path
    return out


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def load_study_log(path: Path = STUDY_LOG) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"condition", "case", "wer", "sub", "del", "ins", "ref_words"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"study_log.csv is missing columns: {sorted(missing)}")
    df = df[~df["condition"].map(is_primock)].copy()
    return df


def aggregate_wer(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("condition")
    agg = pd.DataFrame({
        "n_cases":    g.size(),
        "ref_words":  g["ref_words"].sum(),
        "errors":     g["sub"].sum() + g["del"].sum() + g["ins"].sum(),
        "wer_macro":  g["wer"].mean(),
    })
    agg["wer_micro"] = agg["errors"] / agg["ref_words"]
    if "wder" in df.columns:
        agg["wder_macro"] = g["wder"].mean()
    return agg.reset_index()


def resolve_clinical(conditions: list[str]) -> dict[str, Path | None]:
    """condition name -> clinical CSV, using canon() with explicit overrides."""
    by_key = find_clinical_files()
    out: dict[str, Path | None] = {}
    for cond in conditions:
        if cond in CONDITION_OVERRIDES:
            stem = CONDITION_OVERRIDES[cond]
            out[cond] = (ROOT / f"{stem}.csv") if stem else None
        else:
            out[cond] = by_key.get(canon(cond))
    return out


def aggregate_negation(clinical_files: dict[str, Path | None],
                       keep_cases: set[str] | None = None) -> pd.DataFrame:
    rows = []
    for key, path in clinical_files.items():
        if path is None or not path.exists():
            continue
        cdf = pd.read_csv(path)
        if not {"negation_n", "negation_err"} <= set(cdf.columns):
            continue
        if keep_cases is not None and "case" in cdf.columns:
            cdf = cdf[cdf["case"].isin(keep_cases)]
        n = cdf["negation_n"].fillna(0).sum()
        e = cdf["negation_err"].fillna(0).sum()
        row = {
            "condition": key,
            "negation_n": float(n),
            "negation_err_count": float(e),
            "negation_error": (e / n) if n else float("nan"),
            "clinical_file": path.name,
            "n_cases_clinical": len(cdf),
        }
        if "medication_n" in cdf.columns:
            mn = cdf["medication_n"].fillna(0).sum()
            me = cdf["medication_err"].fillna(0).sum()
            row["medication_n"] = float(mn)
            row["medication_error"] = (me / mn) if mn else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_table(common_cases: bool = False) -> pd.DataFrame:
    df = load_study_log()

    keep_cases: set[str] | None = None
    if common_cases:
        per_cond = df.groupby("condition")["case"].apply(set)
        keep_cases = set.intersection(*per_cond.tolist()) if len(per_cond) else set()
        if not keep_cases:
            raise SystemExit("the intersection of case sets is empty — drop the "
                             "pilot/subset conditions and rerun")
        df = df[df["case"].isin(keep_cases)]

    wer = aggregate_wer(df)
    mapping = resolve_clinical(sorted(wer["condition"].unique()))
    neg = aggregate_negation(mapping, keep_cases)
    table = wer.merge(neg, on="condition", how="left") if not neg.empty else wer

    missing = [c for c, p in mapping.items() if p is None]
    if missing:
        table.attrs["no_negation"] = missing
    return table.sort_values("wer_micro").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

def select(table: pd.DataFrame,
           baseline: str | None = None,
           neg_tolerance: float = 0.005,
           primary: str = "wer_micro") -> dict:
    """
    Minimise WER subject to negation error not exceeding the baseline's by more
    than neg_tolerance. Conditions with no negation measurement are excluded
    from the screened pool: an unmeasured constraint is not a satisfied one.
    """
    unconstrained = table.loc[table[primary].idxmin()]

    result = {
        "argmin_wer": {
            "condition": unconstrained["condition"],
            primary: float(unconstrained[primary]),
            "negation_error": float(unconstrained.get("negation_error", float("nan"))),
        }
    }

    if baseline is None:
        result["screened"] = None
        return result

    base_rows = table[table["condition"] == baseline]
    if base_rows.empty:
        raise SystemExit(f"baseline '{baseline}' not found. Conditions: "
                         f"{sorted(table['condition'])}")
    base = base_rows.iloc[0]

    if pd.isna(base.get("negation_error", float("nan"))):
        raise SystemExit(f"baseline '{baseline}' has no negation measurement; "
                         f"the screen cannot be anchored")

    eligible = table[
        table["negation_error"].notna()
        & (table["negation_error"] <= base["negation_error"] + neg_tolerance)
    ]

    if eligible.empty:
        result["screened"] = {
            "condition": None,
            "reason": "non-degradation screen returned the empty set",
            "n_candidates": int(len(table)),
        }
        return result

    best = eligible.loc[eligible[primary].idxmin()]
    result["screened"] = {
        "condition": best["condition"],
        primary: float(best[primary]),
        "negation_error": float(best["negation_error"]),
        "baseline_negation_error": float(base["negation_error"]),
        "n_eligible": int(len(eligible)),
        "n_candidates": int(len(table)),
    }
    return result


def paired_bootstrap(df: pd.DataFrame, cond_a: str, cond_b: str,
                     n_boot: int = 10_000, seed: int = 0) -> dict:
    """
    Paired bootstrap over cases for micro-WER(a) - micro-WER(b).
    Cheap here because per-case error counts are already tabulated.
    """
    a = df[df["condition"] == cond_a].set_index("case")
    b = df[df["condition"] == cond_b].set_index("case")
    cases = sorted(set(a.index) & set(b.index))
    if not cases:
        raise SystemExit(f"no shared cases between {cond_a} and {cond_b}")

    ea = {c: a.loc[c, ["sub", "del", "ins"]].sum() for c in cases}
    wa = {c: a.loc[c, "ref_words"] for c in cases}
    eb = {c: b.loc[c, ["sub", "del", "ins"]].sum() for c in cases}
    wb = {c: b.loc[c, "ref_words"] for c in cases}

    def micro(sample, err, wrd):
        return sum(err[c] for c in sample) / max(1, sum(wrd[c] for c in sample))

    observed = micro(cases, ea, wa) - micro(cases, eb, wb)

    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        s = [rng.choice(cases) for _ in cases]
        deltas.append(micro(s, ea, wa) - micro(s, eb, wb))
    deltas.sort()

    p = 2 * min(sum(d >= 0 for d in deltas), sum(d <= 0 for d in deltas)) / n_boot
    return {
        "n_cases": len(cases),
        "delta_wer": observed,
        "ci_low": deltas[int(0.025 * n_boot)],
        "ci_high": deltas[int(0.975 * n_boot)],
        "p_two_sided": min(1.0, p),
    }


def paired_bootstrap_negation(cond_a: str, cond_b: str,
                              n_boot: int = 10_000, seed: int = 0) -> dict:
    """
    Paired bootstrap over cases for negation_error(a) - negation_error(b),
    where negation_error = sum(negation_err) / sum(negation_n).

    This is the test that decides whether a non-degradation screen is doing
    real work or rejecting configurations on a difference inside the noise
    floor. At n=40 a sub-percentage-point difference is unlikely to clear it.
    """
    mapping = resolve_clinical([cond_a, cond_b])
    for cond in (cond_a, cond_b):
        if mapping.get(cond) is None:
            raise SystemExit(f"no clinical file for '{cond}' — cannot test negation")

    da = pd.read_csv(mapping[cond_a]).set_index("case")
    db = pd.read_csv(mapping[cond_b]).set_index("case")
    cases = sorted(set(da.index) & set(db.index))
    if not cases:
        raise SystemExit(f"no shared cases between {cond_a} and {cond_b}")

    na = {c: float(da.loc[c, "negation_n"] or 0) for c in cases}
    ea = {c: float(da.loc[c, "negation_err"] or 0) for c in cases}
    nb = {c: float(db.loc[c, "negation_n"] or 0) for c in cases}
    eb = {c: float(db.loc[c, "negation_err"] or 0) for c in cases}

    tot_na, tot_nb = sum(na.values()), sum(nb.values())
    warning = None
    if abs(tot_na - tot_nb) > 1e-6:
        warning = (f"negation denominators differ ({cond_a}={tot_na:.0f}, "
                   f"{cond_b}={tot_nb:.0f}): the two conditions were scored "
                   f"against different reference token sets, so this "
                   f"comparison is not valid until that is resolved")

    def rate(sample, err, n):
        d = sum(n[c] for c in sample)
        return (sum(err[c] for c in sample) / d) if d else float("nan")

    observed = rate(cases, ea, na) - rate(cases, eb, nb)

    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        s = [rng.choice(cases) for _ in cases]
        d = rate(s, ea, na) - rate(s, eb, nb)
        if d == d:  # drop nan
            deltas.append(d)
    deltas.sort()

    p = 2 * min(sum(d >= 0 for d in deltas), sum(d <= 0 for d in deltas)) / len(deltas)
    out = {
        "n_cases": len(cases),
        f"negation_error[{cond_a}]": rate(cases, ea, na),
        f"negation_error[{cond_b}]": rate(cases, eb, nb),
        "delta_negation": observed,
        "ci_low": deltas[int(0.025 * len(deltas))],
        "ci_high": deltas[int(0.975 * len(deltas))],
        "p_two_sided": min(1.0, p),
    }
    if warning:
        out["WARNING"] = warning
    return out


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--common-cases", action="store_true",
                    help="restrict all conditions to their shared case set")
    ap.add_argument("--baseline", default=None,
                    help="condition name in study_log.csv to anchor the screen")
    ap.add_argument("--neg-tolerance", type=float, default=0.005)
    ap.add_argument("--report-join", action="store_true",
                    help="print the condition <-> clinical file mapping and exit")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="paired bootstrap on WER between two conditions")
    ap.add_argument("--compare-negation", nargs=2, metavar=("A", "B"),
                    help="paired bootstrap on negation error between two conditions")
    args = ap.parse_args()

    if args.report_join:
        df = load_study_log()
        conds = sorted(df["condition"].unique())
        mapping = resolve_clinical(conds)
        print(f"{len(conds)} conditions in study_log.csv\n")
        print(f"{'condition':32s} clinical file")
        for c in conds:
            p = mapping[c]
            note = " (override)" if c in CONDITION_OVERRIDES else ""
            print(f"{c:32s} {p.name if p else '-- none: excluded from screen --'}{note}")
        return

    table = build_table(common_cases=args.common_cases)

    cols = ["condition", "n_cases", "ref_words", "wer_micro", "wer_macro",
            "negation_error", "negation_n"]
    cols = [c for c in cols if c in table.columns]
    pd.set_option("display.width", 200)
    print("\n=== conditions ranked by micro WER ===")
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if table.attrs.get("no_negation"):
        print(f"\n  NOTE: no negation measurement for "
              f"{table.attrs['no_negation']} — excluded from the screened pool.")

    spread = table["n_cases"].nunique()
    if spread > 1 and not args.common_cases:
        print(f"\n  WARNING: conditions cover {spread} different case counts. "
              f"WER is not comparable across them. Rerun with --common-cases.")

    result = select(table, baseline=args.baseline, neg_tolerance=args.neg_tolerance)
    print("\n=== selection ===")
    print(f"argmin WER : {result['argmin_wer']}")
    print(f"screened   : {result['screened']}")

    if args.compare:
        df = load_study_log()
        print("\n=== paired bootstrap: WER ===")
        for k, v in paired_bootstrap(df, args.compare[0], args.compare[1]).items():
            print(f"  {k:28s} {v}")

    if args.compare_negation:
        print("\n=== paired bootstrap: negation error ===")
        for k, v in paired_bootstrap_negation(*args.compare_negation).items():
            print(f"  {k:28s} {v}")

    table.to_csv("selection_table.csv", index=False)
    print("\nwrote selection_table.csv")


if __name__ == "__main__":
    main()
