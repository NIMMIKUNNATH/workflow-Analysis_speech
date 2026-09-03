#!/usr/bin/env python3
"""
Synthetic validation suite for the metric implementations.

Released synthetic scorer-validation suite accompanying the manuscript. Every
case has hand-computed expected values; the suite calls the retained scorer
functions and reports pass/fail.

Emits synthetic_validation.csv, which becomes a supplementary table.

ADAPTER
-------
The imports below are best guesses from your repository layout. If they fail,
edit the ADAPTER block only -- the cases themselves are implementation-agnostic.

Expected signatures:
    compute_wer_wder(ref_pairs, hyp_words) -> dict with wer / wder / sa_wer
        ref_pairs : [(word, speaker), ...]
        hyp_words : [(word, speaker), ...]   (or [word, ...]; see HYP_IS_PAIRS)
    compute_der(ref_segs, hyp_segs) -> dict with der / missed / false_alarm /
                                       confusion
        segs : [(start_s, end_s, speaker), ...]

USAGE
    python validation/synthetic_validation.py \
        --module scoring.final_score_primock \
        --out validation/synthetic_validation_reproduced.csv
"""

import argparse
import importlib
import inspect
import sys
import traceback
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# --------------------------------------------------------------- adapter ---
HYP_IS_PAIRS = True   # set False if compute_wer_wder takes a flat word list
RESULTS_ARE_RATIOS = True  # retained scorer returns ratios, including values >1
COLLAR = 0.25         # NIST collar used in Methods 3.7.2


def load_scorers(module_name):
    try:
        m = importlib.import_module(module_name)
    except Exception:
        sys.exit(f"ERROR: could not import '{module_name}'.\n"
                 f"Run this from the directory containing that file, or pass\n"
                 f"--module with the right name.\n\n{traceback.format_exc()}")
    missing = [f for f in ("compute_wer_wder", "compute_der")
               if not hasattr(m, f)]
    if missing:
        cands = [n for n, o in vars(m).items()
                 if callable(o) and any(k in n.lower()
                                        for k in ("wer", "der", "score"))]
        sys.exit(f"ERROR: {module_name} has no {missing}.\n"
                 f"Callables that look relevant: {cands}\n"
                 f"Edit the ADAPTER block to point at the right names.")
    for f in ("compute_wer_wder", "compute_der"):
        print(f"  {f}{inspect.signature(getattr(m, f))}")
    return m.compute_wer_wder, m.compute_der


# ----------------------------------------------------------------- cases ---
# ref / hyp are lists of (word, speaker). Expected values are hand-computed.
# WER    = 100 * (S + D + I) / N
# WDER   = 100 * (speaker-wrong among substituted + correct words) / (S + C)
# SA-WER = 100 * (words wrong in form OR speaker, plus insertions) / N

LEX_CASES = [
    dict(
        id="L01", name="identical",
        ref=[("the", "A"), ("patient", "A"), ("has", "A"), ("no", "A"),
             ("chest", "A"), ("pain", "A")],
        hyp=[("the", "A"), ("patient", "A"), ("has", "A"), ("no", "A"),
             ("chest", "A"), ("pain", "A")],
        wer=0.0, wder=0.0, sa_wer=0.0,
        note="N=6, no errors",
    ),
    dict(
        id="L02", name="one substitution",
        ref=[("the", "A"), ("patient", "A"), ("has", "A"), ("no", "A"),
             ("chest", "A"), ("pain", "A")],
        hyp=[("the", "A"), ("patient", "A"), ("has", "A"), ("some", "A"),
             ("chest", "A"), ("pain", "A")],
        wer=100 / 6, wder=0.0, sa_wer=100 / 6,
        note="S=1, N=6",
    ),
    dict(
        id="L03", name="terminal deletion",
        ref=[("no", "A"), ("chest", "A"), ("pain", "A")],
        hyp=[("no", "A"), ("chest", "A")],
        wer=100 / 3, wder=0.0, sa_wer=100 / 3,
        note="D=1, N=3",
    ),
    dict(
        id="L04", name="internal deletion",
        ref=[("the", "A"), ("patient", "A"), ("has", "A"), ("no", "A"),
             ("chest", "A"), ("pain", "A")],
        hyp=[("the", "A"), ("patient", "A"), ("has", "A"), ("chest", "A"),
             ("pain", "A")],
        wer=100 / 6, wder=0.0, sa_wer=100 / 6,
        note="D=1 mid-sequence; the negation-deletion case",
    ),
    dict(
        id="L05", name="one insertion",
        ref=[("no", "A"), ("chest", "A"), ("pain", "A")],
        hyp=[("no", "A"), ("uh", "A"), ("chest", "A"), ("pain", "A")],
        wer=100 / 3, wder=0.0, sa_wer=100 / 3,
        note="I=1, N=3; SA-WER includes insertions",
    ),
    dict(
        id="L06", name="S + D + I combined",
        ref=[("a", "A"), ("b", "A"), ("c", "A"), ("d", "A"), ("e", "A")],
        hyp=[("a", "A"), ("x", "A"), ("c", "A"), ("z", "A"), ("e", "A"),
             ("f", "A")],
        wer=100 * 3 / 5, wder=0.0, sa_wer=100 * 3 / 5,
        note="S=2, I=1, N=5",
    ),
    dict(
        id="L07", name="empty hypothesis",
        ref=[("no", "A"), ("chest", "A"), ("pain", "A")],
        hyp=[],
        wer=None, wder=None, sa_wer=None, expect_none=True,
        note="specified scorer policy: an empty transcript returns no result",
    ),
    dict(
        id="L08", name="hypothesis longer than reference",
        ref=[("yes", "A")],
        hyp=[("yes", "A"), ("i", "A"), ("do", "A")],
        wer=200.0, wder=0.0, sa_wer=200.0,
        note="I=2, N=1; WER exceeds 100%",
    ),
]

SPK_CASES = [
    dict(
        id="S01", name="all speakers wrong, words correct",
        ref=[("hello", "A"), ("there", "A"), ("hi", "B"), ("doctor", "B")],
        hyp=[("hello", "B"), ("there", "B"), ("hi", "A"), ("doctor", "A")],
        wer=0.0, wder=0.0, sa_wer=0.0,
        note="WDER is permutation-invariant: the A<->B swap maps to zero",
    ),
    dict(
        id="S02", name="one word on the wrong speaker",
        ref=[("a", "A"), ("b", "A"), ("c", "B"), ("d", "B")],
        hyp=[("a", "A"), ("b", "B"), ("c", "B"), ("d", "B")],
        wer=0.0, wder=25.0, sa_wer=25.0,
        note="1 of 4 aligned words mislabelled; no mapping fixes it",
    ),
    dict(
        id="S03", name="speaker error on a substituted word",
        ref=[("a", "A"), ("b", "A"), ("c", "B"), ("d", "B")],
        hyp=[("a", "A"), ("x", "B"), ("c", "B"), ("d", "B")],
        wer=25.0, wder=25.0, sa_wer=25.0,
        note="substitutions count in the WDER denominator",
    ),
    dict(
        id="S04", name="deletion excluded from WDER",
        ref=[("a", "A"), ("b", "A"), ("c", "B"), ("d", "B")],
        hyp=[("a", "A"), ("c", "B"), ("d", "B")],
        wer=25.0, wder=0.0, sa_wer=25.0,
        note="D excluded from WDER; 3 aligned words all correctly labelled",
    ),
    dict(
        id="S05", name="SA-WER >= WER by construction",
        ref=[("a", "A"), ("b", "A"), ("c", "B"), ("d", "B")],
        hyp=[("a", "A"), ("x", "A"), ("c", "A"), ("d", "B")],
        wer=25.0, wder=100 * 1 / 4, sa_wer=50.0,
        note="1 lexical error + 1 speaker-only error",
    ),
]

# segs are (start, end, speaker) in seconds; 250 ms collar
DER_CASES = [
    dict(
        id="D01", name="perfect diarization",
        ref=[(0.0, 10.0, "A"), (10.0, 20.0, "B")],
        hyp=[(0.0, 10.0, "A"), (10.0, 20.0, "B")],
        der=0.0, missed=0.0, false_alarm=0.0, confusion=0.0,
        note="identical boundaries and labels",
    ),
    dict(
        id="D02", name="labels swapped",
        ref=[(0.0, 10.0, "A"), (10.0, 20.0, "B")],
        hyp=[(0.0, 10.0, "B"), (10.0, 20.0, "A")],
        der=0.0, missed=0.0, false_alarm=0.0, confusion=0.0,
        note="optimal mapping absorbs a global swap",
    ),
    dict(
        id="D03", name="two seconds missed",
        ref=[(0.0, 10.0, "A")],
        hyp=[(0.0, 8.0, "A")],
        der=None, missed=None, false_alarm=0.0, confusion=0.0,
        note="~2 s missed of 10 s, less collar at the boundary",
    ),
    dict(
        id="D04", name="two seconds false alarm",
        ref=[(0.0, 10.0, "A")],
        hyp=[(0.0, 12.0, "A")],
        der=None, missed=0.0, false_alarm=None, confusion=None,
        note="hypothesis over-runs the reference",
    ),
    dict(
        id="D05", name="whole second speaker confused",
        ref=[(0.0, 10.0, "A"), (10.0, 20.0, "B")],
        hyp=[(0.0, 10.0, "A"), (10.0, 20.0, "A")],
        der=None, missed=0.0, false_alarm=0.0, confusion=None,
        note="half of reference speech assigned to the wrong speaker",
    ),
    dict(
        id="D06", name="empty hypothesis",
        ref=[(0.0, 10.0, "A")],
        hyp=[],
        der=100.0, missed=100.0, false_alarm=0.0, confusion=0.0,
        note="all reference speech missed",
    ),
]


# --------------------------------------------------------------- harness ---
def get(d, *names):
    """Pull a value out of a result dict under any of several key spellings."""
    for n in names:
        if n in d:
            return d[n]
    lower = {k.lower().replace("-", "_"): v for k, v in d.items()}
    for n in names:
        k = n.lower().replace("-", "_")
        if k in lower:
            return lower[k]
    return None


def as_percent(v):
    """Convert the retained scorer's ratio-valued outputs to percent."""
    if v is None:
        return None
    v = float(v)
    return v * 100.0 if RESULTS_ARE_RATIOS else v


def check(obs, exp, tol):
    if exp is None:
        return "n/a", ""
    if obs is None:
        return "FAIL", "scorer returned nothing"
    return ("PASS", "") if abs(obs - exp) <= tol else \
           ("FAIL", f"diff {obs - exp:+.4f}")


def run_lexical(fn, cases, tol, rows):
    for c in cases:
        hyp = c["hyp"] if HYP_IS_PAIRS else [w for w, _ in c["hyp"]]
        try:
            r = fn(c["ref"], hyp)
        except Exception as e:
            for m in ("wer", "wder", "sa_wer"):
                rows.append(dict(case=c["id"], name=c["name"], metric=m,
                                 expected=c.get(m), observed=None,
                                 status="ERROR", detail=str(e)[:80],
                                 note=c["note"]))
            continue
        if r is None:
            for m in ("wer", "wder", "sa_wer"):
                expected = c.get(m)
                status = "PASS" if c.get("expect_none") else \
                         ("n/a" if expected is None else "FAIL")
                detail = "expected no result" if c.get("expect_none") else \
                         "scorer returned no result"
                rows.append(dict(case=c["id"], name=c["name"], metric=m,
                                 expected=expected, observed=None,
                                 status=status, detail=detail,
                                 note=c["note"]))
            continue
        for m, keys in [("wer", ("wer",)),
                        ("wder", ("wder",)),
                        ("sa_wer", ("sa_wer", "sawer", "sa wer"))]:
            obs = as_percent(get(r, *keys))
            st, det = check(obs, c.get(m), tol)
            rows.append(dict(case=c["id"], name=c["name"], metric=m,
                             expected=c.get(m), observed=obs,
                             status=st, detail=det, note=c["note"]))


def run_der(fn, cases, tol, rows):
    for c in cases:
        try:
            r = fn(c["ref"], c["hyp"])
        except Exception as e:
            for m in ("der", "missed", "false_alarm", "confusion"):
                rows.append(dict(case=c["id"], name=c["name"], metric=m,
                                 expected=c.get(m), observed=None,
                                 status="ERROR", detail=str(e)[:80],
                                 note=c["note"]))
            continue
        for m, keys in [("der", ("der",)),
                        ("missed", ("missed", "missed_speech")),
                        ("false_alarm", ("false_alarm", "fa", "falsealarm")),
                        ("confusion", ("confusion", "conf"))]:
            obs = as_percent(get(r, *keys))
            st, det = check(obs, c.get(m), tol)
            rows.append(dict(case=c["id"], name=c["name"], metric=m,
                             expected=c.get(m), observed=obs,
                             status=st, detail=det, note=c["note"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="final_score_primock",
                    help="module exposing compute_wer_wder and compute_der")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--out", default="synthetic_validation.csv")
    args = ap.parse_args()

    print(f"importing {args.module}")
    wer_fn, der_fn = load_scorers(args.module)

    rows = []
    run_lexical(wer_fn, LEX_CASES, args.tol, rows)
    run_lexical(wer_fn, SPK_CASES, args.tol, rows)
    run_der(der_fn, DER_CASES, args.tol, rows)

    df = pd.DataFrame(rows)
    df["expected"] = df["expected"].astype(float).round(4)
    df["observed"] = df["observed"].astype(float).round(4)
    df.to_csv(args.out, index=False)

    checked = df[df.status.isin(["PASS", "FAIL", "ERROR"])]
    n_pass = (checked.status == "PASS").sum()
    print("\n" + df[["case", "name", "metric", "expected", "observed",
                     "status", "detail"]].to_string(index=False))
    print(f"\n{n_pass}/{len(checked)} checks passed "
          f"({(df.status == 'n/a').sum()} cases have no closed-form "
          f"expectation under a collar and are reported as observed values)")
    print(f"wrote {args.out}")

    bad = df[df.status.isin(["FAIL", "ERROR"])]
    if len(bad):
        print("\nfailures:")
        print(bad[["case", "metric", "expected", "observed", "detail"]]
              .to_string(index=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
