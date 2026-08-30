#!/usr/bin/env python3
"""
final_minimax.py - safety-constrained minimax configuration selection.

THE RULE
    For configuration theta and outcome k, let e_k(theta) be the development-set
    error and tau_k the maximum acceptable value. Define the normalised ratio

        R_k(theta) = e_k(theta) / tau_k

    The safety-eligible set is

        Theta_safe = { theta : e_k(theta) <= tau_k for every k }

    and the selection is

        theta* = argmin_{theta in Theta_safe} max_k R_k(theta)

    i.e. among configurations that violate no margin, choose the one whose worst
    normalised error is smallest. WER breaks ties only.

WHY BASELINE-ANCHORED TAU IS THE DEFAULT
    There is no published threshold for lexicon-based medication or negation
    error, so any absolute tau would be invented. Setting tau_k to the incumbent
    baseline makes the rule mechanical: R_k < 1 means "better than the system
    currently in use", and a configuration is eligible only if it degrades
    nothing. That is a standard non-inferiority framing and needs no external
    anchor.

    THIS IS STILL A CHOICE MADE WITH THE DATA VISIBLE. Say so in the manuscript.
    A baseline-anchored margin is defensible because it is mechanical, not
    because it is blind. The sensitivity analysis below is what carries the
    argument: if the selection is unchanged across a range of tau, the specific
    values are not load-bearing.

WHAT IT REPORTS
    1. the full ratio matrix, every configuration by every outcome
    2. eligibility and worst ratio, ranked
    3. the WER-only choice alongside the minimax choice - the comparison that
       shows whether optimising WER alone would have selected differently
    4. bootstrap selection stability: resample recordings, recompute the winner,
       report how often each configuration wins. A minimax optimum that wins
       35% of resamples is a set, not a point.
    5. sensitivity across a tau grid

USAGE
    # baseline-anchored (default)
    python final_minimax.py --results results --clinrate clinrate --sa sa

    # explicit thresholds
    python final_minimax.py --results results --clinrate clinrate --sa sa \\
        --tau-med 12.0 --tau-neg 5.5 --tau-wder 2.5 --tau-sawer 20.0

    # upper-confidence-bound (robust) variant
    python final_minimax.py --results results --clinrate clinrate --sa sa --ucb
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID = ["case", "recording", "file", "name", "id", "stem"]
OUTCOMES = ["medication", "negation", "wder", "sa_wer"]

# Not one-factor family members. Reported, never silently selected from.
FOCUSED = {"best_combo", "v2_merge150"}
TRANSFER = {"v2_beam1", "v2_beam3", "v2_vad_500", "v2_vad_off"}

def idcol(df, label):
    for c in ID:
        if c in df.columns:
            return c
    sys.exit(f"{label}: no identifier column. Saw {list(df.columns)}")


def load(results, clinrate, sa):
    """Assemble a per-recording frame per condition across all outcomes."""
    conds = {}
    for p in sorted(Path(results).glob("*.csv")):
        name = p.stem
        d = pd.read_csv(p).rename(columns={idcol(pd.read_csv(p), name): "case"})
        keep = d[["case"] + [c for c in ("wer", "wder") if c in d.columns]]
        conds[name] = keep

    for name in list(conds):
        cp = Path(clinrate) / f"{name}.csv"
        if cp.exists():
            c = pd.read_csv(cp)
            c = c.rename(columns={idcol(c, name): "case"})
            conds[name] = conds[name].merge(
                c[["case"] + [m for m in ("medication", "negation") if m in c.columns]],
                on="case", how="left")
        sp = Path(sa) / f"sa_{name}.csv"
        if sp.exists():
            s = pd.read_csv(sp)
            s = s.rename(columns={idcol(s, name): "case"})
            if "sa_wer" in s.columns:
                conds[name] = conds[name].merge(
                    s[["case", "sa_wer"]], on="case", how="left")
    return conds


def pct(series):
    """Return mean as a percentage, whatever scale the column is on."""
    v = series.dropna().astype(float)
    if not len(v):
        return np.nan
    return v.mean() * (100.0 if v.max() <= 1.5 else 1.0)


def upper95(series, n_boot, seed):
    v = series.dropna().astype(float).to_numpy()
    if len(v) < 3:
        return np.nan
    scale = 100.0 if v.max() <= 1.5 else 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    return float(np.percentile(v[idx].mean(axis=1), 97.5)) * scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--clinrate", required=True)
    ap.add_argument("--sa", required=True)
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--tau-med", type=float)
    ap.add_argument("--tau-neg", type=float)
    ap.add_argument("--tau-wder", type=float)
    ap.add_argument("--tau-sawer", type=float)
    ap.add_argument("--ucb", action="store_true",
                    help="use upper 95%% confidence limits instead of means")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--include-focused", action="store_true",
                    help="allow post-hoc focused experiments into the selection")
    ap.add_argument("--include-transfer", action="store_true",
                    help="allow two-factor transfer conditions into the selection")
    args = ap.parse_args()

    conds = load(args.results, args.clinrate, args.sa)
    if args.baseline not in conds:
        sys.exit(f"baseline '{args.baseline}' not found")

    # ---- point estimates -------------------------------------------------
    est = {}
    for name, d in conds.items():
        row = {}
        for k in OUTCOMES + ["wer"]:
            if k in d.columns:
                row[k] = (upper95(d[k], args.boot, args.seed) if args.ucb
                          else pct(d[k]))
            else:
                row[k] = np.nan
        row["n"] = len(d)
        est[name] = row
    E = pd.DataFrame(est).T

    complete = E.dropna(subset=OUTCOMES)
    dropped = sorted(set(E.index) - set(complete.index))

    # ---- thresholds ------------------------------------------------------
    b = E.loc[args.baseline]
    tau = {
        "medication": args.tau_med if args.tau_med else b["medication"],
        "negation":   args.tau_neg if args.tau_neg else b["negation"],
        "wder":       args.tau_wder if args.tau_wder else b["wder"],
        "sa_wer":     args.tau_sawer if args.tau_sawer else b["sa_wer"],
    }
    anchored = not any([args.tau_med, args.tau_neg, args.tau_wder, args.tau_sawer])

    print(f"\n{'='*94}")
    print(f"SAFETY-CONSTRAINED MINIMAX SELECTION"
          f"{'  ·  UPPER 95% BOUNDS' if args.ucb else '  ·  POINT ESTIMATES'}")
    print(f"{'='*94}")
    print("thresholds (tau):", "  ".join(f"{k}={v:.2f}" for k, v in tau.items()))
    if anchored:
        print(f"anchored to the '{args.baseline}' condition: a configuration is")
        print("eligible only if it degrades no outcome relative to baseline.")
        print("NOTE: chosen with the data visible. Disclose this, and rely on")
        print("the sensitivity analysis rather than on the specific values.")
    if dropped:
        print(f"\nexcluded, incomplete outcomes ({len(dropped)}): {', '.join(dropped)}")

    # ---- ratio matrix ----------------------------------------------------
    R = pd.DataFrame({k: complete[k] / tau[k] for k in OUTCOMES})
    R["worst"] = R.max(axis=1)
    R["eligible"] = (R[OUTCOMES] <= 1.0).all(axis=1)
    R["wer"] = complete["wer"]
    R["family"] = ["focused" if i in FOCUSED else
                   "transfer" if i in TRANSFER else "primary" for i in R.index]

    # BUG FIX. The selection pool previously excluded only the focused
    # experiments and silently left the four two-factor transfer conditions in.
    # Selection and multiplicity are separate questions, but the answer is the
    # same for both: a two-factor condition cannot compete against one-factor
    # conditions in a one-factor selection. Both groups are now opt-in.
    excluded_families = set()
    if not args.include_focused:
        excluded_families.add("focused")
    if not args.include_transfer:
        excluded_families.add("transfer")
    sel_pool = R[~R.family.isin(excluded_families)]
    if excluded_families:
        n_out = len(R) - len(sel_pool)
        print(f"\nselection pool: {len(sel_pool)} primary conditions "
              f"({n_out} excluded: {', '.join(sorted(excluded_families))}). "
              f"All are shown in the matrix below but cannot be selected.")

    print(f"\n{'-'*94}")
    hdr = (f"{'configuration':<20}{'WER':>7}{'R_med':>8}{'R_neg':>8}{'R_wder':>8}"
           f"{'R_sa':>8}{'worst':>8}  {'family':<9} eligible")
    print(hdr); print("-" * len(hdr))
    for i, r in R.sort_values("worst").iterrows():
        print(f"{i:<20}{r['wer']:>7.2f}{r['medication']:>8.3f}{r['negation']:>8.3f}"
              f"{r['wder']:>8.3f}{r['sa_wer']:>8.3f}{r['worst']:>8.3f}"
              f"  {r['family']:<9} {'yes' if r['eligible'] else 'NO'}")

    # ---- the two selections ---------------------------------------------
    print(f"\n{'='*94}")
    wer_only = sel_pool["wer"].idxmin()
    print(f"WER-ONLY SELECTION      {wer_only}  (WER {sel_pool.loc[wer_only,'wer']:.2f}%)")
    wr = R.loc[wer_only]
    print(f"  its ratios: med {wr['medication']:.3f}  neg {wr['negation']:.3f}  "
          f"wder {wr['wder']:.3f}  sa {wr['sa_wer']:.3f}  -> worst {wr['worst']:.3f}"
          f"  {'ELIGIBLE' if wr['eligible'] else 'VIOLATES A MARGIN'}")

    elig = sel_pool[sel_pool.eligible]
    if elig.empty:
        print("\nMINIMAX SELECTION       none - no configuration satisfies every")
        print("margin. Under the prespecified rule no configuration is declared")
        print("and development continues. Do NOT relax tau to obtain a winner.")
        theta = None
    else:
        best = elig["worst"].min()
        tied = elig[np.isclose(elig["worst"], best, atol=1e-9)]
        theta = tied["wer"].idxmin() if len(tied) > 1 else tied.index[0]
        print(f"MINIMAX SELECTION       {theta}  (worst ratio {best:.3f}, "
              f"WER {R.loc[theta,'wer']:.2f}%)")
        if len(tied) > 1:
            print(f"  {len(tied)} tied on worst ratio; WER broke the tie")
        print(f"  {len(elig)}/{len(sel_pool)} configurations are safety-eligible")

    if theta and theta != wer_only:
        print(f"\n  THE TWO RULES DISAGREE: WER-only picks {wer_only}, "
              f"minimax picks {theta}.")
        print("  This is the comparison the paper turns on - report both.")
    elif theta:
        print(f"\n  Both rules select {theta}. The minimax analysis still shows")
        print("  the choice is robust across clinical and speaker outcomes,")
        print("  which selection on WER alone cannot establish.")

    # ---- selection stability --------------------------------------------
    if theta:
        print(f"\n{'-'*94}")
        print(f"SELECTION STABILITY  ·  {args.boot} bootstrap resamples of recordings")
        print("-" * 94)
        cases = list(conds[args.baseline]["case"])
        rng = np.random.default_rng(args.seed)
        wins = {}
        for _ in range(args.boot):
            draw = rng.choice(len(cases), len(cases), replace=True)
            picked = [cases[i] for i in draw]
            best_name, best_worst = None, np.inf
            for name in sel_pool.index:
                d = conds[name].set_index("case")
                sub = d.reindex(picked)
                worst = -np.inf
                ok = True
                for k in OUTCOMES:
                    if k not in sub.columns:
                        ok = False; break
                    v = pct(sub[k])
                    if np.isnan(v):
                        ok = False; break
                    r = v / tau[k]
                    if r > 1.0:
                        ok = False
                    worst = max(worst, r)
                if ok and worst < best_worst:
                    best_name, best_worst = name, worst
            if best_name:
                wins[best_name] = wins.get(best_name, 0) + 1
        tot = sum(wins.values()) or 1
        for name, c in sorted(wins.items(), key=lambda x: -x[1])[:8]:
            print(f"  {name:<24}{c/tot*100:6.1f}%  ({c}/{tot})")
        # BUG FIX. This previously printed max(wins.values()), i.e. the MODAL
        # winner's share, while labelling it as the selected configuration's
        # stability. When the point-estimate winner is not the modal bootstrap
        # winner - exactly the case where stability matters - that overstated
        # the result. Report the selected configuration's own share, and name
        # the modal winner separately when they differ.
        sel_share = wins.get(theta, 0) / tot
        modal = max(wins, key=wins.get) if wins else None
        print(f"\n  Selected configuration ({theta}) wins "
              f"{sel_share*100:.1f}% of resamples.")
        if modal and modal != theta:
            print(f"  MODAL bootstrap winner is {modal} at "
                  f"{wins[modal]/tot*100:.1f}% - it is NOT the point-estimate")
            print("  selection. Report both; the disagreement is the finding.")
        print("  Below roughly 50%, report a set of comparable configurations")
        print("  rather than a single optimum.")

    # ---- tau sensitivity -------------------------------------------------
    print(f"\n{'-'*94}")
    print("THRESHOLD SENSITIVITY  ·  all tau scaled together")
    print("-" * 94)
    for f in (0.8, 0.9, 1.0, 1.1, 1.2):
        t = {k: v * f for k, v in tau.items()}
        Rf = pd.DataFrame({k: complete[k] / t[k] for k in OUTCOMES})
        ok = (Rf <= 1.0).all(axis=1)
        pool = ok[[i for i in ok.index if i in sel_pool.index]]
        e = pool[pool]
        if e.empty:
            print(f"  tau x {f:.1f}   no eligible configuration")
        else:
            w = Rf.loc[e.index].max(axis=1)
            pick = w.idxmin()
            print(f"  tau x {f:.1f}   {pick:<24}worst {w.min():.3f}   "
                  f"{len(e)} eligible")
    print("\n  If the selection is unchanged across this range, the specific")
    print("  tau values are not load-bearing and the choice is defensible.\n")


if __name__ == "__main__":
    main()
