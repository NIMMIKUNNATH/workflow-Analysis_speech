#!/usr/bin/env python3
"""
final_surrogate.py - does an additive surrogate over parameter effects predict
the observed multi-factor combinations?

WHY THIS EXISTS
    A natural next step after a one-factor-at-a-time search is surrogate
    optimisation: fit a cheap model of the response surface from the sampled
    points, then use it to propose configurations worth running. Before
    spending GPU time on that, it is worth asking whether the surface is
    learnable from the data we have.

    It is not, and the reason is informative rather than a limitation.

THE TEST
    The one-factor conditions give a main effect for each parameter, measured
    against a fixed baseline. The six multi-factor conditions (four transfer,
    two focused) were never used to fit anything, so they are a genuine
    held-out set.

    1. ADDITIVE SURROGATE. Predict each combination as
           baseline + sum of its main effects
       and compare against the observed value. If effects compose, the error
       should be small relative to the spread across configurations.

    2. INTERACTION MODEL. If the additive prediction fails, fit
           delta = a * (model effect) + b * (parameter effect)
       by least squares. The value of b says how much of a parameter's
       measured effect survives once the model is changed. b near 1 means
       effects compose; b near 0 means the model change absorbs them.

    3. IMPLICATION FOR FURTHER SEARCH. If b is near zero the response surface
       is flat in the parameter dimensions once the model is fixed, and neither
       a grid nor a surrogate-guided search can find anything. That conclusion
       is worth more than the search would have been, and costs no GPU time.

WHAT IT DOES NOT DO
    This is not a general surrogate optimiser. Fitting a Gaussian process or
    RBF surrogate over 16 parameters from 42 points - most of them varying one
    parameter from a single baseline - would be badly under-determined, and the
    fitted surface would mostly reflect the prior rather than the data. The
    additive-versus-interaction test asks the one question the data can answer.

USAGE
    python final_surrogate.py
    python final_surrogate.py --results /path/to/results --auto
"""
import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

# ---------------------------------------------------------------------------
# Measured values, 40-recording development subset, 25 August 2026.
# Override with --results to read from the condition CSVs instead.
# ---------------------------------------------------------------------------
BASELINE = {"wer": 18.54, "negation": 5.13}

SINGLE = {   # condition -> (wer, negation).  One factor changed from baseline.
    "model_v2":     (14.26, 5.87),   # model large-v3 -> large-v2
    "beam1":        (15.94, 5.61),   # beam 5 -> 1
    "beam3":        (16.95, 5.12),   # beam 5 -> 3
    "vad_sil_500":  (17.35, 5.73),   # VAD min silence 250 -> 500 ms
    "vad_sil_1000": (17.25, 4.33),   # VAD min silence 250 -> 1000 ms
    "vad_off":      (18.10, 4.43),   # VAD filter on -> off
    "merge_150":    (16.97, 5.45),   # merge gap 1.00 -> 1.50 s
}

COMBO = {    # condition -> (component single-factor terms, wer, negation)
    "v2_beam1":    (["model_v2", "beam1"],        13.87, 6.78),
    "v2_beam3":    (["model_v2", "beam3"],        13.64, 5.88),
    "v2_vad_500":  (["model_v2", "vad_sil_500"],  13.62, 6.36),
    "best_combo":  (["model_v2", "vad_sil_1000"], 13.36, 6.24),
    "v2_vad_off":  (["model_v2", "vad_off"],      14.01, 5.56),
    "v2_merge150": (["model_v2", "merge_150"],    13.82, 5.83),
}

# Mapping used only by --auto, to read the same quantities from result CSVs.
CSV_NAMES = {
    "model_v2": "model_largev2", "beam1": "beam1", "beam3": "beam3",
    "vad_sil_500": "vad_silence_500", "vad_sil_1000": "vad_silence_1000",
    "vad_off": "vad_off", "merge_150": "merge_gap_150",
    "v2_beam1": "v2_beam1", "v2_beam3": "v2_beam3",
    "v2_vad_500": "v2_vad_500", "best_combo": "best_combo",
    "v2_vad_off": "v2_vad_off", "v2_merge150": "v2_merge150",
}


def load_from_csvs(results, clinrate):
    """Rebuild SINGLE/COMBO means from the condition CSVs, for reproducibility."""
    if pd is None:
        sys.exit("pandas required for --auto")

    def mean_of(name, folder, col):
        p = Path(folder) / f"{name}.csv"
        if not p.exists():
            return None
        d = pd.read_csv(p)
        if col not in d.columns:
            return None
        v = d[col].dropna().astype(float)
        return v.mean() * (100.0 if v.max() <= 1.5 else 1.0)

    out = {}
    for key, csvname in CSV_NAMES.items():
        w = mean_of(csvname, results, "wer")
        n = mean_of(csvname, clinrate, "negation")
        if w is None or n is None:
            print(f"  missing: {csvname}")
            continue
        out[key] = (w, n)
    return out


def additive_test(baseline, single, combo, metric_ix, label):
    """Predict each held-out combination as baseline + sum of main effects."""
    print(f"\n{'='*78}")
    print(f"1. ADDITIVE SURROGATE  ·  {label}")
    print(f"{'='*78}")
    print("   prediction = baseline + sum of the single-factor effects")
    print("   the combinations were NOT used to fit anything\n")

    base = baseline["wer" if metric_ix == 0 else "negation"]
    eff = {k: v[metric_ix] - base for k, v in single.items()}

    print(f"{'combination':<15}{'predicted':>11}{'observed':>11}{'error':>9}")
    print("-" * 46)
    errs = []
    for name, (terms, w, n) in combo.items():
        obs = w if metric_ix == 0 else n
        miss = [t for t in terms if t not in eff]
        if miss:
            print(f"{name:<15}   missing main effect: {', '.join(miss)}")
            continue
        pred = base + sum(eff[t] for t in terms)
        errs.append(obs - pred)
        print(f"{name:<15}{pred:>11.2f}{obs:>11.2f}{obs - pred:>+9.2f}")
    if not errs:
        return None
    e = np.array(errs)
    print("-" * 46)
    print(f"{'mean error':<15}{'':>11}{'':>11}{e.mean():>+9.2f}")
    print(f"{'RMSE':<15}{'':>11}{'':>11}{np.sqrt((e**2).mean()):>9.2f}")

    vals = [v[metric_ix] for v in list(single.values())] + \
           [(c[1] if metric_ix == 0 else c[2]) for c in combo.values()] + [base]
    rng = max(vals) - min(vals)
    rmse = np.sqrt((e**2).mean())
    print(f"\n   observed range across all configurations: {rng:.2f}")
    print(f"   surrogate RMSE as a fraction of that range: {rmse/rng*100:.0f}%")
    if rmse / rng > 0.10:
        print("   -> the additive surrogate does NOT predict the combinations.")
    else:
        print("   -> effects compose acceptably; an additive surrogate is usable.")
    return e


def interaction_fit(baseline, single, combo, metric_ix, label):
    """
    Fit  delta = a * model_effect + b * parameter_effect  by least squares.
    b is the fraction of a parameter's measured effect that survives the
    model change. Additivity implies a = b = 1.
    """
    print(f"\n{'='*78}")
    print(f"2. INTERACTION MODEL  ·  {label}")
    print(f"{'='*78}")
    base = baseline["wer" if metric_ix == 0 else "negation"]
    eff = {k: v[metric_ix] - base for k, v in single.items()}

    rows, ys, names = [], [], []
    for name, (terms, w, n) in combo.items():
        model_terms = [t for t in terms if t.startswith("model")]
        param_terms = [t for t in terms if not t.startswith("model")]
        if not all(t in eff for t in terms):
            continue
        rows.append([sum(eff[t] for t in model_terms),
                     sum(eff[t] for t in param_terms)])
        ys.append((w if metric_ix == 0 else n) - base)
        names.append(name)
    if len(rows) < 3:
        print("   too few combinations to fit")
        return
    A, y = np.array(rows), np.array(ys)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    resid = y - pred

    print(f"   fitted:  delta = {coef[0]:.3f} * model_effect "
          f"+ {coef[1]:.3f} * parameter_effect")
    print(f"   additive model assumes both coefficients equal 1.000")
    print(f"   RMSE {np.sqrt((resid**2).mean()):.3f}\n")
    print(f"{'combination':<15}{'observed':>11}{'fitted':>10}{'residual':>10}")
    print("-" * 46)
    for nm, o, f in zip(names, y, pred):
        print(f"{nm:<15}{o:>11.2f}{f:>10.2f}{o - f:>10.2f}")

    b = coef[1]
    print(f"\n   INTERPRETATION")
    if abs(b) < 0.15:
        print(f"   The parameter coefficient is {b:.3f}. Once the model is")
        print(f"   changed, a parameter delivers roughly {abs(b)*100:.0f}% of the effect")
        print(f"   it had on the baseline model. The response surface is")
        print(f"   essentially FLAT in the parameter dimensions.")
        print(f"\n   Neither a grid search nor a surrogate-guided search over")
        print(f"   these parameters can find a configuration materially better")
        print(f"   than the model change alone. Further search is not warranted.")
    elif abs(b) < 0.6:
        print(f"   The parameter coefficient is {b:.3f}: effects are strongly")
        print(f"   sub-additive but not absent. A targeted search might yield")
        print(f"   a modest gain; budget accordingly.")
    else:
        print(f"   The parameter coefficient is {b:.3f}: effects largely compose.")
        print(f"   A surrogate over the parameter space is worth fitting, and")
        print(f"   grid or surrogate-guided search may find better points.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", help="directory of per-condition CSVs")
    ap.add_argument("--clinrate", help="directory of per-recording clinical rates")
    ap.add_argument("--auto", action="store_true",
                    help="rebuild the values from CSVs instead of the table above")
    args = ap.parse_args()

    single, combo = dict(SINGLE), dict(COMBO)
    if args.auto:
        if not (args.results and args.clinrate):
            sys.exit("--auto needs --results and --clinrate")
        print("rebuilding values from result CSVs ...")
        got = load_from_csvs(args.results, args.clinrate)
        for k in single:
            if CSV_NAMES[k] in [CSV_NAMES[x] for x in got]:
                single[k] = got[k]
        for k in combo:
            if k in got:
                combo[k] = (combo[k][0], got[k][0], got[k][1])

    print(f"\nbaseline: WER {BASELINE['wer']:.2f}, negation {BASELINE['negation']:.2f}")
    print(f"{len(single)} single-factor effects, {len(combo)} held-out combinations")

    for ix, label in ((0, "WER"), (1, "NEGATION ERROR")):
        additive_test(BASELINE, single, combo, ix, label)
        interaction_fit(BASELINE, single, combo, ix, label)

    print(f"\n{'='*78}")
    print("NOTE ON SCOPE")
    print(f"{'='*78}")
    print("Every combination tested pairs a model change with one parameter.")
    print("Three-factor combinations were not evaluated, so the conclusion")
    print("that the surface is flat applies to two-factor composition and is")
    print("an extrapolation beyond it. State it that way.\n")


if __name__ == "__main__":
    main()
