#!/usr/bin/env python3
"""
final_surrogate_opt.py - surrogate optimisation over the ASR parameter space,
reporting the surrogate's own prediction error alongside its proposal.

WHY THIS EXISTS

    The natural next step after a one-factor-at-a-time search is Bayesian
    optimisation: fit a Gaussian-process surrogate to the evaluated
    configurations, then propose the configuration minimising the predicted
    objective. Optuna, Ray Tune and MATLAB's surrogateopt all do this.

    Before spending GPU time on further trials it is worth asking whether the
    surrogate can see anything. This script fits the surrogate, cross-validates
    it, searches the full parameter grid, and reports THREE numbers together:

        - the predicted improvement over the best observed configuration
        - the surrogate's leave-one-out prediction error
        - how much of the search space is within one prediction error of the
          proposed optimum

    A proposal whose predicted gain is smaller than the model's own error is
    not actionable, and a search space where most candidates are
    indistinguishable from the optimum is not worth searching. Reporting the
    proposal WITHOUT these numbers - which is the default in most tuning
    libraries - would present noise as a recommendation.

WHAT IT IS NOT

    Not an acquisition-function optimiser. It does not propose a next trial to
    run; it evaluates whether running one is justified. If the answer is yes,
    use Optuna or BoTorch with expected improvement.

USAGE

    python final_surrogate_opt.py
    python final_surrogate_opt.py --target negation --model v2
    python final_surrogate_opt.py --csv my_conditions.csv --target wer
"""
import argparse
import itertools
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (ConstantKernel, Matern,
                                                  WhiteKernel)
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import LeaveOneOut
except ImportError:
    sys.exit("pip install scikit-learn")

# ---------------------------------------------------------------------------
# Evaluated configurations. Columns:
#   model, beam, vad_silence_ms, merge_gap_s, min_interval_s,
#   vad_threshold, vad_pad_ms, vad_filter, compression_ratio, WER, negation
# Replace with --csv to use your own.
# ---------------------------------------------------------------------------
DATA = {
 "baseline":         ("v3", 5,  250, 1.00, 0.10, 0.50, 400, 1, 2.4, 18.54,  5.13),
 "model_small":      ("sm", 5,  250, 1.00, 0.10, 0.50, 400, 1, 2.4, 14.69,  8.58),
 "model_medium":     ("md", 5,  250, 1.00, 0.10, 0.50, 400, 1, 2.4, 14.51,  8.85),
 "model_largev2":    ("v2", 5,  250, 1.00, 0.10, 0.50, 400, 1, 2.4, 14.26,  5.87),
 "model_turbo":      ("tb", 5,  250, 1.00, 0.10, 0.50, 400, 1, 2.4, 17.52,  4.95),
 "beam1":            ("v3", 1,  250, 1.00, 0.10, 0.50, 400, 1, 2.4, 15.94,  5.61),
 "beam3":            ("v3", 3,  250, 1.00, 0.10, 0.50, 400, 1, 2.4, 16.95,  5.12),
 "beam8":            ("v3", 8,  250, 1.00, 0.10, 0.50, 400, 1, 2.4, 18.78,  5.72),
 "vad_silence_500":  ("v3", 5,  500, 1.00, 0.10, 0.50, 400, 1, 2.4, 17.35,  5.73),
 "vad_silence_1000": ("v3", 5, 1000, 1.00, 0.10, 0.50, 400, 1, 2.4, 17.25,  4.33),
 "vad_thresh_030":   ("v3", 5,  250, 1.00, 0.10, 0.30, 400, 1, 2.4, 18.02,  4.74),
 "vad_thresh_070":   ("v3", 5,  250, 1.00, 0.10, 0.70, 400, 1, 2.4, 18.15,  7.91),
 "vad_pad_200":      ("v3", 5,  250, 1.00, 0.10, 0.50, 200, 1, 2.4, 18.06,  5.83),
 "vad_pad_600":      ("v3", 5,  250, 1.00, 0.10, 0.50, 600, 1, 2.4, 17.71,  5.32),
 "vad_off":          ("v3", 5,  250, 1.00, 0.10, 0.50, 400, 0, 2.4, 18.10,  4.43),
 "merge_gap_030":    ("v3", 5,  250, 0.30, 0.10, 0.50, 400, 1, 2.4, 19.95,  5.12),
 "merge_gap_150":    ("v3", 5,  250, 1.50, 0.10, 0.50, 400, 1, 2.4, 16.97,  5.45),
 "merge_gap_200":    ("v3", 5,  250, 2.00, 0.10, 0.50, 400, 1, 2.4, 17.60,  5.05),
 "min_interval_00":  ("v3", 5,  250, 1.00, 0.00, 0.50, 400, 1, 2.4, 17.57,  5.00),
 "min_interval_25":  ("v3", 5,  250, 1.00, 0.25, 0.50, 400, 1, 2.4, 17.56,  5.67),
 "min_interval_40":  ("v3", 5,  250, 1.00, 0.40, 0.50, 400, 1, 2.4, 17.29, 10.14),
 "compression_20":   ("v3", 5,  250, 1.00, 0.10, 0.50, 400, 1, 2.0, 17.72,  4.51),
 "compression_30":   ("v3", 5,  250, 1.00, 0.10, 0.50, 400, 1, 3.0, 17.83,  5.05),
 "v2_beam1":         ("v2", 1,  250, 1.00, 0.10, 0.50, 400, 1, 2.4, 13.87,  6.78),
 "v2_beam3":         ("v2", 3,  250, 1.00, 0.10, 0.50, 400, 1, 2.4, 13.64,  5.88),
 "v2_vad_500":       ("v2", 5,  500, 1.00, 0.10, 0.50, 400, 1, 2.4, 13.62,  6.36),
 "best_combo":       ("v2", 5, 1000, 1.00, 0.10, 0.50, 400, 1, 2.4, 13.36,  6.24),
 "v2_vad_off":       ("v2", 5,  250, 1.00, 0.10, 0.50, 400, 0, 2.4, 14.01,  5.56),
 "v2_merge150":      ("v2", 5,  250, 1.50, 0.10, 0.50, 400, 1, 2.4, 13.82,  5.83),
}

MODELS = ["v3", "v2", "md", "sm", "tb"]     # v3 is the encoding reference
GRID = {
    "beam":        [1, 3, 5, 8],
    "vad_silence": [250, 500, 1000],
    "merge_gap":   [0.30, 1.00, 1.50, 2.00],
    "min_interval":[0.00, 0.10, 0.25, 0.40],
    "vad_thresh":  [0.30, 0.50, 0.70],
    "vad_pad":     [200, 400, 600],
    "vad_filter":  [0, 1],
    "compression": [2.0, 2.4, 3.0],
}
# Excluded from the grid on evidence, not preference. Temperature and the
# quality-fallback thresholds are omitted entirely: temperature 0 halved the
# transcript on two recordings reproducibly, and with all three thresholds
# disabled the output was byte-identical because the fallback can never fire.
# A sampler exploring that region would wander through a dead subspace.


def encode(model, beam, vad_sil, merge, min_int, vad_thr, vad_pad, vad_on, comp):
    """One-hot the model, log-scale the silence duration, leave the rest linear."""
    return ([1.0 if model == m else 0.0 for m in MODELS[1:]] +
            [beam, np.log(vad_sil), merge, min_int, vad_thr, vad_pad / 100.0,
             vad_on, comp])


def loo_rmse(make_model, X, y):
    pred = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        m = make_model()
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def make_gp(d):
    k = (ConstantKernel(1.0) * Matern(length_scale=np.ones(d), nu=2.5)
         + WhiteKernel(0.1))
    return GaussianProcessRegressor(kernel=k, normalize_y=True, alpha=1e-6,
                                    n_restarts_optimizer=3, random_state=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="wer", choices=["wer", "negation"])
    ap.add_argument("--model", default="v2", choices=MODELS,
                    help="hold the model fixed and search the parameters")
    ap.add_argument("--csv", help="override the built-in table")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    data = dict(DATA)
    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        need = ["name", "model", "beam", "vad_silence", "merge_gap",
                "min_interval", "vad_thresh", "vad_pad", "vad_filter",
                "compression", "wer", "negation"]
        miss = [c for c in need if c not in df.columns]
        if miss:
            sys.exit(f"{args.csv} missing columns: {miss}")
        data = {r["name"]: tuple(r[c] for c in need[1:]) for _, r in df.iterrows()}

    names = list(data)
    X = np.array([encode(*data[k][:9]) for k in names])
    ti = 9 if args.target == "wer" else 10
    y = np.array([data[k][ti] for k in names], dtype=float)

    print(f"\n{'='*84}")
    print(f"SURROGATE OPTIMISATION  ·  target: {args.target.upper()}")
    print(f"{'='*84}")
    print(f"{len(names)} evaluated configurations, {X.shape[1]} encoded parameters")
    print(f"observed range: {y.min():.2f} to {y.max():.2f}  "
          f"({y.max()-y.min():.2f} points)")

    # ---- can the surrogate predict at all? ------------------------------
    print(f"\n{'-'*84}")
    print("STEP 1  ·  CAN THE SURROGATE PREDICT?  leave-one-out cross-validation")
    print(f"{'-'*84}")
    r_null = float(y.std())
    r_ridge = loo_rmse(lambda: Ridge(alpha=1.0), X, y)
    r_gp = loo_rmse(lambda: make_gp(X.shape[1]), X, y)
    span = y.max() - y.min()
    print(f"   null model (predict the mean)   RMSE {r_null:.3f}")
    print(f"   ridge regression                RMSE {r_ridge:.3f}"
          f"   ({r_ridge/span*100:.0f}% of range)")
    print(f"   Gaussian process (Matern 5/2)   RMSE {r_gp:.3f}"
          f"   ({r_gp/span*100:.0f}% of range)")
    if r_gp >= r_null:
        print("\n   The surrogate is NO BETTER than predicting the mean.")
        print("   It has learned nothing from the parameters. Any proposal it")
        print("   makes is an artefact of the kernel, not of the data.")
    else:
        print(f"\n   The surrogate beats the null model by "
              f"{(1-r_gp/r_null)*100:.0f}%, so it has learned something.")

    # ---- search the grid -------------------------------------------------
    gp = make_gp(X.shape[1]).fit(X, y)
    combos = list(itertools.product(
        [args.model], GRID["beam"], GRID["vad_silence"], GRID["merge_gap"],
        GRID["min_interval"], GRID["vad_thresh"], GRID["vad_pad"],
        GRID["vad_filter"], GRID["compression"]))
    G = np.array([encode(*c) for c in combos])
    mu, sd = gp.predict(G, return_std=True)
    order = np.argsort(mu)

    print(f"\n{'-'*84}")
    print(f"STEP 2  ·  PROPOSED OPTIMA over {len(combos):,} candidates "
          f"(model fixed at {args.model})")
    print(f"{'-'*84}")
    hdr = (f"{'rank':>4}  {'beam':>4} {'vadsil':>7} {'merge':>6} {'minint':>7} "
           f"{'vthr':>5} {'pad':>4} {'vad':>4} {'comp':>5}   "
           f"{'predicted':>10} {'+/-1sd':>8}")
    print(hdr); print("-" * len(hdr))
    for r, i in enumerate(order[:args.top], 1):
        c = combos[i]
        print(f"{r:>4}  {c[1]:>4} {c[2]:>7} {c[3]:>6.2f} {c[4]:>7.2f} "
              f"{c[5]:>5.2f} {c[6]:>4} {c[7]:>4} {c[8]:>5.1f}   "
              f"{mu[i]:>10.2f} {sd[i]:>8.2f}")

    # ---- is the proposal actionable? ------------------------------------
    best = order[0]
    obs_best = y.min()
    gain = obs_best - mu[best]
    within = int((mu <= mu[best] + r_gp).sum())

    print(f"\n{'-'*84}")
    print("STEP 3  ·  IS THE PROPOSAL ACTIONABLE?")
    print(f"{'-'*84}")
    print(f"   predicted value at the proposed optimum   {mu[best]:.2f} "
          f"+/- {sd[best]:.2f}")
    print(f"   best OBSERVED configuration               {obs_best:.2f}")
    print(f"   predicted improvement                     {gain:+.2f} points")
    print(f"   surrogate cross-validated error           {r_gp:.3f} points")
    print(f"   spread of predictions across the grid     "
          f"{mu.max()-mu.min():.2f} points")
    print(f"   candidates within one CV-error of the optimum   "
          f"{within:,} of {len(combos):,} ({within/len(combos)*100:.0f}%)")

    print(f"\n   VERDICT")
    if r_gp >= r_null:
        print("   The surrogate has no predictive power on this target.")
        print("   Do not run the proposed configuration on its recommendation.")
    elif gain < r_gp:
        print(f"   The predicted gain ({gain:+.2f}) is SMALLER than the surrogate's")
        print(f"   own prediction error ({r_gp:.2f}). The proposal is not")
        print(f"   distinguishable from the best configuration already evaluated.")
        print(f"   Further search over these parameters is not justified.")
    elif within / len(combos) > 0.25:
        print(f"   The predicted gain exceeds the prediction error, but "
              f"{within/len(combos)*100:.0f}% of")
        print(f"   the space is within one error of the optimum. The surface is")
        print(f"   too flat to locate a unique optimum; expect any of a large")
        print(f"   set of configurations to perform equivalently.")
    else:
        print(f"   The predicted gain ({gain:+.2f}) exceeds the prediction error")
        print(f"   ({r_gp:.2f}), and the optimum is reasonably isolated. Running")
        print(f"   the proposed configuration is justified. Confirm with a paired")
        print(f"   comparison against the current best before adopting it.")

    print(f"\n{'-'*84}")
    print("NOTE ON THE SEARCH SPACE")
    print(f"{'-'*84}")
    print("Temperature and the quality-fallback thresholds are excluded from")
    print("the grid on evidence: temperature 0 halved the transcript on two")
    print("recordings reproducibly, and with all three thresholds disabled the")
    print("output was byte-identical because the fallback can never fire. A")
    print("sampler exploring that region would spend trials in a dead subspace.")
    print("Any tuning library pointed at this problem should exclude them too.\n")


if __name__ == "__main__":
    main()
