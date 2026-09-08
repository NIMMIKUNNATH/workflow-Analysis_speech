#!/usr/bin/env python3
"""
make_response_figures.py

Parameter response figures: how each setting changes the model's behaviour.

The existing figures show that WER and clinical outcomes disagree ACROSS
configurations. These show, for each parameter that was swept, HOW the model
responds as that parameter is varied — which is the question "how did the model
react" actually asks.

  Figure R1  Parameter response grid. One panel per swept parameter, WER and
             negation on twin axes, baseline marked. Shows which parameters
             move which metric, and in which direction.
  Figure R2  Model size response. Whisper small / medium / large-v2 / large-v3
             / turbo across four outcomes.
  Figure R3  Effect sizes with confidence intervals, sorted. Which parameters
             matter at all, on WER and on negation, side by side.
  Figure R4  Error composition. Substitutions, deletions and insertions per
             condition, showing that equal WER can hide different behaviour.
  Figure R5  Per-recording response. How consistent each parameter's effect is
             across the 40 recordings, not just on average.

Run from the code directory:  python make_response_figures.py
"""

import glob
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.environ.get("ASR_RESULTS_ROOT", os.path.join(REPO_ROOT, "results"))
CLINICAL_ROOT = os.environ.get(
    "ASR_CLINICAL_RESULTS_ROOT", os.path.join(RESULTS_ROOT, "clinical")
)
CLIN = os.path.join(CLINICAL_ROOT, "clinical_*.csv")
LOG = os.path.join(os.environ.get("ASR_STUDY_ROOT", RESULTS_ROOT), "study_log.csv")

DEEP, WARN, GOOD, MUTE = "#147D92", "#C4513B", "#347859", "#66777F"
INK, GRID = "#1A1A1A", "#E1E9ED"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9,
    "axes.labelsize": 9.5, "axes.titlesize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 300,
})

DUPES = {"largev2", "medium40", "small", "baseline40"}
SKIP = {"272", "full", "errors"} | DUPES

DISPLAY_LABEL = {
    "best_combo": "large-v2 + VAD silence 1000 ms",
    "tpe_trial32": "large-v2 TPE trial 32",
    "tpe_trial35": "large-v3 TPE trial 35",
    "v2_vad_500": "large-v2 + VAD silence 500 ms",
    "v2_beam3": "large-v2 + beam 3",
    "v2_merge150": "large-v2 + merge gap 1.50 s",
    "v2_beam1": "large-v2 + beam 1",
    "v2_vad_off": "large-v2 + VAD off",
    "model_largev2": "model: large-v2",
    "model_medium": "model: medium",
    "model_small": "model: small",
    "model_turbo": "model: large-v3-turbo",
    "cond_prev_off": "previous-text conditioning off",
    "compute_fp16": "compute: float16",
    "no_repeat_3": "no-repeat 3-gram",
}


def display_labels(index):
    return [DISPLAY_LABEL.get(str(value), str(value).replace("_", " ")) for value in index]


# ---------------------------------------------------------------- loading --

def load_clinical():
    rows = []
    for f in glob.glob(CLIN):
        name = os.path.basename(f)[9:-4]
        if name in SKIP or name.startswith("primock"):
            continue
        d = pd.read_csv(f)
        if len(d) < 30 or "negation_rate" not in d:
            continue
        rows.append({
            "condition": name, "n": len(d),
            "wer": d.wer.mean() * 100,
            "neg": d.negation_rate.mean() * 100,
            "med": d.medication_rate.mean() * 100,
            "clin": d.clinical_rate.mean() * 100,
            "critical": d.critical_n.sum() if "critical_n" in d else np.nan,
        })
    return pd.DataFrame(rows).set_index("condition")


def load_log():
    d = pd.read_csv(LOG)
    d["wer"] *= 100
    d["wder"] *= 100
    return d


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}.pdf / .png")


# The parameter sweeps actually run, with the numeric level of each condition.
# baseline is inserted at its own level where the sweep includes the default.
SWEEPS = [
    ("Beam size", "",
     [("beam1", 1), ("beam3", 3), ("baseline", 5), ("beam8", 8)]),
    ("VAD speech threshold", "",
     [("vad_thresh_030", 0.30), ("baseline", 0.50), ("vad_thresh_070", 0.70)]),
    ("VAD minimum silence", "ms",
     [("vad_silence_500", 500), ("vad_silence_1000", 1000), ("baseline", 2000)]),
    ("VAD speech padding", "ms",
     [("vad_pad_200", 200), ("baseline", 400), ("vad_pad_600", 600)]),
    ("Merge gap", "s",
     [("merge_gap_030", 0.30), ("baseline", 1.00),
      ("merge_gap_150", 1.50), ("merge_gap_200", 2.00)]),
    ("Minimum retained interval", "s",
     [("min_interval_00", 0.00), ("baseline", 0.10),
      ("min_interval_25", 0.25), ("min_interval_40", 0.40)]),
    ("No-speech threshold", "",
     [("nospeech_040", 0.40), ("baseline", 0.60),
      ("nospeech_075", 0.75), ("nospeech_090", 0.90)]),
    ("Log-probability threshold", "",
     [("logprob_-15", -1.5), ("baseline", -1.0), ("logprob_-05", -0.5)]),
    ("Compression-ratio threshold", "",
     [("compression_20", 2.0), ("baseline", 2.4), ("compression_30", 3.0)]),
    ("Beam patience", "",
     [("baseline", 1.0), ("patience_150", 1.5), ("patience_200", 2.0)]),
    ("Length penalty", "",
     [("length_pen_080", 0.8), ("baseline", 1.0), ("length_pen_120", 1.2)]),
]


# ---------------------------------------------------------------- FIG R1 ---

def fig_response_grid(g):
    """One panel per swept parameter: WER and negation as the level changes."""
    avail = []
    for title, unit, levels in SWEEPS:
        pts = [(lvl, c) for c, lvl in levels if c in g.index]
        # drop duplicate levels (compression_30 and baseline share 3.0)
        seen, keep = set(), []
        for lvl, c in sorted(pts):
            if lvl in seen:
                continue
            seen.add(lvl)
            keep.append((lvl, c))
        if len(keep) >= 3:
            avail.append((title, unit, keep))

    ncol = 3
    nrow = int(np.ceil(len(avail) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.5 * nrow))
    axes = np.atleast_1d(axes).ravel()

    base_wer = g.loc["baseline", "wer"] if "baseline" in g.index else None
    base_neg = g.loc["baseline", "neg"] if "baseline" in g.index else None

    for ax, (title, unit, pts) in zip(axes, avail):
        x = [p[0] for p in pts]
        wer = [g.loc[p[1], "wer"] for p in pts]
        neg = [g.loc[p[1], "neg"] for p in pts]

        ax.plot(x, wer, "o-", color=DEEP, lw=1.5, ms=5, zorder=3)
        ax.set_ylabel("WER (%)", color=DEEP, fontsize=8.5)
        ax.tick_params(axis="y", labelcolor=DEEP)
        ax.set_title(f"{title}" + (f" ({unit})" if unit else ""),
                     fontsize=9.5, pad=6)

        ax2 = ax.twinx()
        ax2.plot(x, neg, "s--", color=WARN, lw=1.5, ms=4.5, zorder=3)
        ax2.set_ylabel("negation (%)", color=WARN, fontsize=8.5)
        ax2.tick_params(axis="y", labelcolor=WARN)
        ax2.grid(False)
        ax2.spines["top"].set_visible(False)

        if base_wer is not None:
            ax.axhline(base_wer, color=DEEP, ls=":", lw=0.8, alpha=0.5)
            ax2.axhline(base_neg, color=WARN, ls=":", lw=0.8, alpha=0.5)

        # A panel spanning 0.1 pp makes a trivial effect look dramatic.
        # Enforce a floor on both axes so panel height is comparable.
        def floor_span(a, vals, minspan):
            lo, hi = min(vals), max(vals)
            if hi - lo < minspan:
                mid = (hi + lo) / 2
                a.set_ylim(mid - minspan / 2, mid + minspan / 2)

        wvals = list(wer) + ([base_wer] if base_wer is not None else [])
        nvals = list(neg) + ([base_neg] if base_neg is not None else [])
        floor_span(ax, wvals, 3.0)
        floor_span(ax2, nvals, 2.0)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in x], fontsize=8)

    for ax in axes[len(avail):]:
        ax.set_visible(False)

    fig.suptitle("Parameter response: word error rate (blue, left axis) and "
                 "negation error (red, right axis)\n"
                 "Dotted lines mark the large-v3 baseline",
                 y=1.005, fontsize=11)
    fig.tight_layout()
    save(fig, "figR1_parameter_response")


# ---------------------------------------------------------------- FIG R2 ---

def fig_model_response(g):
    """How the four outcomes respond to model size."""
    models = [("model_small", "small"), ("model_medium", "medium"),
              ("model_largev2", "large-v2"), ("baseline", "large-v3"),
              ("model_turbo", "v3-turbo")]
    models = [(c, l) for c, l in models if c in g.index]
    if len(models) < 3:
        print("  (skipped R2: model conditions missing)")
        return

    labels = [l for _, l in models]
    x = np.arange(len(models))
    # Medication is intentionally shown as a raw count over the fixed
    # 99-token denominator. The previous macro-rate panel was visually
    # inconsistent with the pooled counts reported in the tables.
    med_counts = {
        c: int(pd.read_csv(os.path.join(
            CLINICAL_ROOT, f"clinical_{c}.csv"
        )).medication_err.sum())
        for c, _ in models
    }
    metrics = [("wer", "Word error rate (%)", DEEP),
               ("neg", "Negation error (%)", WARN),
               ("med_count", "Medication errors (count/99)", GOOD),
               ("clin", "Clinical term error (%)", MUTE)]

    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
    for ax, (col, lab, colr) in zip(axes, metrics):
        v = ([med_counts[c] for c, _ in models] if col == "med_count"
             else [g.loc[c, col] for c, _ in models])
        bars = ax.bar(x, v, color=colr, width=0.6, zorder=3)
        best = int(np.argmin(v))
        bars[best].set_edgecolor(INK)
        bars[best].set_linewidth(1.4)
        for xi, vi in zip(x, v):
            label = f"{int(vi)}" if col == "med_count" else f"{vi:.1f}"
            ax.text(xi, vi, label, ha="center", va="bottom",
                    fontsize=8, color=MUTE)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_title(lab, fontsize=9.5, pad=6)
        ax.set_ylim(0, max(v) * 1.22)

    fig.suptitle("Model size response. The best model differs by outcome; "
                 "outlined bars mark the minimum.", y=1.04, fontsize=11)
    fig.tight_layout()
    save(fig, "figR2_model_response")


# ---------------------------------------------------------------- FIG R3 ---

def fig_effect_sizes(g):
    """Change from baseline on WER and on negation, sorted, side by side."""
    if "baseline" not in g.index:
        print("  (skipped R3: no baseline)")
        return
    b = g.loc["baseline"]
    d = g.drop(index=[c for c in ("baseline", "canary") if c in g.index]).copy()
    d["dwer"] = d.wer - b.wer
    d["dneg"] = d.neg - b.neg
    d = d.sort_values("dwer")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, max(6, 0.2 * len(d))),
                                 sharey=True)
    y = np.arange(len(d))

    a1.barh(y, d.dwer, color=[GOOD if v < 0 else WARN for v in d.dwer],
            height=0.62, zorder=3)
    a1.axvline(0, color=INK, lw=0.9)
    a1.set_yticks(y)
    a1.set_yticklabels(display_labels(d.index), fontsize=7.5)
    a1.set_xlabel("Change in WER from baseline (pp)")
    a1.set_title("Word error rate", fontsize=10, pad=6)
    a1.invert_yaxis()

    a2.barh(y, d.dneg, color=[GOOD if v < 0 else WARN for v in d.dneg],
            height=0.62, zorder=3)
    a2.axvline(0, color=INK, lw=0.9)
    a2.set_xlabel("Change in negation error from baseline (pp)")
    a2.set_title("Negation error", fontsize=10, pad=6)

    fig.suptitle("Effect of each condition relative to the large-v3 baseline.\n"
                 "Conditions are ordered by WER change; the negation panel "
                 "does not follow the same order.",
                 y=1.005, fontsize=11)
    fig.tight_layout()
    save(fig, "figR3_effect_sizes")


# ---------------------------------------------------------------- FIG R4 ---

def fig_error_composition(log):
    """Substitutions, deletions, insertions per condition."""
    g = log.groupby("condition").agg(
        sub=("sub", "sum"), dele=("del", "sum"), ins=("ins", "sum"),
        ref=("ref_words", "sum"), wer=("wer", "mean"))
    for c in ("sub", "dele", "ins"):
        g[c] = 100 * g[c] / g.ref
    g = g.drop(index=[c for c in DUPES if c in g.index], errors="ignore")
    g = g.sort_values("wer")

    fig, ax = plt.subplots(figsize=(11, max(6, 0.2 * len(g))))
    y = np.arange(len(g))
    ax.barh(y, g["sub"].values, color=DEEP, height=0.62, label="substitutions", zorder=3)
    ax.barh(y, g["dele"].values, left=g["sub"].values, color=WARN, height=0.62,
            label="deletions", zorder=3)
    ax.barh(y, g["ins"].values, left=g["sub"].values + g["dele"].values, color=GOOD, height=0.62,
            label="insertions", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(display_labels(g.index), fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Rate over reference words (%)")
    ax.set_title("Error composition by condition, ordered by total WER.\n"
                 "Conditions with similar WER differ in how that error arises.",
                 fontsize=10.5, pad=8)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.045), ncol=3,
        frameon=False, borderaxespad=0.0
    )
    fig.tight_layout()
    save(fig, "figR4_error_composition")


# ---------------------------------------------------------------- FIG R5 ---

def fig_consistency(log):
    """Per-recording spread of a few key conditions against baseline."""
    keep = ["baseline", "model_largev2", "tpe_trial32", "min_interval_40",
            "vad_silence_1000", "cond_prev_off", "canary"]
    keep = [c for c in keep if c in set(log.condition)]
    if len(keep) < 3:
        print("  (skipped R5: conditions missing)")
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    data = [log[log.condition == c].wer.values for c in keep]
    bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.55,
                    medianprops=dict(color=INK, linewidth=1.3),
                    flierprops=dict(marker="o", markersize=3,
                                    markerfacecolor=MUTE,
                                    markeredgecolor="none", alpha=0.6))
    for patch, c in zip(bp["boxes"], keep):
        patch.set_facecolor(WARN if c == "canary" else DEEP)
        patch.set_alpha(0.65)
        patch.set_edgecolor(INK)
        patch.set_linewidth(0.7)

    for i, c in enumerate(keep, start=1):
        v = log[log.condition == c].wer.values
        ax.scatter(np.random.normal(i, 0.055, len(v)), v, s=7,
                   color=INK, alpha=0.35, zorder=4)

    ax.set_xticklabels(keep, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("Recording-level WER (%)")
    ax.set_title("Per-recording spread. Between-recording variation is large "
                 "relative to\nthe differences between configurations, which "
                 "is why comparisons are paired.",
                 fontsize=10.5, pad=8)
    fig.tight_layout()
    save(fig, "figR5_per_recording_spread")


if __name__ == "__main__":
    g = load_clinical()
    print(f"clinical conditions: {len(g)}")
    try:
        log = load_log()
        print(f"study_log rows: {len(log)}\n")
    except Exception as e:
        log = None
        print(f"study_log unavailable ({e})\n")

    print("writing figures:")
    fig_response_grid(g)
    fig_model_response(g)
    fig_effect_sizes(g)
    if log is not None:
        fig_error_composition(log)
        fig_consistency(log)

    print("""
NOTE ON FIGURE R1
  Some sweeps place the baseline at a level that another condition also
  covers (compression ratio 3.0 is both). Duplicate levels are collapsed,
  keeping the first. Verify the panels against conditions.json before use.

NOTE ON INTERPRETATION
  These are response curves over three or four levels each, on 40 recordings.
  They show direction and rough magnitude, not dose-response relationships in
  any stronger sense. Where a panel looks monotone, check the paired
  confidence interval in study.py --analyse before describing it as a trend.
""")
