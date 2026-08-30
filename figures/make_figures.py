#!/usr/bin/env python3
"""
make_figures.py

Publication figures for the WER-versus-clinical-outcome analysis.

  Figure 1  WER against negation error across configurations, the central
            divergence result
  Figure 2  Rank agreement across three outcomes, showing that WER tracks
            speaker attribution, ignores medication, and inverts negation
  Figure 3  Critical error counts against WER, absolute rather than rate
  Figure 4  Minimum-interval dose response, the single-parameter case

Writes PDF for the manuscript and PNG for slides, at 300 dpi.

Run from the code directory:  python make_figures.py
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.environ.get(
    "ASR_CLINICAL_RESULTS_ROOT", os.path.join(REPO_ROOT, "results", "clinical")
)
CLIN = os.path.join(RESULTS_ROOT, "clinical_*.csv")
OUT = "."

# Duplicate condition files: same data written under two names.
DUPES = {"largev2", "medium40", "small", "baseline40"}
SKIP = {"272", "full", "errors"} | DUPES

DEEP = "#065A82"
TEAL = "#1C7293"
WARN = "#B85042"
GOOD = "#2C5F2D"
MUTE = "#5A6B75"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 300,
})


def load():
    rows = []
    for f in glob.glob(CLIN):
        name = os.path.basename(f)[9:-4]
        if name in SKIP or name.startswith("primock"):
            continue
        d = pd.read_csv(f)
        if len(d) < 30 or "negation_rate" not in d:
            continue
        rows.append({
            "condition": name,
            "n": len(d),
            "wer": d.wer.mean() * 100,
            "neg": d.negation_rate.mean() * 100,
            "med": d.medication_rate.mean() * 100,
            "clin": d.clinical_rate.mean() * 100,
            "critical": d.critical_n.sum() if "critical_n" in d else np.nan,
        })
    return pd.DataFrame(rows).sort_values("wer").reset_index(drop=True)


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}.pdf, {stem}.png")


# --------------------------------------------------------------- FIGURE 1 --

def fig_divergence(g):
    """WER against negation error, the central result."""
    w = g[g.condition != "canary"]          # Canary plotted separately
    c = g[g.condition == "canary"]

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    ax.scatter(w.wer, w.neg, s=34, color=DEEP, alpha=0.75,
               edgecolor="white", linewidth=0.6, zorder=3,
               label="Whisper configurations")

    # least-squares line over the Whisper configurations only
    m, b = np.polyfit(w.wer, w.neg, 1)
    xs = np.linspace(w.wer.min() - 0.3, w.wer.max() + 0.3, 100)
    ax.plot(xs, m * xs + b, color=WARN, linewidth=1.4, zorder=2,
            label=f"fit, slope {m:+.2f}")

    if len(c):
        ax.scatter(c.wer, c.neg, s=60, marker="D", color=WARN,
                   edgecolor="white", linewidth=0.8, zorder=4,
                   label="Canary-1b-v2")
        ax.annotate("Canary-1b-v2", (c.wer.iloc[0], c.neg.iloc[0]),
                    textcoords="offset points", xytext=(-8, -14),
                    ha="right", fontsize=8, color=WARN)

    LABEL = {
        "best_combo": ("lowest WER", (6, 6)),
        "min_interval_40": ("min interval 0.40 s", (8, 2)),
        "vad_silence_1000": ("lowest negation error", (8, -4)),
        "cond_prev_off": ("previous-text off", (-6, 6)),
    }
    for k, (txt, off) in LABEL.items():
        r = g[g.condition == k]
        if not len(r):
            continue
        ax.annotate(txt, (r.wer.iloc[0], r.neg.iloc[0]),
                    textcoords="offset points", xytext=off,
                    ha="left" if off[0] > 0 else "right",
                    fontsize=8, color=MUTE)
        ax.scatter(r.wer, r.neg, s=34, facecolor="none",
                   edgecolor=INK if (INK := "#1A1A1A") else None,
                   linewidth=0.9, zorder=5)

    t, p = kendalltau(w.wer, w.neg)
    rho, _ = spearmanr(w.wer, w.neg)
    ax.text(0.03, 0.96,
            f"Kendall $\\tau$ = {t:+.3f}  (p = {p:.4f})\n"
            f"Spearman $\\rho$ = {rho:+.3f}\nn = {len(w)} configurations",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            color=MUTE,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#F2F6F8",
                      edgecolor="#D8E2E8", linewidth=0.6))

    ax.set_xlabel("Word error rate (%)")
    ax.set_ylabel("Negation error rate (%)")
    ax.set_title("Lower word error rate does not imply lower negation error",
                 pad=10)
    ax.legend(loc="lower right", frameon=False)
    save(fig, "fig1_wer_vs_negation")


# --------------------------------------------------------------- FIGURE 2 --

def fig_rank_agreement(g):
    """Kendall tau between WER and three outcomes."""
    w = g[g.condition != "canary"]

    outcomes = [("WDER", 0.444, 1.7e-5, "from study_log, 45 conditions"),
                ("Medication", kendalltau(w.wer, w.med)[0],
                 kendalltau(w.wer, w.med)[1], f"{len(w)} conditions"),
                ("Negation", kendalltau(w.wer, w.neg)[0],
                 kendalltau(w.wer, w.neg)[1], f"{len(w)} conditions")]

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    y = np.arange(len(outcomes))[::-1]
    taus = [o[1] for o in outcomes]
    cols = [GOOD if t > 0.3 else (MUTE if t > 0 else WARN) for t in taus]

    ax.barh(y, taus, color=cols, height=0.5, zorder=3)
    ax.axvline(0, color="#1A1A1A", linewidth=0.9, zorder=4)

    for yi, (lab, t, p, note) in zip(y, outcomes):
        off = 0.02 if t >= 0 else -0.02
        ax.text(t + off, yi, f"  $\\tau$ = {t:+.3f}, p = {p:.4g}",
                va="center", ha="left" if t >= 0 else "right",
                fontsize=8.5, color=MUTE)

    ax.set_yticks(y)
    ax.set_yticklabels([o[0] for o in outcomes])
    ax.set_xlabel("Kendall $\\tau$ with word error rate")
    ax.set_xlim(-0.65, 0.75)
    ax.set_title("WER tracks speaker attribution, ignores medication,\n"
                 "and inverts negation", pad=10)
    ax.grid(axis="y", visible=False)
    save(fig, "fig2_rank_agreement")


# --------------------------------------------------------------- FIGURE 3 --

def fig_critical(g):
    """Absolute critical error counts against WER."""
    if g.critical.isna().all():
        print("  (skipped fig3: no critical_n column)")
        return
    w = g.dropna(subset=["critical"])
    fig, ax = plt.subplots(figsize=(5.5, 3.6))

    base = w[w.condition == "baseline"]
    ax.scatter(w.wer, w.critical, s=34, color=DEEP, alpha=0.75,
               edgecolor="white", linewidth=0.6, zorder=3)

    if len(base):
        ax.axhline(base.critical.iloc[0], color=MUTE, linestyle="--",
                   linewidth=0.9, zorder=2)
        ax.text(w.wer.min(), base.critical.iloc[0] + 6,
                f"baseline, {int(base.critical.iloc[0])} critical errors",
                fontsize=8, color=MUTE)

    for k, txt in [("min_interval_40", "min interval 0.40 s"),
                   ("canary", "Canary-1b-v2"),
                   ("vad_silence_1000", "VAD silence 1000 ms")]:
        r = w[w.condition == k]
        if not len(r):
            continue
        ax.annotate(txt, (r.wer.iloc[0], r.critical.iloc[0]),
                    textcoords="offset points", xytext=(7, 4),
                    fontsize=8, color=WARN if k != "vad_silence_1000" else GOOD)
        ax.scatter(r.wer, r.critical, s=48, facecolor="none",
                   edgecolor=WARN if k != "vad_silence_1000" else GOOD,
                   linewidth=1.1, zorder=5)

    ax.set_xlabel("Word error rate (%)")
    ax.set_ylabel("Critical clinical errors (count over 40 recordings)")
    ax.set_title("Critical error count is not a function of word error rate",
                 pad=10)
    save(fig, "fig3_critical_errors")


# --------------------------------------------------------------- FIGURE 4 --

def fig_dose_response(g):
    """Minimum retained interval: WER down, negation up."""
    order = ["min_interval_00", "min_interval_25", "min_interval_40"]
    labels = ["0 s", "0.25 s", "0.40 s"]
    sub = [g[g.condition == c] for c in order]
    if any(len(s) == 0 for s in sub):
        print("  (skipped fig4: missing a min_interval condition)")
        return

    wer = [s.wer.iloc[0] for s in sub]
    neg = [s.neg.iloc[0] for s in sub]
    base = g[g.condition == "baseline"]

    fig, ax1 = plt.subplots(figsize=(4.6, 3.4))
    x = np.arange(3)

    ax1.plot(x, wer, "o-", color=DEEP, linewidth=1.6, markersize=6,
             label="WER", zorder=3)
    ax1.set_ylabel("Word error rate (%)", color=DEEP)
    ax1.tick_params(axis="y", labelcolor=DEEP)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Minimum retained interval")

    ax2 = ax1.twinx()
    ax2.plot(x, neg, "s--", color=WARN, linewidth=1.6, markersize=6,
             label="negation error", zorder=3)
    ax2.set_ylabel("Negation error rate (%)", color=WARN)
    ax2.tick_params(axis="y", labelcolor=WARN)
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)

    if len(base):
        ax1.axhline(base.wer.iloc[0], color=DEEP, linestyle=":",
                    linewidth=0.9, alpha=0.6)
        ax2.axhline(base.neg.iloc[0], color=WARN, linestyle=":",
                    linewidth=0.9, alpha=0.6)

    ax1.set_title("One parameter, two metrics,\nopposite directions", pad=10)
    save(fig, "fig4_min_interval_dose")


if __name__ == "__main__":
    g = load()
    print(f"loaded {len(g)} conditions "
          f"(duplicates and PriMock57 files excluded)\n")
    print("writing figures:")
    fig_divergence(g)
    fig_rank_agreement(g)
    fig_critical(g)
    fig_dose_response(g)
    print("""
LaTeX:
  \\begin{figure}[htbp]\\centering
  \\includegraphics[width=0.7\\textwidth]{fig1_wer_vs_negation.pdf}
  \\caption{...}\\label{fig:divergence}\\end{figure}

Check before use: Figure 2 takes the WDER value from the earlier
study_log analysis (tau = 0.444 over 45 conditions) rather than
recomputing it here, because the clinical CSVs do not carry WDER.
If that number changes, update it in fig_rank_agreement().
""")
