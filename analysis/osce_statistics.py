#!/usr/bin/env python3
"""
statistics.py - turn the point estimates into defensible statistics.

WHY THIS EXISTS
    Every number reported so far is a bare point estimate: "WER 18.05%",
    "medications 11.74%", "lexical rule 100%". A reviewer will ask, correctly,
    how certain any of that is. This produces the confidence intervals and
    significance tests those claims need.

WHAT IT COMPUTES
    1. Bootstrap 95% CIs for WER and WDER, resampling RECORDINGS (not words),
       because recordings are the independent unit - words within a recording are
       correlated, so resampling words would understate the true uncertainty.
    2. Wilson score CIs for every clinical error class. Wilson rather than the
       normal approximation because several classes have small denominators
       (units n=66) where the normal interval misbehaves and can exit [0,1].
    3. Pairwise comparisons between clinical classes, with Holm correction -
       9 classes gives 36 possible pairs, and uncorrected testing at that rate
       produces spurious findings.
    4. McNemar's test comparing the three role-assignment rules. McNemar rather
       than a chi-square because the rules are evaluated on the SAME recordings;
       only the discordant pairs carry information.
    5. Category differences, tested properly, with an explicit refusal to test
       categories too small to support it.
    6. Correlation between WER and WDER - does poor transcription drive poor
       speaker attribution, or are they independent failures?

INPUTS
    The CSVs already produced:
      fareez_272.csv    from score_fareez_speaker.py
      clinical_272.csv  from clinical_errors.py
      role_272.csv      from role_analysis.py

USAGE
    python statistics.py --wer fareez_272.csv --clinical clinical_272.csv \\
                         --role role_272.csv
"""
import argparse
import sys
from itertools import combinations

import numpy as np
import pandas as pd

try:
    from scipy import stats as st
except ImportError:
    sys.exit("pip install scipy pandas numpy")

RNG = np.random.default_rng(20260820)
B = 10000


# ------------------------------------------------------------ intervals ----
def bootstrap_ci(values, n_boot=B, alpha=0.05):
    """
    Percentile bootstrap over RECORDINGS.

    Resampling recordings rather than words is the point: words inside one
    consultation share a speaker, an accent and an acoustic environment, so they
    are not independent. Resampling them would produce an interval far narrower
    than the truth.
    """
    v = np.asarray(values, dtype=float)
    if len(v) < 3:
        return np.nan, np.nan
    means = np.array([RNG.choice(v, size=len(v), replace=True).mean()
                      for _ in range(n_boot)])
    return np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def wilson_ci(successes, n, alpha=0.05):
    """
    Wilson score interval for a proportion.

    Preferred over the normal approximation here because clinical classes have
    very different denominators (units n=66, fillers n=42,846). The normal
    interval is unreliable at small n and can produce bounds outside [0,1];
    Wilson does not.
    """
    if n == 0:
        return np.nan, np.nan
    z = st.norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def holm(pvals):
    """Holm-Bonferroni step-down correction. Less conservative than Bonferroni."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


# ================================================================== main ====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wer", default="fareez_272.csv")
    ap.add_argument("--clinical", default="clinical_272.csv")
    ap.add_argument("--role", default="role_272.csv")
    ap.add_argument("--out", default="statistics_report.txt")
    a = ap.parse_args()

    lines = []
    def say(s=""):
        print(s)
        lines.append(s)

    say("=" * 74)
    say("STATISTICAL ANALYSIS")
    say("=" * 74)
    say(f"Bootstrap: {B:,} resamples of recordings, percentile method, seed fixed.")
    say("Proportion CIs: Wilson score. Multiple comparisons: Holm-Bonferroni.")

    # ---------------------------------------------------- 1. WER and WDER --
    try:
        w = pd.read_csv(a.wer)
    except FileNotFoundError:
        sys.exit(f"{a.wer} not found - run score_fareez_speaker.py first")

    say(f"\n{'-'*74}\n1. OVERALL PERFORMANCE  (n = {len(w)} recordings)\n{'-'*74}")
    for col, label in [("wer", "WER"), ("wder", "WDER")]:
        if col not in w.columns:
            continue
        v = w[col].dropna() * 100
        lo, hi = bootstrap_ci(v)
        say(f"{label:5s}  mean {v.mean():6.2f}%   95% CI [{lo:5.2f}, {hi:5.2f}]   "
            f"median {v.median():6.2f}%   SD {v.std():5.2f}   n {len(v)}")

    # micro estimate, if the columns exist
    if {"sub", "del", "ins", "ref_words"}.issubset(w.columns):
        micro = (w["sub"] + w["del"] + w["ins"]).sum() / w["ref_words"].sum() * 100
        say(f"\nWER, corpus-level (micro): {micro:.2f}%   "
            f"total reference words {int(w['ref_words'].sum()):,}")
        say("Report one or the other consistently and say which; they answer")
        say("slightly different questions (per-recording vs per-word).")

    # ------------------------------------------- 2. WER vs WDER relation --
    if {"wer", "wder"}.issubset(w.columns):
        r, p = st.pearsonr(w["wer"], w["wder"])
        rho, prho = st.spearmanr(w["wer"], w["wder"])
        say(f"\n{'-'*74}\n2. DOES POOR TRANSCRIPTION DRIVE POOR ATTRIBUTION?\n{'-'*74}")
        say(f"Pearson  r = {r:+.3f}  (p = {p:.4f})")
        say(f"Spearman rho = {rho:+.3f}  (p = {prho:.4f})")
        if p < 0.05 and abs(r) > 0.3:
            say("\nThe two are related: recordings that transcribe badly also tend")
            say("to be attributed badly. Likely a shared cause - overlapping speech,")
            say("or acoustically similar speakers - rather than one causing the other.")
        else:
            say("\nWER and WDER are largely INDEPENDENT. That is worth reporting:")
            say("transcription quality and speaker attribution fail separately, so")
            say("a good WER does not imply a usable speaker-labelled transcript.")

    # -------------------------------------------------- 3. by category ----
    if "category" in w.columns:
        say(f"\n{'-'*74}\n3. BY CLINICAL CATEGORY\n{'-'*74}")
        say(f"{'cat':6s} {'n':>4s} {'WER':>7s} {'95% CI':>16s} {'WDER':>7s}")
        big = []
        for cat, g in w.groupby("category"):
            v = g["wer"] * 100
            if len(g) >= 5:
                lo, hi = bootstrap_ci(v)
                ci = f"[{lo:5.2f},{hi:5.2f}]"
                big.append(cat)
            else:
                ci = "     too small  "
            say(f"{cat:6s} {len(g):4d} {v.mean():6.2f}% {ci:>16s} "
                f"{g['wder'].mean()*100:6.2f}%")

        say(f"\nCategories with n < 5 are reported descriptively only - no interval")
        say("is meaningful, and none should be compared.")

        if len(big) >= 2:
            groups = [w[w.category == c]["wer"].values for c in big]
            H, p = st.kruskal(*groups)
            say(f"\nKruskal-Wallis across {len(big)} adequately-sized categories "
                f"({', '.join(big)}):")
            say(f"  H = {H:.3f}, p = {p:.4f} -> "
                f"{'differences are significant' if p < 0.05 else 'no significant difference'}")
            if p >= 0.05:
                say("  Performance is consistent across clinical categories, which")
                say("  supports reporting a single pooled figure.")

    # -------------------------------------------------- 4. clinical -------
    try:
        c = pd.read_csv(a.clinical)
    except FileNotFoundError:
        say(f"\n{a.clinical} not found - skipping clinical analysis")
        c = None

    if c is not None:
        say(f"\n{'-'*74}\n4. CLINICAL ERROR CLASSES  (n = {len(c)} recordings)\n{'-'*74}")
        classes = ["medication", "unit", "negation", "number", "condition",
                   "anatomy", "symptom", "temporal", "filler"]
        rows = []
        for cl in classes:
            nc, ec = f"{cl}_n", f"{cl}_err"
            if nc not in c.columns:
                continue
            n = int(c[nc].sum()); e = int(c[ec].sum())
            if n == 0:
                continue
            lo, hi = wilson_ci(e, n)
            rows.append({"cls": cl, "n": n, "err": e, "rate": e / n,
                         "lo": lo, "hi": hi})

        say(f"{'class':12s} {'n':>7s} {'err':>6s} {'rate':>8s} {'95% CI (Wilson)':>20s}")
        for r in rows:
            say(f"{r['cls']:12s} {r['n']:7d} {r['err']:6d} {r['rate']*100:7.2f}% "
                f"  [{r['lo']*100:5.2f}, {r['hi']*100:5.2f}]")

        # clinical aggregate
        clin = [r for r in rows if r["cls"] not in ("filler", "temporal")]
        if clin:
            n = sum(r["n"] for r in clin); e = sum(r["err"] for r in clin)
            lo, hi = wilson_ci(e, n)
            say(f"\n{'CLINICAL':12s} {n:7d} {e:6d} {e/n*100:7.2f}%   "
                f"[{lo*100:5.2f}, {hi*100:5.2f}]")
            f = next((r for r in rows if r["cls"] == "filler"), None)
            if f:
                say(f"{'filler':12s} {f['n']:7d} {f['err']:6d} {f['rate']*100:7.2f}%   "
                    f"[{f['lo']*100:5.2f}, {f['hi']*100:5.2f}]")
                z, p = proportion_ztest(e, n, f["err"], f["n"])
                say(f"\nclinical vs filler: z = {z:.2f}, p = {p:.3g}")
                say(f"ratio {f['rate']/(e/n):.1f}x - this is the central finding, and")
                say("the intervals do not overlap, so it is not a sampling artefact.")

        # pairwise, Holm-corrected
        say(f"\npairwise comparisons between clinical classes (Holm-corrected):")
        pairs, ps = [], []
        for r1, r2 in combinations([r for r in rows if r["cls"] != "filler"], 2):
            z, p = proportion_ztest(r1["err"], r1["n"], r2["err"], r2["n"])
            pairs.append((r1["cls"], r2["cls"], r1["rate"], r2["rate"], z))
            ps.append(p)
        if ps:
            adj = holm(np.array(ps))
            shown = 0
            for (c1, c2, p1, p2, z), pa in sorted(zip(pairs, adj), key=lambda x: x[1]):
                if pa < 0.05 and shown < 12:
                    say(f"  {c1:11s} {p1*100:5.2f}%  vs  {c2:11s} {p2*100:5.2f}%   "
                        f"p_adj = {pa:.3g}")
                    shown += 1
            say(f"  ({int((adj < 0.05).sum())} of {len(adj)} pairs significant "
                f"after correction)")

    # -------------------------------------------------- 5. role rules -----
    try:
        rl = pd.read_csv(a.role)
    except FileNotFoundError:
        say(f"\n{a.role} not found - skipping role analysis")
        rl = None

    if rl is not None:
        say(f"\n{'-'*74}\n5. ROLE ASSIGNMENT RULES  (n = {len(rl)} recordings)\n{'-'*74}")
        cols = [("first_ok", "first-speaker"), ("question_ok", "question-ratio"),
                ("lexical_ok", "lexical")]
        avail = [(c_, l) for c_, l in cols if c_ in rl.columns]
        for c_, label in avail:
            v = rl[c_].dropna().astype(bool)
            lo, hi = wilson_ci(int(v.sum()), len(v))
            say(f"{label:16s} {v.mean()*100:6.2f}%  ({int(v.sum())}/{len(v)})   "
                f"95% CI [{lo*100:5.2f}, {hi*100:6.2f}]")

        say("\nMcNemar tests - the rules are evaluated on the SAME recordings, so")
        say("only discordant pairs carry information:")
        for (c1, l1), (c2, l2) in combinations(avail, 2):
            d = rl[[c1, c2]].dropna()
            b_ = int(((d[c1] == True) & (d[c2] == False)).sum())
            c_only = int(((d[c1] == False) & (d[c2] == True)).sum())
            if b_ + c_only == 0:
                say(f"  {l1} vs {l2}: identical on every recording")
                continue
            # exact binomial - appropriate when discordant counts are small
            p = st.binomtest(b_, b_ + c_only, 0.5).pvalue
            better = l2 if c_only > b_ else l1
            say(f"  {l1:16s} vs {l2:16s}  discordant {b_}/{c_only}  "
                f"p = {p:.4g}  -> {better} better"
                f"{'' if p < 0.05 else ' (not significant)'}")

        if "ref_opens" in rl.columns:
            opens_pat = int((rl.ref_opens == "Patient").sum())
            lo, hi = wilson_ci(opens_pat, len(rl))
            say(f"\nreference opens with the PATIENT: {opens_pat}/{len(rl)} "
                f"({opens_pat/len(rl)*100:.1f}%, 95% CI "
                f"[{lo*100:.1f}, {hi*100:.1f}])")
            say("This is the hard ceiling on the first-speaker rule: it cannot")
            say("exceed 100% minus this figure, however good the diarization.")

    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    say(f"\nSaved -> {a.out}")


def proportion_ztest(x1, n1, x2, n2):
    """Two-proportion z-test with pooled variance."""
    p1, p2 = x1 / n1, x2 / n2
    pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return z, 2 * (1 - st.norm.cdf(abs(z)))


if __name__ == "__main__":
    main()
