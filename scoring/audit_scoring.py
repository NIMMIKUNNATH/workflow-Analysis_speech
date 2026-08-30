#!/usr/bin/env python3
"""
audit_scoring.py - check the benchmark numbers before they go into a paper.

Three issues were found by inspecting score_fareez_speaker.py. This script
measures how much each one actually matters on YOUR data, rather than leaving
them as theoretical concerns.

ISSUE 1 - MICRO vs MACRO AVERAGING
    The reported "WER mean 18.02%" is the mean of per-recording WERs (macro).
    The ASR convention is usually corpus-level (micro): total errors divided by
    total reference words. They differ whenever recordings vary in length, and
    yours range from ~780 to ~2200 reference words.

    Macro weights a 780-word recording the same as a 2200-word one. Micro weights
    by content. Neither is wrong, but a reviewer will ask which you used, and the
    two numbers must not be mixed between tables.

ISSUE 2 - NO TEXT NORMALISATION IN THE WER PATH
    clinical_errors.py treats "8"/"eight" and "nope"/"no" as formatting, not
    errors - that fix removed 198 spurious errors from 62 recordings. But
    score_fareez_speaker.py has no such handling, so those same differences are
    still counted as substitutions in the headline WER.

    The ASR literature reports that text normalisation moves WER by 1-5 absolute
    points. Whisper ships an EnglishTextNormalizer precisely for this. This
    quantifies the effect on your corpus so you can report a normalised figure or
    justify the raw one.

ISSUE 3 - TWO RECORDINGS FAILED
    RES0002 and RES0054 returned "empty after parsing". A paper must say why 270
    of 272 were analysed, not leave it unexplained.

ALSO CHECKED
    - a latent bug: if the pipeline finds only ONE speaker cluster, WDER is
      reported as 100% rather than the true best-mapping value. Confirms whether
      any recording hit this.
    - that the scorer's WER agrees with a direct jiwer call (no silent drift).

USAGE
    python audit_scoring.py \\
        --result-dir ${ASR_CACHE_ROOT} \\
        --ref-dir "${ASR_DATA_ROOT}/Clean Transcripts"
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

try:
    import jiwer
except ImportError:
    sys.exit("pip install jiwer pandas openpyxl")

NUM_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
}
NEG_EQUIV = {"nope": "no", "nah": "no", "negative": "no"}


def norm(s):
    s = re.sub(r"[^a-z0-9' ]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def normalise_tokens(words):
    """Digit/word and negation-variant folding, matching clinical_errors.py."""
    out = []
    for w in words:
        w = NUM_MAP.get(w, w)
        w = NEG_EQUIV.get(w, w)
        out.append(w)
    return out


def parse_reference(path):
    pairs, cur = [], None
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^\s*([DP])\s*:\s*(.*)$", line, flags=re.I)
        if m:
            cur, text = m.group(1).upper(), m.group(2)
        else:
            if cur is None:
                continue
            text = line
        pairs += [(w, cur) for w in norm(text).split()]
    return pairs


def parse_result(path):
    df = pd.read_excel(path)
    col = next((c for c in df.columns if "transcript" in c.lower()
                and "researcher" not in c.lower()), None)
    if col is None or "Speaker" not in df.columns:
        return [], "no transcript or Speaker column"
    pairs = []
    for _, r in df.iterrows():
        if pd.isna(r["Speaker"]) or pd.isna(r[col]):
            continue
        spk = str(r["Speaker"]).strip()
        pairs += [(w, spk) for w in norm(r[col]).split()]
    return pairs, f"{len(df)} rows"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--csv", default="audit.csv")
    a = ap.parse_args()

    rows, failures = [], []
    for xlsx in sorted(Path(a.result_dir).glob("*_large_v3_review.xlsx")):
        case = xlsx.stem.split("_large_v3")[0]
        ref_p = Path(a.ref_dir) / f"{case}.txt"
        if not ref_p.exists():
            failures.append((case, "no transcript file"))
            continue

        ref_pairs = parse_reference(ref_p)
        hyp_pairs, note = parse_result(xlsx)

        if not ref_pairs:
            failures.append((case, f"reference parsed to 0 words ({ref_p.stat().st_size} bytes)"))
            continue
        if not hyp_pairs:
            failures.append((case, f"result parsed to 0 words ({note})"))
            continue

        rw = [w for w, _ in ref_pairs]
        hw = [w for w, _ in hyp_pairs]

        raw = jiwer.process_words(" ".join(rw), " ".join(hw))
        nrm = jiwer.process_words(" ".join(normalise_tokens(rw)),
                                  " ".join(normalise_tokens(hw)))

        # speaker clusters actually produced
        n_clusters = len({s for _, s in hyp_pairs})

        rows.append({
            "case": case,
            "category": re.match(r"^([A-Z]+)", case).group(1),
            "ref_words": len(rw),
            "hyp_words": len(hw),
            "wer_raw": raw.wer,
            "wer_norm": nrm.wer,
            "errors_raw": raw.substitutions + raw.deletions + raw.insertions,
            "errors_norm": nrm.substitutions + nrm.deletions + nrm.insertions,
            "sub": raw.substitutions, "del": raw.deletions, "ins": raw.insertions,
            "n_clusters": n_clusters,
        })

    if not rows:
        sys.exit("nothing to audit")
    df = pd.DataFrame(rows)
    df.to_csv(a.csv, index=False)

    print(f"\n{'='*72}\nSCORING AUDIT - {len(df)} recordings analysed\n{'='*72}")

    # ---- issue 1: micro vs macro -----------------------------------------
    macro = df.wer_raw.mean() * 100
    micro = df.errors_raw.sum() / df.ref_words.sum() * 100
    print("\n1. AVERAGING METHOD")
    print(f"   macro (mean of per-recording WER) : {macro:6.2f}%   <- currently reported")
    print(f"   micro (total errors / total words): {micro:6.2f}%")
    print(f"   difference                        : {abs(macro-micro):6.2f} points")
    if abs(macro - micro) < 0.5:
        print("   The two agree closely, so the choice does not change any claim.")
        print("   State which was used and move on.")
    else:
        print("   These differ enough to matter. Report the micro (corpus-level)")
        print("   figure as primary - it is the ASR convention - and state it.")

    # ---- issue 2: normalisation ------------------------------------------
    nmacro = df.wer_norm.mean() * 100
    nmicro = df.errors_norm.sum() / df.ref_words.sum() * 100
    saved = int(df.errors_raw.sum() - df.errors_norm.sum())
    print("\n2. TEXT NORMALISATION (digit/word and negation variants)")
    print(f"   raw        macro {macro:6.2f}%   micro {micro:6.2f}%")
    print(f"   normalised macro {nmacro:6.2f}%   micro {nmicro:6.2f}%")
    print(f"   spurious errors removed: {saved}")
    print(f"   effect on WER: {macro-nmacro:.2f} points (macro), "
          f"{micro-nmicro:.2f} points (micro)")
    if macro - nmacro > 0.3:
        print("   Worth reporting the normalised figure, or at minimum stating")
        print("   that the raw figure includes formatting differences.")
    else:
        print("   Small effect. The raw figure is defensible as reported.")

    # ---- issue 3: failures -----------------------------------------------
    print("\n3. RECORDINGS NOT ANALYSED")
    if failures:
        for case, why in failures:
            print(f"   {case:10s} {why}")
        print("\n   Each of these needs a one-line explanation in the paper.")
    else:
        print("   none")

    # ---- latent bug check -------------------------------------------------
    one = df[df.n_clusters < 2]
    print("\n4. SINGLE-CLUSTER RECORDINGS (would inflate WDER to 100%)")
    if len(one):
        print(f"   {len(one)} affected: {', '.join(one.case)}")
        print("   Their WDER values are wrong and must be recomputed or excluded.")
    else:
        print("   none - the latent bug never fired, WDER values are sound")

    # ---- distribution sanity ---------------------------------------------
    print("\n5. DISTRIBUTION")
    print(f"   WER      median {df.wer_raw.median()*100:5.2f}%  "
          f"IQR {df.wer_raw.quantile(.25)*100:.2f}-{df.wer_raw.quantile(.75)*100:.2f}%  "
          f"SD {df.wer_raw.std()*100:.2f}")
    print(f"   ref words: total {int(df.ref_words.sum()):,}  "
          f"min {int(df.ref_words.min())}  max {int(df.ref_words.max())}")
    worst = df.nlargest(3, "wer_raw")[["case", "wer_raw"]]
    print("   worst 3: " + ", ".join(f"{r['case']} {r['wer_raw']*100:.1f}%"
                                     for _, r in worst.iterrows()))
    # df.sub would resolve to the pandas string method, not the column
    n_sub = int(df["sub"].sum()); n_del = int(df["del"].sum()); n_ins = int(df["ins"].sum())
    print(f"\n   insertions {n_ins:,} vs deletions {n_del:,} "
          f"vs substitutions {n_sub:,}")
    if n_ins > n_del:
        print("   More insertions than deletions - check for hallucination in the")
        print("   worst recordings before attributing this to speech recovery.")

    print(f"\nSaved -> {a.csv}")


if __name__ == "__main__":
    main()
