#!/usr/bin/env python3
"""
score_fareez.py - score the pipeline against the published Fareez et al. (2022)
OSCE dataset (272 simulated patient-physician interviews, Scientific Data).

WHY A SEPARATE SCORER
    score_run.py measures speaker error by TIME: it needs Start/End timestamps on
    every reference segment. This dataset has no timestamps - transcripts are
    plain dialogue:

        D: What brought you in today?
        P: Sure, I'm just having a lot of chest pain...

    So time-weighted speaker error cannot be computed. Instead this uses
    WORD-LEVEL DIARIZATION ERROR RATE (WDER), which needs no timestamps at all:

        1. Normalise and align the reference and hypothesis word sequences
        2. For every word the aligner matches (hit or substitution), compare the
           speaker label attached to each side
        3. WDER = mismatched aligned words / total aligned words

    This is a recognised metric for joint ASR-and-diarization systems, and it is
    arguably fairer than the time-weighted version because it counts errors on
    words that were actually transcribed rather than on seconds of audio.

    The two numbers are NOT interchangeable. Do not compare a WDER here against a
    time-weighted speaker error from score_run.py - report them separately and say
    which is which.

WHY THIS DATASET MATTERS
    The reference was produced independently of this pipeline. The local reference
    in the local study was made by post-editing this pipeline's own output, so
    53% of its words are unchanged from the system being measured - a bias that
    flatters the system and cannot be quantified. Nothing of that kind applies
    here, which makes these numbers the ones to publish.

USAGE
    python score_fareez.py --result  <pipeline output .xlsx> \\
                           --ref     "/path/to/Clean Transcripts/CAR0001.txt"

    Whole folder at once:
    python score_fareez.py --result-dir <folder of .xlsx> \\
                           --ref-dir "/path/to/Clean Transcripts" \\
                           --csv fareez_results.csv
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd

try:
    import jiwer
except ImportError:
    sys.exit("pip install jiwer pandas openpyxl")


# D is the doctor / medical student, P is the simulated patient. The local study
# calls these Student and Patient, so map onto the same vocabulary.
SPEAKER_PREFIX = {"D": "Student", "P": "Patient"}


def norm(s):
    s = re.sub(r"[^a-z0-9' ]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


# ------------------------------------------------------------- reference ----
def parse_reference(path):
    """
    Read a Clean Transcript into a flat list of (word, speaker) pairs.

    Lines look like "D: What brought you in today?". A line without a prefix is
    treated as a continuation of the previous speaker rather than dropped, since
    losing it would inflate the deletion count and misattribute the words after it.
    """
    turns, current = [], None
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^\s*([DP])\s*:\s*(.*)$", line, flags=re.I)
        if m:
            current = SPEAKER_PREFIX[m.group(1).upper()]
            text = m.group(2)
        else:
            if current is None:
                continue          # preamble before the first labelled turn
            text = line
        for w in norm(text).split():
            turns.append((w, current))
    return turns


# ------------------------------------------------------------ hypothesis ----
def parse_result(path):
    """Flatten the pipeline's review sheet into (word, speaker) pairs."""
    df = pd.read_excel(path)
    txt_col = next((c for c in df.columns
                    if "transcript" in c.lower() and "researcher" not in c.lower()), None)
    if txt_col is None or "Speaker" not in df.columns:
        sys.exit(f"{path}: expected a 'Speaker' column and a transcript column, "
                 f"found {list(df.columns)}")
    out = []
    for _, r in df.iterrows():
        if pd.isna(r["Speaker"]) or pd.isna(r[txt_col]):
            continue
        spk = str(r["Speaker"]).strip()
        for w in norm(r[txt_col]).split():
            out.append((w, spk))
    return out


# ---------------------------------------------------------------- scoring ---
def score(ref_pairs, hyp_pairs):
    ref_words = [w for w, _ in ref_pairs]
    hyp_words = [w for w, _ in hyp_pairs]
    if not ref_words or not hyp_words:
        return None

    out = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words))
    align = out.alignments[0]

    # Collect (reference speaker, hypothesis speaker) for every word the aligner
    # matched. Insertions and deletions are excluded: there is no counterpart to
    # compare against, and counting them here would double-penalise ASR errors
    # that WER has already measured.
    pairs = []
    for ch in align:
        if ch.type in ("equal", "substitute"):
            n = ch.ref_end_idx - ch.ref_start_idx
            for k in range(n):
                pairs.append((ref_pairs[ch.ref_start_idx + k][1],
                              hyp_pairs[ch.hyp_start_idx + k][1]))
    if not pairs:
        return None

    # Speaker IDs are arbitrary ("Speaker A" may be either person), so try both
    # orientations and take the better. A run is not penalised for naming.
    ref_roles = sorted({r for r, _ in pairs})
    hyp_roles = sorted({h for _, h in pairs})
    candidates = [{h: h for h in hyp_roles}]
    if len(ref_roles) == 2 and len(hyp_roles) == 2:
        candidates += [{hyp_roles[0]: ref_roles[0], hyp_roles[1]: ref_roles[1]},
                       {hyp_roles[0]: ref_roles[1], hyp_roles[1]: ref_roles[0]}]
    best = min(
        sum(1 for r, h in pairs if m.get(h, h) != r) / len(pairs)
        for m in candidates
    )

    return {
        "wer": round(out.wer, 4),
        "wder": round(best, 4),
        "sub": out.substitutions, "del": out.deletions, "ins": out.insertions,
        "ref_words": len(ref_words), "hyp_words": len(hyp_words),
        "aligned_words": len(pairs),
    }


# ------------------------------------------------ patient sex from content --
SEX_PATTERNS = [
    (re.compile(r"\bi'?m a (\d{1,3}[- ]?year[- ]?old )?(male|man|boy)\b"), "M"),
    (re.compile(r"\bi'?m a (\d{1,3}[- ]?year[- ]?old )?(female|woman|girl|lady)\b"), "F"),
    (re.compile(r"\b(\d{1,3})[, ]+i'?m a (male|man)\b"), "M"),
    (re.compile(r"\b(\d{1,3})[, ]+i'?m a (female|woman)\b"), "F"),
]


def patient_sex(ref_pairs):
    """
    Recover the patient's stated sex from the transcript.

    OSCE histories almost always open with age and sex ("Sure 39, I'm a male"),
    so this is recoverable from text for a large fraction of cases WITHOUT
    listening to anything. It matters because the strongest local finding was that
    speaker error tripled between mixed-sex and same-sex pairs (1.49% vs 4.92%) -
    on n=2. This dataset can test that properly.

    Only the PATIENT's sex is recoverable this way; the physician rarely states
    theirs. So this supports "patient male vs patient female" comparisons, not a
    full four-way pairing analysis, unless the physician's sex is labelled by ear.
    """
    text = " ".join(w for w, spk in ref_pairs if spk == "Patient")
    for pat, sex in SEX_PATTERNS:
        if pat.search(text):
            return sex
    return "?"


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result")
    ap.add_argument("--ref")
    ap.add_argument("--result-dir")
    ap.add_argument("--ref-dir")
    ap.add_argument("--csv", default="fareez_results.csv")
    args = ap.parse_args()

    jobs = []
    if args.result and args.ref:
        jobs.append((Path(args.result), Path(args.ref)))
    elif args.result_dir and args.ref_dir:
        rd, fd = Path(args.result_dir), Path(args.ref_dir)
        for x in sorted(rd.glob("*.xlsx")):
            stem = x.stem.split("_large_v3")[0]
            ref = fd / f"{stem}.txt"
            if ref.exists():
                jobs.append((x, ref))
            else:
                print(f"  no transcript for {x.name} - skipped")
    else:
        sys.exit("give --result and --ref, or --result-dir and --ref-dir")

    if not jobs:
        sys.exit("nothing to score")

    rows = []
    for res_p, ref_p in jobs:
        ref_pairs = parse_reference(ref_p)
        hyp_pairs = parse_result(res_p)
        m = score(ref_pairs, hyp_pairs)
        if not m:
            print(f"  {ref_p.stem}: empty after parsing - skipped")
            continue
        m["case"] = ref_p.stem
        m["category"] = re.match(r"^([A-Z]+)", ref_p.stem).group(1)
        m["patient_sex"] = patient_sex(ref_pairs)
        rows.append(m)
        print(f"  {m['case']:10s} WER {m['wer']*100:6.2f}%   "
              f"WDER {m['wder']*100:5.2f}%   "
              f"ref {m['ref_words']:5d} / hyp {m['hyp_words']:5d}  "
              f"[{m['patient_sex']}]")

    if not rows:
        sys.exit("no scored files")

    df = pd.DataFrame(rows)[["case", "category", "patient_sex", "wer", "wder",
                             "sub", "del", "ins", "ref_words", "hyp_words",
                             "aligned_words"]]
    df.to_csv(args.csv, index=False)

    print(f"\n{'='*66}\n{len(df)} recordings scored\n{'='*66}")
    print(f"WER   mean {df.wer.mean()*100:6.2f}%   median {df.wer.median()*100:6.2f}%   "
          f"SD {df.wer.std()*100:5.2f}   range {df.wer.min()*100:.1f}-{df.wer.max()*100:.1f}%")
    print(f"WDER  mean {df.wder.mean()*100:6.2f}%   median {df.wder.median()*100:6.2f}%   "
          f"SD {df.wder.std()*100:5.2f}   range {df.wder.min()*100:.1f}-{df.wder.max()*100:.1f}%")

    if len(df) >= 4:
        print("\nby clinical category:")
        g = df.groupby("category").agg(n=("wer", "size"), wer=("wer", "mean"),
                                       wder=("wder", "mean"))
        for cat, r in g.iterrows():
            print(f"  {cat:5s} n={int(r['n']):3d}   WER {r['wer']*100:6.2f}%   "
                  f"WDER {r['wder']*100:5.2f}%")

        known = df[df.patient_sex != "?"]
        if len(known) >= 4 and known.patient_sex.nunique() > 1:
            print(f"\nby patient sex ({len(known)} of {len(df)} recoverable from text):")
            g2 = known.groupby("patient_sex").agg(n=("wder", "size"),
                                                  wer=("wer", "mean"),
                                                  wder=("wder", "mean"))
            for sx, r in g2.iterrows():
                print(f"  {sx}     n={int(r['n']):3d}   WER {r['wer']*100:6.2f}%   "
                      f"WDER {r['wder']*100:5.2f}%")

    print(f"\nSaved -> {args.csv}")
    print("\nNote: WDER is word-level and is NOT comparable with the time-weighted\n"
          "speaker error reported by score_run.py. Report them separately.")


if __name__ == "__main__":
    main()
