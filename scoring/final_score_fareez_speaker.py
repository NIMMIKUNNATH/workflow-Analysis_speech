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
def read_transcript(path):
    """
    Read a reference transcript, detecting its encoding.

    Two of the 272 Fareez transcripts (RES0002, RES0054) are UTF-16
    little-endian while the rest are UTF-8. Reading UTF-16 as UTF-8 yields a
    string full of NUL bytes, which the word-splitter then reduces to nothing -
    so those two recordings silently scored zero words and were dropped from
    every analysis. That is a 2-recording sample loss caused entirely by an
    encoding assumption.

    The BOM (\xff\xfe or \xfe\xff) identifies UTF-16 unambiguously, so detect
    it from the raw bytes rather than guessing from the decoded text.
    """
    raw = Path(path).read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def parse_reference(path):
    """
    Read a Clean Transcript into a flat list of (word, speaker) pairs.

    Lines look like "D: What brought you in today?". A line without a prefix is
    treated as a continuation of the previous speaker rather than dropped, since
    losing it would inflate the deletion count and misattribute the words after it.
    """
    turns, current = [], None
    for raw in read_transcript(path).splitlines():
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

    # ---- per-role word error rate -------------------------------------
    # WER computed separately over the doctor's words and the patient's.
    # This is NOT the same as WDER: WDER asks "was this word given to the right
    # speaker", whereas this asks "for the words this speaker actually said, how
    # many did the system get right". A corpus can have excellent WDER and still
    # transcribe one role far worse than the other - short patient backchannels
    # ("no", "mm", "yeah") are a plausible asymmetry, and one that matters
    # clinically because negations mostly come from the patient.
    #
    # Method: walk the alignment once and, for each reference word belonging to
    # this role, decide whether it was recovered. Substituted and deleted words
    # count as errors; insertions have no reference speaker and cannot be
    # attributed, so they are excluded, exactly as in WDER.
    #
    # This is therefore a SUBSTITUTION-PLUS-DELETION rate, not a WER in the same
    # sense as the headline figure, which includes insertions. Report it as
    # such - the two are not directly comparable.
    #
    # NOTE. An earlier version built parallel r_words / h_words lists and zipped
    # them. That is wrong: deletions append to the reference list only, so after
    # the first deletion every subsequent pair is offset and the substitution
    # count is meaningless. Counting in one pass avoids the problem entirely.
    per_role = {}
    for role in sorted({r for _, r in ref_pairs}):
        n = err = 0
        for ch in align:
            if ch.type in ("equal", "substitute"):
                for k in range(ch.ref_end_idx - ch.ref_start_idx):
                    i = ch.ref_start_idx + k
                    if ref_pairs[i][1] != role:
                        continue
                    n += 1
                    if ref_words[i] != hyp_words[ch.hyp_start_idx + k]:
                        err += 1
            elif ch.type == "delete":
                for k in range(ch.ref_end_idx - ch.ref_start_idx):
                    if ref_pairs[ch.ref_start_idx + k][1] == role:
                        n += 1
                        err += 1
        if not n:
            continue
        per_role[role] = {"n": n, "err": err, "wer": round(err / n, 4)}

    # ---- speaker-attributed WER ---------------------------------------
    # Words counted correct only if BOTH the word and its speaker label are
    # right. This is the metric that reflects what a reader of the transcript
    # actually sees: a correctly recognised word under the wrong name is not
    # usable evidence of what the student asked.
    #
    # INSERTIONS ARE INCLUDED. The headline WER counts them, so a metric billed
    # as stricter than WER must count them too - otherwise SA-WER scores BETTER
    # than WER (there are roughly 4,795 insertions against 1,912 deletions on the
    # baseline) and the comparison is indefensible. With them in, SA-WER >= WER
    # always, and the gap between the two is exactly the cost of speaker
    # mislabelling, which is the quantity of interest.
    #
    # The speaker mapping is chosen the same way as for WDER, on the same
    # `pairs`, so the two metrics cannot disagree about which cluster is which.
    best_map = min(candidates, key=lambda m:
                   sum(1 for r, h in pairs if m.get(h, h) != r))
    sa_correct = 0
    for ch in align:
        if ch.type == "equal":
            for k in range(ch.ref_end_idx - ch.ref_start_idx):
                rspk = ref_pairs[ch.ref_start_idx + k][1]
                hspk = hyp_pairs[ch.hyp_start_idx + k][1]
                if best_map.get(hspk, hspk) == rspk:
                    sa_correct += 1
    sa_wer = (round(((len(ref_words) - sa_correct) + out.insertions)
                    / len(ref_words), 4) if ref_words else None)

    res = {
        "wer": round(out.wer, 4),
        "wder": round(best, 4),
        "sa_wer": sa_wer,
        "sub": out.substitutions, "del": out.deletions, "ins": out.insertions,
        "ref_words": len(ref_words), "hyp_words": len(hyp_words),
        "aligned_words": len(pairs),
        "_aligned": pairs,
    }
    for role, d in per_role.items():
        key = "doctor" if role in ("D", "Student", "doctor") else "patient"
        res[f"{key}_n"] = d["n"]
        res[f"{key}_err"] = d["err"]
        res[f"{key}_wer"] = d["wer"]
    return res


# ------------------------------------------------------ role assignment ----
STUDENT_MARKERS = ["can you", "have you", "do you", "did you", "would you",
                   "tell me", "any other", "i would like to", "let me",
                   "on a scale", "how long", "what brings", "before we start"]
PATIENT_MARKERS = ["i feel", "i have", "my pain", "i've been", "it hurts",
                   "i'm worried", "i had", "i think it", "for me"]


def role_by_first_speaker(hyp_pairs):
    """
    The intuitive rule: in an OSCE the doctor opens, so whichever cluster speaks
    first is the doctor.

    Cheap and usually right - but it rests on a single moment. A cough, a door,
    a throat-clear, or one mislabelled opening turn inverts the ENTIRE transcript,
    because with two speakers naming one determines both. That fragility is the
    reason it is worth measuring rather than assuming.
    """
    if not hyp_pairs:
        return {}
    first = hyp_pairs[0][1]
    others = [s for _, s in hyp_pairs if s != first]
    if not others:
        return {first: "Student"}
    return {first: "Student", others[0]: "Patient"}


def role_by_content(hyp_pairs):
    """
    The alternative: the doctor interrogates, the patient narrates. Score each
    cluster on question marks and interrogative phrasing.

    Slower to fool, because it uses the whole recording rather than one moment.
    """
    text = {}
    for w, spk in hyp_pairs:
        text.setdefault(spk, []).append(w)
    if len(text) < 2:
        return {s: "Student" for s in text}

    scores = {}
    for spk, words in text.items():
        blob = " ".join(words)
        n = max(len(words), 1)
        st = sum(blob.count(m) for m in STUDENT_MARKERS)
        pt = sum(blob.count(m) for m in PATIENT_MARKERS)
        scores[spk] = (st - pt) / n * 100

    ranked = sorted(scores, key=scores.get, reverse=True)
    mapping = {ranked[0]: "Student"}
    for s in ranked[1:]:
        mapping[s] = "Patient"
    return mapping


def evaluate_roles(ref_pairs, hyp_pairs, aligned):
    """
    Compare both rules against the truth.

    'Truth' here is the mapping that minimises WDER - i.e. the orientation the
    scorer already chooses when it tries both ways round. That is the best any
    role-assignment rule could do, so it is the right thing to measure against.
    """
    if not aligned:
        return {}
    hyp_roles = sorted({h for _, h in aligned})
    ref_roles = sorted({r for r, _ in aligned})
    if len(hyp_roles) != 2 or len(ref_roles) != 2:
        return {"first_ok": None, "content_ok": None, "rules_agree": None,
                "ref_opens_with": ref_pairs[0][1] if ref_pairs else None}

    cands = [{hyp_roles[0]: ref_roles[0], hyp_roles[1]: ref_roles[1]},
             {hyp_roles[0]: ref_roles[1], hyp_roles[1]: ref_roles[0]}]
    best = min(cands, key=lambda m:
               sum(1 for r, h in aligned if m[h] != r))

    first_map = role_by_first_speaker(hyp_pairs)
    content_map = role_by_content(hyp_pairs)

    def agrees(m):
        return all(m.get(k) == best.get(k) for k in hyp_roles if k in m)

    return {
        "first_ok": agrees(first_map),
        "content_ok": agrees(content_map),
        "rules_agree": all(first_map.get(k) == content_map.get(k)
                           for k in hyp_roles),
        "ref_opens_with": ref_pairs[0][1] if ref_pairs else None,
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
        m.update(evaluate_roles(ref_pairs, hyp_pairs, m.pop("_aligned")))
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

    keep = ["case", "category", "patient_sex", "wer", "wder", "sa_wer",
            "doctor_wer", "patient_wer", "doctor_n", "patient_n",
            "doctor_err", "patient_err",
            "first_ok", "content_ok", "rules_agree", "ref_opens_with",
            "sub", "del", "ins", "ref_words", "hyp_words", "aligned_words"]
    df = pd.DataFrame(rows)
    df = df[[c for c in keep if c in df.columns]]
    df.to_csv(args.csv, index=False)

    print(f"\n{'='*66}\n{len(df)} recordings scored\n{'='*66}")
    print(f"WER   mean {df.wer.mean()*100:6.2f}%   median {df.wer.median()*100:6.2f}%   "
          f"SD {df.wer.std()*100:5.2f}   range {df.wer.min()*100:.1f}-{df.wer.max()*100:.1f}%")
    print(f"WDER  mean {df.wder.mean()*100:6.2f}%   median {df.wder.median()*100:6.2f}%   "
          f"SD {df.wder.std()*100:5.2f}   range {df.wder.min()*100:.1f}-{df.wder.max()*100:.1f}%")
    if "sa_wer" in df.columns and df.sa_wer.notna().any():
        gap = (df.sa_wer.mean() - df.wer.mean()) * 100
        print(f"SA-WER mean {df.sa_wer.mean()*100:6.2f}%   "
              f"(word AND speaker both correct; insertions included)")
        print(f"  SA-WER exceeds WER by {gap:+5.2f} points - that gap IS the cost "
              f"of speaker mislabelling.")

    # per-role WER, computed over each speaker's own reference words
    if {"doctor_n", "patient_n"}.issubset(df.columns):
        dn, de = int(df.doctor_n.sum()), int(df.doctor_err.sum())
        pn, pe = int(df.patient_n.sum()), int(df.patient_err.sum())
        print(f"\nper-role word error rate")
        print(f"  doctor    {de:6d} / {dn:7d}  = {de/dn*100:6.2f}%")
        print(f"  patient   {pe:6d} / {pn:7d}  = {pe/pn*100:6.2f}%")
        diff = pe/pn*100 - de/dn*100
        print(f"  difference {diff:+6.2f} points  "
              f"({'patient harder' if diff > 0 else 'doctor harder'})")
        print("\n  Per-role WER is NOT WDER. WDER asks whether a word went to the")
        print("  right speaker; this asks how accurately each speaker's own words")
        print("  were transcribed. A systematic gap matters clinically because")
        print("  negations and short responses come mostly from the patient.")
        print("  It is also NOT comparable to the headline WER above: insertions")
        print("  have no reference speaker, so these are substitution-plus-")
        print("  deletion rates. Label them that way in any table.")

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

    # --- role assignment ---------------------------------------------------
    rr = df.dropna(subset=["first_ok"])
    if len(rr):
        print(f"\nrole assignment ({len(rr)} recordings with two clusters):")
        print(f"  first-speaker rule correct : {rr.first_ok.mean()*100:5.1f}%")
        print(f"  content-based rule correct : {rr.content_ok.mean()*100:5.1f}%")
        print(f"  the two rules agree        : {rr.rules_agree.mean()*100:5.1f}%")
        opens = df.ref_opens_with.value_counts(dropna=True)
        if len(opens):
            tot = opens.sum()
            shown = ", ".join(f"{k} {v}/{tot} ({v/tot*100:.0f}%)"
                              for k, v in opens.items())
            print(f"  reference opens with       : {shown}")
        print("\n  A rule that is right 95% of the time is usable. One that is right")
        print("  70% of the time inverts a third of the transcripts entirely, since")
        print("  with two speakers a single wrong assignment flips both labels.")

    print(f"\nSaved -> {args.csv}")
    print("\nNote: WDER is word-level and is NOT comparable with the time-weighted\n"
          "speaker error reported by score_run.py. Report them separately.")


if __name__ == "__main__":
    main()
