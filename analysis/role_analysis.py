#!/usr/bin/env python3
"""
role_analysis.py - characterise WHY doctor/patient role assignment fails.

WHY THIS MATTERS MORE THAN WER
    Diarization can be near-perfect and the transcript still useless. If the two
    speaker clusters are swapped, every question the student asked is attributed
    to the patient and every symptom the patient reported is attributed to the
    student. A transcript that is 97% accurate by WER becomes unusable for
    scoring - and nothing in WER or WDER reveals it, because both permit
    permutation-invariant speaker mapping.

    Role assignment is therefore a separate task with a separate failure mode,
    and it is binary per recording: right, or catastrophically wrong.

WHAT IT MEASURES
    Three rules, scored against the truth on every recording:
      first-speaker   - whoever talks first is the doctor
      question-ratio  - the cluster asking more questions is the doctor
      lexical         - interrogative phrasing and clinical vocabulary

    Then, for every failure: how long the opening turn was, how lopsided the
    question ratio was, and how close the decision was. That turns "85% correct"
    into "here is when it fails and here is why".

USAGE
    python role_analysis.py \\
        --result-dir ${ASR_CACHE_ROOT} \\
        --ref-dir "${ASR_DATA_ROOT}/Clean Transcripts" \\
        --csv role_analysis.csv
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

INTERROGATIVE = ["can you", "have you", "do you", "did you", "are you",
                 "would you", "could you", "tell me", "how long", "how many",
                 "when did", "what kind", "any other", "before we", "let me",
                 "on a scale", "i would like", "what brings"]
PATIENT_PHRASE = ["i feel", "i have", "my pain", "i've been", "it hurts",
                  "i'm worried", "i had", "i think it", "for me", "i took",
                  "i noticed", "it started"]


def norm(s):
    s = re.sub(r"[^a-z0-9' ]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def parse_reference(path):
    pairs, cur = [], None
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^\s*([DP])\s*:\s*(.*)$", line, flags=re.I)
        if m:
            cur = "Student" if m.group(1).upper() == "D" else "Patient"
            text, marked = m.group(2), True
        else:
            if cur is None:
                continue
            text, marked = line, False
        pairs += [(w, cur) for w in norm(text).split()]
    return pairs


def parse_result(path):
    df = pd.read_excel(path)
    col = next((c for c in df.columns if "transcript" in c.lower()
                and "researcher" not in c.lower()), None)
    if col is None or "Speaker" not in df.columns:
        return [], []
    words, turns = [], []
    for _, r in df.iterrows():
        if pd.isna(r["Speaker"]) or pd.isna(r[col]):
            continue
        spk = str(r["Speaker"]).strip()
        ws = norm(r[col]).split()
        if not ws:
            continue
        turns.append((spk, len(ws), str(r[col])))
        words += [(w, spk) for w in ws]
    return words, turns


def rule_first(turns):
    if not turns:
        return {}
    first = turns[0][0]
    others = [t[0] for t in turns if t[0] != first]
    return {first: "Student", others[0]: "Patient"} if others else {first: "Student"}


def rule_questions(turns):
    """Count '?' per cluster. The doctor interrogates."""
    q, tot = {}, {}
    for spk, n, raw in turns:
        q[spk] = q.get(spk, 0) + raw.count("?")
        tot[spk] = tot.get(spk, 0) + n
    if len(q) < 2:
        return {s: "Student" for s in q}, 0.0
    rates = {s: q[s] / max(tot[s], 1) * 100 for s in q}
    ranked = sorted(rates, key=rates.get, reverse=True)
    margin = rates[ranked[0]] - rates[ranked[1]]
    return {ranked[0]: "Student", ranked[1]: "Patient"}, margin


def rule_lexical(words):
    text = {}
    for w, spk in words:
        text.setdefault(spk, []).append(w)
    if len(text) < 2:
        return {s: "Student" for s in text}, 0.0
    sc = {}
    for spk, ws in text.items():
        blob = " ".join(ws)
        n = max(len(ws), 1)
        sc[spk] = (sum(blob.count(p) for p in INTERROGATIVE)
                   - sum(blob.count(p) for p in PATIENT_PHRASE)) / n * 100
    ranked = sorted(sc, key=sc.get, reverse=True)
    return {ranked[0]: "Student", ranked[1]: "Patient"}, sc[ranked[0]] - sc[ranked[1]]


def truth_mapping(ref_pairs, hyp_words):
    """
    The mapping that minimises speaker error - the best any rule could achieve.
    Derived by aligning the word sequences, so it does not need timestamps.
    """
    rw = [w for w, _ in ref_pairs]
    hw = [w for w, _ in hyp_words]
    if not rw or not hw:
        return None, 0
    out = jiwer.process_words(" ".join(rw), " ".join(hw))
    aligned = []
    for ch in out.alignments[0]:
        if ch.type in ("equal", "substitute"):
            for k in range(ch.ref_end_idx - ch.ref_start_idx):
                aligned.append((ref_pairs[ch.ref_start_idx + k][1],
                                hyp_words[ch.hyp_start_idx + k][1]))
    if not aligned:
        return None, 0
    hyp_roles = sorted({h for _, h in aligned})
    if len(hyp_roles) != 2:
        return None, len(aligned)
    cands = [{hyp_roles[0]: "Student", hyp_roles[1]: "Patient"},
             {hyp_roles[0]: "Patient", hyp_roles[1]: "Student"}]
    best = min(cands, key=lambda m: sum(1 for r, h in aligned if m[h] != r))
    return best, len(aligned)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--csv", default="role_analysis.csv")
    a = ap.parse_args()

    rows = []
    for xlsx in sorted(Path(a.result_dir).glob("*_large_v3_review.xlsx")):
        case = xlsx.stem.split("_large_v3")[0]
        ref_p = Path(a.ref_dir) / f"{case}.txt"
        if not ref_p.exists():
            continue
        ref_pairs = parse_reference(ref_p)
        hyp_words, turns = parse_result(xlsx)
        if not ref_pairs or not hyp_words:
            continue
        truth, n_aligned = truth_mapping(ref_pairs, hyp_words)
        if truth is None:
            continue

        first = rule_first(turns)
        ques, q_margin = rule_questions(turns)
        lex, l_margin = rule_lexical(hyp_words)

        def ok(m):
            return all(m.get(k) == truth.get(k) for k in truth if k in m)

        rows.append({
            "case": case,
            "category": re.match(r"^([A-Z]+)", case).group(1),
            "ref_opens": ref_pairs[0][1],
            "first_ok": ok(first),
            "question_ok": ok(ques),
            "lexical_ok": ok(lex),
            "opening_turn_words": turns[0][1] if turns else 0,
            "question_margin": round(q_margin, 3),
            "lexical_margin": round(l_margin, 3),
            "n_turns": len(turns),
            "aligned_words": n_aligned,
        })

    if not rows:
        sys.exit("nothing analysed")
    df = pd.DataFrame(rows)
    df.to_csv(a.csv, index=False)

    n = len(df)
    print(f"\n{'='*72}\nROLE ASSIGNMENT - {n} recordings\n{'='*72}")
    print(f"reference opens with the doctor : "
          f"{(df.ref_opens=='Student').mean()*100:5.1f}%")
    print()
    for col, label in [("first_ok", "first-speaker rule"),
                       ("question_ok", "question-ratio rule"),
                       ("lexical_ok", "lexical rule")]:
        print(f"{label:22s} correct : {df[col].mean()*100:5.1f}%  "
              f"({int(df[col].sum())}/{n})")

    # where the rules disagree is where assignment is fragile
    dis = df[(df.first_ok != df.lexical_ok) | (df.question_ok != df.lexical_ok)]
    print(f"\nrules disagree on {len(dis)} of {n} recordings")

    fails = df[~df.first_ok]
    if len(fails):
        print(f"\nfirst-speaker rule failed on {len(fails)}:")
        for _, r in fails.iterrows():
            print(f"  {r['case']:10s} ref opens with {r['ref_opens']:8s} "
                  f"opening turn {int(r['opening_turn_words']):3d} words   "
                  f"lexical {'ok' if r['lexical_ok'] else 'ALSO FAILED'}")
        print(f"\n  median opening turn, failures : "
              f"{fails.opening_turn_words.median():.0f} words")
        print(f"  median opening turn, successes: "
              f"{df[df.first_ok].opening_turn_words.median():.0f} words")
        print("\n  A short opening turn is the risk: a cough, a greeting or a")
        print("  mislabelled first word inverts the whole transcript, because")
        print("  with two speakers naming one determines both.")

    lex_fail = df[~df.lexical_ok]
    if len(lex_fail):
        print(f"\nlexical rule failed on {len(lex_fail)} - decision margins:")
        for _, r in lex_fail.iterrows():
            print(f"  {r['case']:10s} margin {r['lexical_margin']:+.3f} "
                  f"({'thin' if abs(r['lexical_margin']) < 0.3 else 'confident but wrong'})")
    else:
        print("\nlexical rule: no failures.")
        thin = df[df.lexical_margin.abs() < 0.3]
        print(f"  {len(thin)} recordings decided on a thin margin (<0.3) - these")
        print("  are the ones at risk on new data, even though all were correct here.")

    print(f"\nSaved -> {a.csv}")


if __name__ == "__main__":
    main()
