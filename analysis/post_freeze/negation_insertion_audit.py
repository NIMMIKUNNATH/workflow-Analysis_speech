#!/usr/bin/env python3
"""
negation_insertion_audit.py
---------------------------
Closes the insertion-attribution gap in the clinical-cue scorer.

The frozen scorer category-attributes reference-token SUBSTITUTIONS and DELETIONS
but not hypothesis-only INSERTIONS. Minimum-interval 0.40 s removes 830 insertions
relative to baseline, so the baseline's negation performance is measured with a
metric that is structurally blind to 830 hypothesis words -- including any
hallucinated negation cues among them.

This script recomputes the negation decomposition with a fourth class:

    deleted      reference cue -> nothing            (scored today)
    substituted  reference cue -> different token    (scored today)
    equivalent   reference cue -> NEG_EQUIV variant  (not an error)
    hallucinated cue appears as a hypothesis-only insertion   <-- NEW

It is ADDITIVE. It does not modify or replace the frozen scorer.

VERIFICATION GATE
-----------------
Run with --verify first. The script must reproduce main-manuscript Table 1
(baseline 101 = 48 D + 53 S; min-interval 0.40 s 200 = 97 D + 103 S) from your
retained transcripts. If it does not, the harness is misconfigured (normalisation,
tokenisation, or file pairing) and the hallucination counts must NOT be trusted.

USAGE
-----
  python negation_insertion_audit.py \
      --refs   /path/to/fareez/references \
      --hyps   /path/to/hypotheses \
      --lexicon lexicon_v1.json \
      --conditions baseline min_interval_00 min_interval_25 min_interval_40 \
      --subset dev_split_v2.json \
      --verify \
      --out negation_insertion_audit.csv

Directory layout assumed:
    hyps/<condition>/<RECORDING_ID>.txt
    refs/<RECORDING_ID>.txt
Override with --hyp-pattern / --ref-pattern if yours differs.
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Normalisation -- MUST match your frozen scorer. Edit here, nowhere else.
# ----------------------------------------------------------------------------

FILLERS = set()          # populated from lexicon_v1.json ("filler_tokens")
NUM_MAP = {}             # populated from lexicon_v1.json ("num_map")

# clinical_errors.py parse_reference(): "^\s*([DP])\s*:\s*(.*)$"
REF_PREFIX = re.compile(r"^\s*([DP])\s*:\s*(.*)$", re.IGNORECASE)
# our regenerated hypotheses use SPEAKER_00: / SPEAKER_01:
HYP_PREFIX = re.compile(r"^\s*(SPEAKER_\d+|Speaker [AB]|doctor|patient|"
                        r"physician|clinician|dr)\s*:\s*", re.IGNORECASE)


def read_text(path):
    """BOM-aware read. RES0002 and RES0054 are UTF-16 in the Fareez release."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    if raw[:3] == b"\xef\xbb\xbf":
        return raw[3:].decode("utf-8")
    return raw.decode("utf-8", errors="replace")


def normalise(text, drop_fillers=False):
    """
    Mirrors clinical_errors.norm():
        s = re.sub(r"[^a-z0-9' ]", " ", str(s).lower())
        s = re.sub(r"\\s+", " ", s).strip()

    Speaker prefixes are stripped first, as parse_reference() does. Fillers are
    NOT removed: the frozen scorer keeps them in the token stream and uses the
    FILLERS list only to classify errors. Removing them here would shift every
    alignment. NUM_MAP is likewise applied by the scorer at classification time,
    not during tokenisation, so it is off by default.
    """
    text = unicodedata.normalize("NFKC", text)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = REF_PREFIX.match(line)
        if m:
            line = m.group(2)
        else:
            line = HYP_PREFIX.sub("", line)
        lines.append(line)
    text = " ".join(lines).lower()
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    toks = text.split()
    if drop_fillers:
        toks = [t for t in toks if t not in FILLERS]
    return toks


# ----------------------------------------------------------------------------
# Alignment: Levenshtein with backtrace -> ops of (kind, ref_tok, hyp_tok)
# ----------------------------------------------------------------------------

def align_jiwer(ref, hyp):
    """
    Mirrors clinical_errors.py: jiwer.process_words over space-joined tokens,
    walking alignment chunks. Chunk types are equal / substitute / delete /
    insert, and substitute chunks pair ref[i+k] with hyp[j+k] positionally --
    exactly as the frozen scorer does.
    """
    import jiwer
    if not ref or not hyp:
        return [("D", r, None) for r in ref] + [("I", None, h) for h in hyp]
    out = jiwer.process_words(" ".join(ref), " ".join(hyp))
    ops = []
    for ch in out.alignments[0]:
        n_ref = ch.ref_end_idx - ch.ref_start_idx
        n_hyp = ch.hyp_end_idx - ch.hyp_start_idx
        if ch.type == "equal":
            for k in range(n_ref):
                ops.append(("C", ref[ch.ref_start_idx + k],
                            hyp[ch.hyp_start_idx + k]))
        elif ch.type == "substitute":
            for k in range(n_ref):
                ops.append(("S", ref[ch.ref_start_idx + k],
                            hyp[ch.hyp_start_idx + k]))
        elif ch.type == "delete":
            for k in range(n_ref):
                ops.append(("D", ref[ch.ref_start_idx + k], None))
        elif ch.type == "insert":
            for k in range(n_hyp):
                ops.append(("I", None, hyp[ch.hyp_start_idx + k]))
    return ops


def align(ref, hyp):
    """
    Standard edit-distance alignment. Returns list of tuples:
        ("C", r, h) correct   ("S", r, h) substitution
        ("D", r, None)        ("I", None, h)
    Backtrace prefers C > S > D > I on ties, matching the usual WER convention.
    """
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)

    for i in range(1, n + 1):
        ri = ref[i - 1]
        row_prev = d[i - 1]
        row_cur = d[i]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            row_cur[j] = min(row_prev[j - 1] + cost,   # sub / match
                             row_prev[j] + 1,          # deletion
                             row_cur[j - 1] + 1)       # insertion

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if d[i][j] == d[i - 1][j - 1] + cost:
                ops.append(("C" if cost == 0 else "S", ref[i - 1], hyp[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], None))
            i -= 1
            continue
        ops.append(("I", None, hyp[j - 1]))
        j -= 1
    ops.reverse()
    return ops


# ----------------------------------------------------------------------------
# Negation classification
# ----------------------------------------------------------------------------

def build_equiv(pairs):
    """NEG_EQUIV -> symmetric class map: token -> canonical class id."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in pairs:
        union(a.lower(), b.lower())
    return {tok: find(tok) for tok in parent}


def classify(ops, cues, equiv):
    """
    Walk the alignment once and bucket every negation event.
    Returns Counter with keys: ref_tokens, correct, equivalent,
    deleted, substituted, hallucinated.
    """
    c = Counter()
    for kind, r, h in ops:
        if r is not None and r in cues:
            c["ref_tokens"] += 1
            if kind == "C":
                c["correct"] += 1
            elif kind == "S":
                # equivalent variant within a predefined semantic class?
                if h is not None and equiv.get(r) is not None \
                        and equiv.get(r) == equiv.get(h):
                    c["equivalent"] += 1
                else:
                    c["substituted"] += 1
            elif kind == "D":
                c["deleted"] += 1
        elif kind == "I" and h is not None and h in cues:
            # hypothesis-only negation cue with no reference counterpart
            c["hallucinated"] += 1
    return c


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

TABLE1 = {  # main-manuscript Table 1, for the verification gate
    "baseline":         dict(errors=101, deleted=48,  substituted=53),
    "min_interval_00":  dict(errors=98,  deleted=45,  substituted=53),
    "min_interval_25":  dict(errors=110, deleted=51,  substituted=59),
    "min_interval_40":  dict(errors=200, deleted=97,  substituted=103),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True)
    ap.add_argument("--hyps", required=True)
    ap.add_argument("--lexicon", required=True)
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--subset", default=None,
                    help="JSON with the 40 dev ids (dev_split_v2.json), or omit for all")
    ap.add_argument("--ref-pattern", default="{rid}.txt")
    ap.add_argument("--hyp-pattern", default="{cond}/{rid}.txt")
    ap.add_argument("--drop-fillers", action="store_true",
                    help="remove filler tokens before aligning (the frozen "
                         "scorer does NOT do this; off by default)")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--own-aligner", action="store_true",
                    help="use the built-in Levenshtein instead of jiwer "
                         "(default is jiwer, matching clinical_errors.py)")
    ap.add_argument("--out", default="negation_insertion_audit.csv")
    ap.add_argument("--per-recording-out", default="negation_insertion_per_recording.csv")
    args = ap.parse_args()

    lex = json.load(open(args.lexicon, encoding="utf-8"))

    def pick(*names):
        for n in names:
            if n in lex:
                return lex[n]
        return None

    cues_raw = pick("clinical_errors.NEGATION",
                    "negation_cues_onefactor", "negation_cues", "negation")
    if cues_raw is None:
        sys.exit("Could not find the negation cue list in the lexicon. "
                 "Expected key 'negation_cues_onefactor' / 'negation_cues' / 'negation'.")
    cues = {c.lower() for c in cues_raw}

    equiv_raw = pick("clinical_errors.NEG_EQUIV",
                     "NEG_EQUIV", "neg_equiv", "negation_equivalence") or []
    if isinstance(equiv_raw, dict):
        # {variant: canonical} mapping
        equiv_pairs = [(k, v) for k, v in equiv_raw.items()]
    elif equiv_raw and isinstance(equiv_raw[0], dict):
        equiv_pairs = [(d["a"], d["b"]) for d in equiv_raw]
    else:
        equiv_pairs = [tuple(p) for p in equiv_raw]
    equiv = build_equiv(equiv_pairs)

    global FILLERS, NUM_MAP
    FILLERS = {t.lower() for t in
               (pick("clinical_errors.FILLERS", "filler_tokens", "fillers") or [])}
    NUM_MAP = {k.lower(): str(v) for k, v in
               (pick("clinical_errors.NUM_MAP", "num_map", "NUM_MAP") or {}).items()}

    print(f"[lexicon] {len(cues)} negation cues, {len(equiv_pairs)} equivalence pairs, "
          f"{len(FILLERS)} fillers, {len(NUM_MAP)} number mappings")

    if args.subset:
        sub = json.load(open(args.subset, encoding="utf-8"))
        ids = (sub if isinstance(sub, list)
               else (sub.get("dev") or sub.get("development") or sub.get("ids") or []))
        ids = sorted(set(ids))
    else:
        ids = sorted(os.path.splitext(f)[0] for f in os.listdir(args.refs)
                     if f.endswith(".txt"))
    print(f"[subset] {len(ids)} recordings")

    rows, per_rec = [], []
    for cond in args.conditions:
        agg = Counter()
        n_scored = 0
        for rid in ids:
            rp = os.path.join(args.refs, args.ref_pattern.format(rid=rid))
            hp = os.path.join(args.hyps, args.hyp_pattern.format(cond=cond, rid=rid))
            if not (os.path.exists(rp) and os.path.exists(hp)):
                continue
            ref = normalise(read_text(rp), args.drop_fillers)
            hyp = normalise(read_text(hp), args.drop_fillers)
            if not hyp:                      # scorer policy L07: no result
                continue
            aligner = align if args.own_aligner else align_jiwer
            c = classify(aligner(ref, hyp), cues, equiv)
            agg.update(c)
            n_scored += 1
            per_rec.append(dict(condition=cond, recording=rid,
                                ref_tokens=c["ref_tokens"],
                                deleted=c["deleted"],
                                substituted=c["substituted"],
                                equivalent=c["equivalent"],
                                hallucinated=c["hallucinated"]))

        N = agg["ref_tokens"]
        scored_err = agg["deleted"] + agg["substituted"]
        total_err = scored_err + agg["hallucinated"]
        rows.append(dict(
            condition=cond,
            recordings=n_scored,
            ref_negation_tokens=N,
            deleted=agg["deleted"],
            substituted=agg["substituted"],
            equivalent=agg["equivalent"],
            hallucinated=agg["hallucinated"],
            scored_errors=scored_err,
            scored_rate_pct=round(100 * scored_err / N, 3) if N else None,
            total_errors_incl_halluc=total_err,
            total_rate_pct=round(100 * total_err / N, 3) if N else None,
        ))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    pd.DataFrame(per_rec).to_csv(args.per_recording_out, index=False)

    pd.set_option("display.width", 200)
    print("\n" + df.to_string(index=False))
    print(f"\n[write] {args.out}")
    print(f"[write] {args.per_recording_out}")

    if args.verify:
        print("\n=== VERIFICATION GATE (main-manuscript Table 1) ===")
        ok = True
        for _, r in df.iterrows():
            exp = TABLE1.get(r["condition"])
            if not exp:
                continue
            hit = (r["scored_errors"] == exp["errors"]
                   and r["deleted"] == exp["deleted"]
                   and r["substituted"] == exp["substituted"])
            ok &= hit
            print(f"  {r['condition']:<20} got {int(r['scored_errors'])}"
                  f" = {int(r['deleted'])}D + {int(r['substituted'])}S   "
                  f"expected {exp['errors']} = {exp['deleted']}D + {exp['substituted']}S"
                  f"   {'PASS' if hit else 'FAIL'}")
        if ok:
            print("\n  All conditions reproduce Table 1. "
                  "Hallucination counts are trustworthy.")
        else:
            print("\n  MISMATCH. Align normalise() with your frozen scorer before "
                  "interpreting the hallucinated column. Do not report these numbers yet.")
            sys.exit(1)

    if "hallucinated" in df:
        print("\n=== INTERPRETATION ===")
        base = df[df.condition == "baseline"]
        if len(base):
            b = int(base.iloc[0]["hallucinated"])
            for _, r in df.iterrows():
                if r["condition"] == "baseline":
                    continue
                print(f"  {r['condition']:<20} hallucinated {int(r['hallucinated']):>4}"
                      f"   (baseline {b:>4}, delta {int(r['hallucinated']) - b:+d})")
            print("\n  If baseline hallucinates MORE negation cues than min-interval 0.40 s,\n"
                  "  the frozen scorer understates baseline negation error and the\n"
                  "  WER-vs-negation discordance holds in both directions -- report it.\n"
                  "  If the counts are comparable, the insertion-attribution objection\n"
                  "  is answered empirically. Either result strengthens the manuscript.")


if __name__ == "__main__":
    main()
