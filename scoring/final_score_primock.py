#!/usr/bin/env python3
"""
score_primock.py - score the pipeline on PriMock57, including TRUE DER.

WHAT THIS ADDS OVER THE FAREEZ SCORER
    Fareez has no reference timestamps, so only word-level diarization error rate
    (WDER) was computable there. PriMock57 utterances carry start and end times,
    so this computes DIARIZATION ERROR RATE properly - missed speech, false alarm
    and speaker confusion, measured in seconds against reference intervals.

    Reporting both on the same recordings also shows how the two metrics relate,
    which is useful context for the Fareez results where only one was available.

    This is additionally the FIRST HELD-OUT TEST of the lexical doctor/patient
    rule. That rule was built by inspecting Fareez transcripts and evaluated on
    the same data - resubstitution accuracy, not performance. PriMock57 is unseen,
    so whatever it scores here is the honest figure.

DER, AS IMPLEMENTED
    The audio is sampled on a fine grid (default 10 ms). At each frame the
    reference and hypothesis each have a set of active speakers. Following the
    NIST convention:

        missed speech   reference speaks, hypothesis does not
        false alarm     hypothesis speaks, reference does not
        confusion       both speak, but the mapped identity differs

        DER = (missed + false alarm + confusion) / total reference speech

    A collar (default 250 ms, the NIST convention) is applied either side of every
    reference boundary and excluded from scoring, because human annotators cannot
    place boundaries to better than roughly that precision and penalising a system
    for millisecond disagreement measures annotation noise rather than
    performance.

    Speaker mapping is chosen to minimise confusion, so a system is not penalised
    for arbitrary cluster naming.

USAGE
    python score_primock.py --result-dir <pipeline output xlsx> \\
        --prepared ${ASR_PRIMOCK_ROOT} --csv primock_results.csv
"""
import argparse
import re
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import jiwer
except ImportError:
    sys.exit("pip install jiwer pandas openpyxl numpy")

FRAME = 0.010     # 10 ms scoring grid
COLLAR = 0.250    # NIST convention

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


def hhmmss_to_sec(v):
    s = str(v).strip()
    if ":" in s:
        parts = s.split(":")
        try:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        except (ValueError, IndexError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


# ------------------------------------------------------------ reference ----
def read_rttm(path):
    segs = []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            segs.append((float(p[3]), float(p[3]) + float(p[4]), p[7]))
    return segs


def read_text(path):
    pairs, cur = [], None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^\s*([DP])\s*:\s*(.*)$", line.strip(), re.I)
        if m:
            cur = "doctor" if m.group(1).upper() == "D" else "patient"
            text = m.group(2)
        else:
            if cur is None:
                continue
            text = line
        pairs += [(w, cur) for w in norm(text).split()]
    return pairs


# ----------------------------------------------------------- hypothesis ----
def read_result(path):
    df = pd.read_excel(path)
    col = next((c for c in df.columns if "transcript" in c.lower()
                and "researcher" not in c.lower()), None)
    if col is None or "Speaker" not in df.columns:
        return [], []
    words, segs = [], []
    for _, r in df.iterrows():
        if pd.isna(r["Speaker"]) or pd.isna(r[col]):
            continue
        spk = str(r["Speaker"]).strip()
        ws = norm(r[col]).split()
        if not ws:
            continue
        words += [(w, spk) for w in ws]
        s, e = hhmmss_to_sec(r.get("Start")), hhmmss_to_sec(r.get("End"))
        if s is not None and e is not None and e > s:
            segs.append((s, e, spk))
    return words, segs


# ------------------------------------------------------------------ DER ----
def frame_labels(segs, n_frames):
    """One label set per frame. Overlap is represented, not collapsed."""
    lab = [set() for _ in range(n_frames)]
    for s, e, spk in segs:
        i0 = max(0, int(s / FRAME))
        i1 = min(n_frames, int(np.ceil(e / FRAME)))
        for i in range(i0, i1):
            lab[i].add(spk)
    return lab


def collar_mask(ref_segs, n_frames):
    """False wherever a frame lies within COLLAR of any reference boundary."""
    keep = np.ones(n_frames, dtype=bool)
    for s, e, _ in ref_segs:
        for b in (s, e):
            i0 = max(0, int((b - COLLAR) / FRAME))
            i1 = min(n_frames, int(np.ceil((b + COLLAR) / FRAME)))
            keep[i0:i1] = False
    return keep


def compute_der(ref_segs, hyp_segs):
    if not ref_segs:
        return None
    end = max(max(e for _, e, _ in ref_segs),
              max((e for _, e, _ in hyp_segs), default=0))
    n = int(np.ceil(end / FRAME)) + 1
    ref, hyp = frame_labels(ref_segs, n), frame_labels(hyp_segs, n)
    keep = collar_mask(ref_segs, n)

    ref_spk = sorted({s for _, _, s in ref_segs})
    hyp_spk = sorted({s for _, _, s in hyp_segs})
    if not hyp_spk:
        total = sum(len(ref[i]) for i in range(n) if keep[i]) * FRAME
        return {"der": 1.0, "missed": 1.0, "false_alarm": 0.0,
                "confusion": 0.0, "ref_speech": total, "scored_frames": int(keep.sum())}

    # choose the mapping that minimises confusion
    best = None
    for perm in permutations(hyp_spk, min(len(hyp_spk), len(ref_spk))):
        mapping = dict(zip(perm, ref_spk))
        missed = fa = conf = total = 0
        for i in range(n):
            if not keep[i]:
                continue
            r = ref[i]
            h = {mapping.get(x, x) for x in hyp[i]}
            total += len(r)
            missed += len(r - h)
            fa += len(h - r)
            # a confusion is reference speech covered by the wrong identity
            conf += min(len(r - h), len(h - r))
        if total == 0:
            continue
        # NIST counts confusion within missed+fa; subtract to avoid double count
        missed -= conf
        fa -= conf
        der = (missed + fa + conf) / total
        cand = {"der": der, "missed": missed / total, "false_alarm": fa / total,
                "confusion": conf / total, "ref_speech": total * FRAME,
                "scored_frames": int(keep.sum())}
        if best is None or cand["der"] < best["der"]:
            best = cand
    return best


# ----------------------------------------------------------------- WDER ----
def compute_wer_wder(ref_pairs, hyp_words):
    rw = [w for w, _ in ref_pairs]
    hw = [w for w, _ in hyp_words]
    if not rw or not hw:
        return None
    out = jiwer.process_words(" ".join(rw), " ".join(hw))
    aligned = []
    for ch in out.alignments[0]:
        if ch.type in ("equal", "substitute"):
            for k in range(ch.ref_end_idx - ch.ref_start_idx):
                aligned.append((ref_pairs[ch.ref_start_idx + k][1],
                                hyp_words[ch.hyp_start_idx + k][1]))
    if not aligned:
        return None
    ref_roles = sorted({r for r, _ in aligned})
    hyp_roles = sorted({h for _, h in aligned})
    cands = [{h: h for h in hyp_roles}]
    if len(ref_roles) == 2 and len(hyp_roles) == 2:
        cands += [{hyp_roles[0]: ref_roles[0], hyp_roles[1]: ref_roles[1]},
                  {hyp_roles[0]: ref_roles[1], hyp_roles[1]: ref_roles[0]}]
    wder = min(sum(1 for r, h in aligned if m.get(h, h) != r) / len(aligned)
               for m in cands)

    # ---- per-role substitution-plus-deletion rate ----------------------
    # The same block as final_score_fareez_speaker.py, so the two corpora are
    # measured identically. On Fareez the DOCTOR was transcribed 2.18 points
    # worse than the patient (10.31% vs 8.13%, CI [1.77, 2.59], 73.9% of
    # recordings). This is the replication test: a second country, real
    # clinicians rather than students, and published accent distributions.
    #
    # Insertions have no reference speaker and cannot be attributed to a role,
    # so they are excluded here but INCLUDED in the headline WER. These are
    # substitution-plus-deletion rates and are not comparable to it.
    #
    # Counting is done in ONE PASS over the alignment. Do not build parallel
    # reference and hypothesis lists and zip them: deletions append to the
    # reference side only, so after the first deletion every pair is offset and
    # the substitution count is meaningless.
    align = out.alignments[0]
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
                    if rw[i] != hw[ch.hyp_start_idx + k]:
                        err += 1
            elif ch.type == "delete":
                for k in range(ch.ref_end_idx - ch.ref_start_idx):
                    if ref_pairs[ch.ref_start_idx + k][1] == role:
                        n += 1
                        err += 1
        if n:
            per_role[role] = {"n": n, "err": err}

    # ---- speaker-attributed WER ----------------------------------------
    # A word counts as correct only if BOTH the word and its speaker label are
    # right - what a reader of the transcript actually gets. INSERTIONS ARE
    # INCLUDED, because the headline WER counts them; without them SA-WER can
    # score better than WER, which is indefensible for a stricter metric.
    # The mapping is the one chosen for WDER, on the same `aligned` list, so the
    # two cannot disagree about which cluster is which.
    best_map = min(cands, key=lambda m:
                   sum(1 for r, h in aligned if m.get(h, h) != r))
    sa_correct = 0
    for ch in align:
        if ch.type == "equal":
            for k in range(ch.ref_end_idx - ch.ref_start_idx):
                rspk = ref_pairs[ch.ref_start_idx + k][1]
                hspk = hyp_words[ch.hyp_start_idx + k][1]
                if best_map.get(hspk, hspk) == rspk:
                    sa_correct += 1
    sa_wer = ((len(rw) - sa_correct) + out.insertions) / len(rw)

    res = {"wer": out.wer, "wder": wder, "sa_wer": sa_wer,
           "sub": out.substitutions,
           "del": out.deletions, "ins": out.insertions,
           "ref_words": len(rw), "hyp_words": len(hw),
           "aligned": aligned, "hyp_roles": hyp_roles}
    for role, d in per_role.items():
        res[f"{role}_n"] = d["n"]
        res[f"{role}_err"] = d["err"]
        res[f"{role}_rate"] = d["err"] / d["n"]
    return res


# ------------------------------------------------- role rule (FROZEN) ------
def lexical_role(hyp_words):
    """
    The rule exactly as developed on Fareez. NOT retuned for this dataset -
    retuning would destroy the point of a held-out test.
    """
    text = {}
    for w, spk in hyp_words:
        text.setdefault(spk, []).append(w)
    if len(text) < 2:
        return {s: "doctor" for s in text}, 0.0
    sc = {}
    for spk, ws in text.items():
        blob = " ".join(ws)
        n = max(len(ws), 1)
        sc[spk] = (sum(blob.count(p) for p in INTERROGATIVE)
                   - sum(blob.count(p) for p in PATIENT_PHRASE)) / n * 100
    ranked = sorted(sc, key=sc.get, reverse=True)
    return ({ranked[0]: "doctor", ranked[1]: "patient"},
            sc[ranked[0]] - sc[ranked[1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--csv", default="primock_results.csv")
    ap.add_argument("--suffix", default="_large_v3_review",
                    help="review-sheet suffix. The pipeline names outputs after "
                         "the MODEL, so a large-v2 or best_combo run writes "
                         "'_large-v2_review' etc. Left hardcoded this script "
                         "would silently score nothing and exit.")
    a = ap.parse_args()

    prep = Path(a.prepared)
    rows = []
    for xlsx in sorted(Path(a.result_dir).glob(f"*{a.suffix}.xlsx")):
        name = xlsx.stem.split(a.suffix)[0]
        rttm_p, text_p = prep / "rttm" / f"{name}.rttm", prep / "text" / f"{name}.txt"
        if not rttm_p.exists() or not text_p.exists():
            continue

        ref_segs = read_rttm(rttm_p)
        ref_pairs = read_text(text_p)
        hyp_words, hyp_segs = read_result(xlsx)
        if not ref_pairs or not hyp_words:
            print(f"  {name}: empty after parsing - skipped")
            continue

        m = compute_wer_wder(ref_pairs, hyp_words)
        if not m:
            continue
        d = compute_der(ref_segs, hyp_segs) or {}

        # held-out test of the frozen rule
        truth = None
        if len(m["hyp_roles"]) == 2:
            cands = [{m["hyp_roles"][0]: "doctor", m["hyp_roles"][1]: "patient"},
                     {m["hyp_roles"][0]: "patient", m["hyp_roles"][1]: "doctor"}]
            truth = min(cands, key=lambda mp:
                        sum(1 for r, h in m["aligned"] if mp[h] != r))
        rule, margin = lexical_role(hyp_words)
        role_ok = (truth is not None and
                   all(rule.get(k) == truth.get(k) for k in truth))

        rows.append({"case": name, "wer": m["wer"], "wder": m["wder"],
                     "sa_wer": m.get("sa_wer"),
                     "doctor_n": m.get("doctor_n"), "doctor_err": m.get("doctor_err"),
                     "doctor_rate": m.get("doctor_rate"),
                     "patient_n": m.get("patient_n"), "patient_err": m.get("patient_err"),
                     "patient_rate": m.get("patient_rate"),
                     "der": d.get("der"), "missed": d.get("missed"),
                     "false_alarm": d.get("false_alarm"),
                     "confusion": d.get("confusion"),
                     "role_ok": role_ok, "role_margin": round(margin, 3),
                     "ref_words": m["ref_words"], "hyp_words": m["hyp_words"],
                     "ref_speech_s": round(d.get("ref_speech", 0), 1)})
        print(f"  {name:28s} WER {m['wer']*100:6.2f}%  WDER {m['wder']*100:5.2f}%  "
              f"DER {(d.get('der') or 0)*100:6.2f}%  role "
              f"{'ok' if role_ok else 'FAILED'}")

    if not rows:
        sys.exit("nothing scored")
    df = pd.DataFrame(rows)
    df.to_csv(a.csv, index=False)

    print(f"\n{'='*70}\nPRIMOCK57 - {len(df)} consultations\n{'='*70}")
    print(f"WER   mean {df.wer.mean()*100:6.2f}%   median {df.wer.median()*100:6.2f}%"
          f"   SD {df.wer.std()*100:5.2f}")
    print(f"WDER  mean {df.wder.mean()*100:6.2f}%   median {df.wder.median()*100:6.2f}%"
          f"   SD {df.wder.std()*100:5.2f}")
    if df.der.notna().any():
        print(f"DER   mean {df.der.mean()*100:6.2f}%   median {df.der.median()*100:6.2f}%"
              f"   SD {df.der.std()*100:5.2f}")
        print(f"        missed {df.missed.mean()*100:5.2f}%   "
              f"false alarm {df.false_alarm.mean()*100:5.2f}%   "
              f"confusion {df.confusion.mean()*100:5.2f}%")
        print(f"\n  DER uses a {COLLAR*1000:.0f} ms collar (NIST convention) and "
              f"minimises\n  speaker confusion over label mappings.")

    if "sa_wer" in df.columns and df.sa_wer.notna().any():
        gap = (df.sa_wer.mean() - df.wer.mean()) * 100
        print(f"\nSA-WER mean {df.sa_wer.mean()*100:6.2f}%   "
              f"(word AND speaker both correct; insertions included)")
        print(f"  exceeds WER by {gap:+5.2f} points - that gap IS the cost of "
              f"speaker mislabelling.")

    # ---- replication of the Fareez role asymmetry ------------------------
    if {"doctor_n", "patient_n"}.issubset(df.columns) and df.doctor_n.notna().any():
        sub = df.dropna(subset=["doctor_n", "patient_n"]).copy()
        sub = sub[(sub.doctor_n > 0) & (sub.patient_n > 0)]
        if len(sub):
            dn, de = int(sub.doctor_n.sum()), int(sub.doctor_err.sum())
            pn, pe = int(sub.patient_n.sum()), int(sub.patient_err.sum())
            micro = de / dn * 100 - pe / pn * 100
            sub["diff"] = (sub.doctor_err / sub.doctor_n
                           - sub.patient_err / sub.patient_n) * 100
            d_arr = sub["diff"].to_numpy()
            print(f"\nPER-ROLE RATE - replication of the Fareez asymmetry")
            print(f"  doctor   {de:6d} / {dn:7d}  = {de/dn*100:6.2f}%")
            print(f"  patient  {pe:6d} / {pn:7d}  = {pe/pn*100:6.2f}%")
            print(f"  gap (micro) {micro:+6.2f} points   "
                  f"(macro {d_arr.mean():+.2f}, SD {d_arr.std(ddof=1):.2f})")

            rng = np.random.default_rng(20260819)
            idx = rng.integers(0, len(d_arr), size=(10000, len(d_arr)))
            boot = d_arr[idx].mean(axis=1)
            lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
            print(f"  bootstrap 95% CI  [{lo:+.2f}, {hi:+.2f}]  "
                  f"({'includes' if lo <= 0 <= hi else 'excludes'} zero)")
            harder = int((d_arr > 0).sum())
            print(f"  doctor harder in {harder}/{len(d_arr)} "
                  f"({harder/len(d_arr)*100:.1f}%)")
            print(f"\n  Fareez, for comparison: doctor 10.31%, patient 8.13%,")
            print(f"  gap +2.18 [+1.77, +2.59], doctor harder in 73.9% of 272.")
            print("  A gap in the SAME direction here is a replication in a")
            print("  second country with practising clinicians. One in the")
            print("  OPPOSITE direction means the Fareez result is a property")
            print("  of that corpus - students playing patients - and must be")
            print("  reported as such, not as a general finding.")
            print("\n  These are substitution-plus-deletion rates: insertions have")
            print("  no reference speaker. Not comparable to the WER above.")

    print(f"\nROLE ASSIGNMENT - held-out test of the rule developed on Fareez")
    print(f"  lexical rule correct: {df.role_ok.mean()*100:5.1f}%  "
          f"({int(df.role_ok.sum())}/{len(df)})")
    thin = df[df.role_margin.abs() < 0.3]
    print(f"  decided on a thin margin (<0.3): {len(thin)}")
    print("\n  This is the first evaluation of the rule on data it was not built")
    print("  from. The Fareez figure was resubstitution accuracy; this is not.")

    print(f"\nSaved -> {a.csv}")


if __name__ == "__main__":
    main()
