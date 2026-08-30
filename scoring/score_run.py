#!/usr/bin/env python3
"""
score_run.py - score an OSCE pipeline run and append it to an experiment log.

WHY THIS IS A SEPARATE FILE
    single_large_v3.py works and scores 5.09% / 8.90%. It stays untouched. This
    reads its output sheet, scores it, and records the parameters that produced
    it. Nothing here can break a run.

WHAT IT DOES
    1. Scores a result sheet against the researcher-verified sheet
    2. Appends parameters + scores as one row to runs_log.csv
    3. Prints the full log sorted by score, so "the best parameters" is a lookup
       rather than something you remember

USAGE
    python score_run.py RESULT.xlsx --ref REVIEWED.xlsx --note "merge_gap=0.4"

    Any parameter can be recorded with -p:
    python score_run.py out.xlsx --ref rev.xlsx \\
        -p diarizer=pyannote -p merge_gap=0.40 -p min_interval=0.15

    Show the log without scoring anything:
    python score_run.py --log

HOW SCORING WORKS
    Row-by-row joining does not work, because different settings put segment
    boundaries in different places - row 12 of one sheet is not row 12 of
    another. So the researcher's labels are treated as what they are: a function
    from TIME to speaker. Every row of the result is scored against whichever
    role she assigned at those timestamps, weighted by overlap duration.

    Validated: feeding single_large_v3.py's own output back through this returns
    5.09% speaker error and 8.90% WER, matching the hand-computed baseline.

A CAVEAT WORTH REMEMBERING
    70% of the reference rows are byte-identical to single_large_v3.py's output,
    and 53% of reference words are verbatim from it - the researcher corrected
    that transcript rather than transcribing from scratch. So WER is biased in
    favour of pipelines that produce similar wording. A different-but-equally-
    valid transcript is penalised for differing. Treat WER as a comparison
    between runs, not as an absolute measure of transcript quality.
"""
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    import jiwer
except ImportError:
    sys.exit("pip install jiwer pandas openpyxl")


BASELINE = {"speaker_err": 0.0509, "wer": 0.0890, "label": "single_large_v3.py"}
LOG_NAME = "runs_log.csv"


# ------------------------------------------------------------------ helpers --
def to_ms(t):
    """'00:01:23.400' -> milliseconds, or None if the cell has no timestamp.

    A reviewed sheet can contain rows with no Start/End at all - the researcher
    adds them to flag speech the pipeline missed entirely ("Portion completely
    missing: ..."). Those rows carry real information but cannot be scored by
    time, so they are skipped rather than crashing the run.
    """
    s = str(t)
    if ":" not in s:
        return None
    h, m, sec = s.split(":")
    return int((int(h) * 3600 + int(m) * 60 + float(sec)) * 1000)


def norm(s):
    s = re.sub(r"[^a-z0-9' ]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def pick_column(df, *musts, absent=()):
    for c in df.columns:
        lc = c.lower()
        if all(m in lc for m in musts) and not any(a in lc for a in absent):
            return c
    return None


# ------------------------------------------------------------------ scoring --
def load_reference(path):
    rev = pd.read_excel(path)
    spk_col = pick_column(rev, "researcher", "speaker")
    txt_col = pick_column(rev, "researcher", "transcript")
    if not spk_col or not txt_col:
        sys.exit(f"{path}: needs 'Researcher-verified speaker' and "
                 f"'Researcher-corrected transcript' columns.\n"
                 f"Found: {list(rev.columns)}")

    intervals, skipped = [], 0
    for _, r in rev.iterrows():
        if pd.isna(r[spk_col]):
            continue
        s_ms, e_ms = to_ms(r["Start"]), to_ms(r["End"])
        if s_ms is None or e_ms is None:
            skipped += 1
            continue
        intervals.append((s_ms, e_ms, str(r[spk_col]).strip()))
    if skipped:
        print(f"  note: {skipped} reviewed row(s) have no timestamps "
              f"(missing-speech notes) - excluded from speaker scoring")
    text = " ".join(norm(x) for x in rev[txt_col].dropna())
    text = re.sub(r"\s+", " ", text).strip()

    if not intervals:
        sys.exit(f"{path}: no verified speaker labels found - is it filled in?")
    return intervals, text


def load_result(path):
    df = pd.read_excel(path)
    txt_col = pick_column(df, "transcript", absent=("researcher",))
    if txt_col is None or "Speaker" not in df.columns:
        sys.exit(f"{path}: needs a 'Speaker' column and a transcript column.\n"
                 f"Found: {list(df.columns)}")
    out = []
    for _, r in df.iterrows():
        if pd.isna(r["Speaker"]):
            continue
        s_ms, e_ms = to_ms(r["Start"]), to_ms(r["End"])
        if s_ms is not None and e_ms is not None:
            out.append((s_ms, e_ms, str(r["Speaker"]).strip(), r[txt_col]))
    return out



# Reviewed sheets do not only contain the two participants. This one also uses
# "Both" (overlapping speech, where either answer is arguably right) and
# "Irrelevant" (the off-camera timer voice, room noise). Neither can fairly be
# scored against a diarizer that must pick ONE speaker, so they are excluded and
# the excluded time is reported. Labels are also stripped - one sheet contains
# both "Student" and "Student " with a trailing space.
NON_SPEAKER_LABELS = {"both", "irrelevant", "unknown", "n/a", "none", ""}


def scorable_roles(intervals):
    """The participant roles, longest-talking first, excluding non-speaker tags."""
    totals = {}
    for s, e, role in intervals:
        if role.strip().lower() in NON_SPEAKER_LABELS:
            continue
        totals[role.strip()] = totals.get(role.strip(), 0) + (e - s)
    return sorted(totals, key=totals.get, reverse=True)[:2], totals


def score(segments, intervals, ref_text):
    roles, _ = scorable_roles(intervals)
    intervals = [(s_, e_, r_.strip()) for s_, e_, r_ in intervals
                 if r_.strip() in roles]
    preds = sorted({s[2] for s in segments})

    def accuracy(mapping):
        ok = total = 0
        for s, e, role, _ in segments:
            for gs, ge, grole in intervals:
                ov = min(e, ge) - max(s, gs)
                if ov <= 0:
                    continue
                total += ov
                if mapping.get(role, role) == grole:
                    ok += ov
        return (ok / total if total else 0.0), total

    # Speaker IDs are arbitrary - "Speaker 0" may be either person. Try both
    # orientations and take the better; a run is not penalised for naming.
    candidates = [{p: p for p in preds}]
    if len(preds) == 2 and len(roles) == 2:
        candidates += [{preds[0]: roles[0], preds[1]: roles[1]},
                       {preds[0]: roles[1], preds[1]: roles[0]}]

    best_map, best_acc, scored = None, -1.0, 0
    for m in candidates:
        a, t = accuracy(m)
        if a > best_acc:
            best_map, best_acc, scored = m, a, t

    hyp = " ".join(norm(s[3]) for s in segments if norm(s[3]))
    out = jiwer.process_words(ref_text, hyp)

    return {
        "speaker_err": round(1 - best_acc, 4),
        "wer": round(out.wer, 4),
        "sub": out.substitutions,
        "del": out.deletions,
        "ins": out.insertions,
        "rows": len(segments),
        "words": len(hyp.split()),
        "ref_words": len(ref_text.split()),
        "scored_s": round(scored / 1000, 1),
        "rows_under_2s": sum(1 for s, e, _, _ in segments if e - s < 2000),
        "role_flipped": best_map != {p: p for p in preds},
    }


# ---------------------------------------------------------------------- log --
def append_log(log_path, row):
    df = pd.DataFrame([row])
    if log_path.exists():
        df = pd.concat([pd.read_csv(log_path), df], ignore_index=True)
    df.to_csv(log_path, index=False)
    return df


def show_log(log_path):
    if not log_path.exists():
        print(f"No log yet at {log_path}")
        return
    df = pd.read_csv(log_path)
    if df.empty:
        print("Log is empty.")
        return

    # Rank by both metrics together. Neither alone tells the story: a run can win
    # on WER and lose on speaker labels, which is exactly what happened on 17 Aug.
    df["combined"] = df["speaker_err"] + df["wer"]
    df = df.sort_values("combined")

    cols = ["run", "speaker_err", "wer", "rows", "words", "note"]
    cols = [c for c in cols if c in df.columns]

    print(f"\n{'='*78}\nEXPERIMENT LOG - {len(df)} runs, best first\n{'='*78}")
    view = df[cols].copy()
    for c in ("speaker_err", "wer"):
        if c in view:
            view[c] = (view[c] * 100).map(lambda v: f"{v:.2f}%")
    print(view.to_string(index=False))

    best = df.iloc[0]
    print(f"\nBEST SO FAR: {best.get('run', '?')}")
    print(f"  speaker error {best['speaker_err']*100:.2f}%  |  "
          f"WER {best['wer']*100:.2f}%")
    print(f"  baseline ({BASELINE['label']}): "
          f"{BASELINE['speaker_err']*100:.2f}% / {BASELINE['wer']*100:.2f}%")

    params = [c for c in df.columns
              if c not in cols + ["combined", "timestamp", "sub", "del", "ins",
                                  "ref_words", "scored_s", "rows_under_2s",
                                  "role_flipped"]]
    if params:
        print("\n  parameters of the best run:")
        for p in params:
            if pd.notna(best.get(p)):
                print(f"    {p} = {best[p]}")

    # A parameter that never varies teaches you nothing. Say so.
    varying = [p for p in params if df[p].nunique(dropna=True) > 1]
    if varying:
        print(f"\n  parameters actually varied across runs: {', '.join(varying)}")
    elif params:
        print("\n  !! no parameter has been varied yet - every run used the same "
              "settings,\n     so the log cannot tell you what helped.")


# --------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result", nargs="?", help="result .xlsx to score")
    ap.add_argument("--ref", help="researcher-reviewed .xlsx (ground truth)")
    ap.add_argument("--name", help="short name for this run")
    ap.add_argument("--note", default="", help="what you changed this time")
    ap.add_argument("-p", "--param", action="append", default=[],
                    metavar="KEY=VALUE", help="parameter to record (repeatable)")
    ap.add_argument("--log-dir", default=".", help="where runs_log.csv lives")
    ap.add_argument("--log", action="store_true", help="just print the log")
    args = ap.parse_args()

    log_path = Path(args.log_dir) / LOG_NAME

    if args.log or not args.result:
        show_log(log_path)
        return

    if not args.ref:
        sys.exit("--ref is required: the researcher-reviewed sheet is the "
                 "ground truth everything is measured against.")

    intervals, ref_text = load_reference(args.ref)
    segments = load_result(args.result)
    m = score(segments, intervals, ref_text)

    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "run": args.name or Path(args.result).stem, **m, "note": args.note}
    # A user parameter must never overwrite a measured value. `-p rows=turn`
    # would otherwise silently replace the row COUNT with the word "turn", and
    # the log would quietly lie to you.
    reserved = set(row)
    for p in args.param:
        if "=" not in p:
            sys.exit(f"--param needs KEY=VALUE, got: {p}")
        k, v = p.split("=", 1)
        k = k.strip()
        if k in reserved:
            print(f"  note: parameter '{k}' clashes with a measured field - "
                  f"recording it as 'p_{k}'")
            k = f"p_{k}"
        row[k] = v.strip()

    print(f"\n{'='*78}\n{row['run']}\n{'='*78}")
    for key, label in (("speaker_err", "Speaker error (time-weighted)"),
                       ("wer", "WER                          ")):
        d = m[key] - BASELINE[key]
        verdict = "BETTER" if d < -0.001 else ("worse" if d > 0.001 else "same")
        print(f"{label}: {m[key]*100:6.2f}%   baseline {BASELINE[key]*100:5.2f}%"
              f"   {abs(d)*100:5.2f} pt {verdict}")
    print(f"\n  rows {m['rows']} ({m['rows_under_2s']} under 2 s)  |  "
          f"words {m['words']} of {m['ref_words']}")
    print(f"  sub {m['sub']}  del {m['del']}  ins {m['ins']}  |  "
          f"scored over {m['scored_s']}s")
    if m["role_flipped"]:
        print("  !! Student/Patient were assigned the wrong way round; scoring "
              "corrected it.")
    if m["words"] < 0.9 * m["ref_words"]:
        print(f"  !! {m['ref_words'] - m['words']} words missing - fix the ASR "
              "before drawing any conclusion about speaker labels.")

    append_log(log_path, row)
    print(f"\nLogged to {log_path}")
    show_log(log_path)


if __name__ == "__main__":
    main()
