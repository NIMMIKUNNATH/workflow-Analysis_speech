#!/usr/bin/env python3
"""
study.py - a systematic, paired parameter study over the Fareez OSCE dataset.

WHY THIS EXISTS
    The earlier work changed one parameter at a time on TWO recordings and
    compared single runs. That design cannot support conclusions, for a reason the
    five-file pilot made obvious: WER varies by +/-5 percentage points BETWEEN
    recordings (14.2% to 25.5%), while every effect being chased was 1-2 points.
    The signal was an order of magnitude smaller than the noise.

THE FIX IS A PAIRED DESIGN
    Every condition is run on the SAME fixed subset of recordings. Comparisons are
    then made per-recording - condition B minus condition A on recording X - so
    the large between-recording variance cancels out completely. What remains is
    the variance of the DIFFERENCE, which is small.

    This is why 40 paired recordings beats 272 unpaired ones for detecting a
    parameter effect, and why the subset must never change between conditions.

    Concretely: unpaired, with SD ~5 points, detecting a 1-point effect needs
    hundreds of recordings per arm. Paired, with difference-SD ~1-2 points, n=40
    detects it comfortably. The subset is drawn once, with a fixed seed, and
    written to disk so every condition and every future run uses the identical set.

WHAT IT PRODUCES
    - results/<condition>/         the review sheets for that condition
    - results/<condition>.csv      per-recording WER and WDER
    - study_log.csv                every condition, every recording, appended
    - a paired comparison table with mean difference, 95% CI and a paired t-test

USAGE
    python study.py --init          # draw the subset, write conditions.json
    python study.py --list          # show planned conditions
    python study.py --run baseline  # run one condition
    python study.py --run-all       # run everything not yet done
    python study.py --analyse       # paired comparisons against baseline

The baseline condition MUST be run first: every comparison is against it.
"""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    import jiwer
    from scipy import stats as scipy_stats
except ImportError:
    sys.exit("pip install jiwer pandas openpyxl scipy")

# ---------------------------------------------------------------- paths ----
# Everything lives under one root so the study is self-contained and portable.
# Configure roots with environment variables; portable defaults are local.
DATA_ROOT = Path(os.environ.get("ASR_DATA_ROOT", "./data"))
AUDIO_DIR = DATA_ROOT / "Audio Recordings"
REF_DIR   = DATA_ROOT / "Clean Transcripts"
CODE_DIR  = Path(os.environ.get("ASR_CODE_ROOT", Path(__file__).resolve().parent))
STUDY     = Path(os.environ.get("ASR_STUDY_ROOT", "./study"))
PIPELINE  = CODE_DIR / "single_large_v3.py"

# Working copies are written to the internal drive. Reading and writing across
# the WSL/Windows boundary is several times slower, and each condition rewrites
# 40 audio files - on an external drive that dominates the runtime.
SCRATCH = Path.home() / "osce_study_scratch"

SUBSET_FILE = STUDY / "subset.json"
CONDITIONS_FILE = STUDY / "conditions.json"
LOG_FILE = STUDY / "study_log.csv"

SUBSET_N = 40
SEED = 20260819

SPEAKER_PREFIX = {"D": "Student", "P": "Patient"}


# ============================================================ conditions ====
# Each condition is a set of edits to the SETTINGS block of single_large_v3.py.
# Only ONE factor differs from baseline in each, so any difference is attributable.
DEFAULT_CONDITIONS = {
    "baseline":        {},
    "beam1":           {"BEAM_SIZE": "1"},
    "beam3":           {"BEAM_SIZE": "3"},
    "beam8":           {"BEAM_SIZE": "8"},
    "vad_off":         {"VAD_FILTER": "False"},
    "model_medium":    {"MODEL_NAME": '"medium"'},
    "model_small":     {"MODEL_NAME": '"small"'},
    "model_largev2":   {"MODEL_NAME": '"large-v2"'},
    "compute_fp16":    {"COMPUTE_TYPE": '"float16"'},
    "merge_gap_030":   {"MERGE_GAP_SECONDS": "0.30"},
    "merge_gap_150":   {"MERGE_GAP_SECONDS": "1.50"},
    "min_interval_00": {"MIN_INTERVAL_DURATION": "0.00"},
    "min_interval_40": {"MIN_INTERVAL_DURATION": "0.40"},
}


# ================================================================ scoring ===
def norm(s):
    s = re.sub(r"[^a-z0-9' ]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def parse_reference(path):
    pairs, current = [], None
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^\s*([DP])\s*:\s*(.*)$", line, flags=re.I)
        if m:
            current, text = SPEAKER_PREFIX[m.group(1).upper()], m.group(2)
        else:
            if current is None:
                continue
            text = line
        pairs += [(w, current) for w in norm(text).split()]
    return pairs


def parse_result(path):
    df = pd.read_excel(path)
    col = next((c for c in df.columns if "transcript" in c.lower()
                and "researcher" not in c.lower()), None)
    if col is None or "Speaker" not in df.columns:
        return []
    out = []
    for _, r in df.iterrows():
        if pd.isna(r["Speaker"]) or pd.isna(r[col]):
            continue
        spk = str(r["Speaker"]).strip()
        out += [(w, spk) for w in norm(r[col]).split()]
    return out


def score_pair(ref_pairs, hyp_pairs):
    if not ref_pairs or not hyp_pairs:
        return None
    out = jiwer.process_words(" ".join(w for w, _ in ref_pairs),
                              " ".join(w for w, _ in hyp_pairs))
    aligned = []
    for ch in out.alignments[0]:
        if ch.type in ("equal", "substitute"):
            for k in range(ch.ref_end_idx - ch.ref_start_idx):
                aligned.append((ref_pairs[ch.ref_start_idx + k][1],
                                hyp_pairs[ch.hyp_start_idx + k][1]))
    if not aligned:
        return None
    refr = sorted({r for r, _ in aligned})
    hypr = sorted({h for _, h in aligned})
    cands = [{h: h for h in hypr}]
    if len(refr) == 2 and len(hypr) == 2:
        cands += [{hypr[0]: refr[0], hypr[1]: refr[1]},
                  {hypr[0]: refr[1], hypr[1]: refr[0]}]
    wder = min(sum(1 for r, h in aligned if m.get(h, h) != r) / len(aligned)
               for m in cands)
    return {"wer": out.wer, "wder": wder, "sub": out.substitutions,
            "del": out.deletions, "ins": out.insertions,
            "ref_words": len(ref_pairs), "hyp_words": len(hyp_pairs)}


# ================================================================== setup ===
def cmd_init():
    STUDY.mkdir(parents=True, exist_ok=True)
    if not AUDIO_DIR.is_dir():
        sys.exit(f"audio not found: {AUDIO_DIR}")

    usable = sorted(p.stem for p in AUDIO_DIR.glob("*.mp3")
                    if (REF_DIR / f"{p.stem}.txt").exists())
    print(f"{len(usable)} recordings have a matching transcript")

    # Stratify by clinical category so no category is missed, and seed the draw so
    # the subset is reproducible by anyone re-running the study.
    by_cat = {}
    for s in usable:
        by_cat.setdefault(re.match(r"^([A-Z]+)", s).group(1), []).append(s)

    rng = random.Random(SEED)
    subset, remaining = [], []
    for cat, items in sorted(by_cat.items()):
        take = max(1, round(SUBSET_N * len(items) / len(usable)))
        rng.shuffle(items)
        subset += items[:take]
        remaining += items[take:]
    rng.shuffle(remaining)
    while len(subset) < SUBSET_N and remaining:
        subset.append(remaining.pop())
    subset = sorted(subset[:SUBSET_N])

    SUBSET_FILE.write_text(json.dumps(
        {"seed": SEED, "n": len(subset), "drawn": datetime.now().isoformat(),
         "cases": subset}, indent=2))
    if not CONDITIONS_FILE.exists():
        CONDITIONS_FILE.write_text(json.dumps(DEFAULT_CONDITIONS, indent=2))

    print(f"\nsubset of {len(subset)}, stratified by category, seed {SEED}:")
    for cat in sorted(by_cat):
        n = sum(1 for s in subset if s.startswith(cat))
        print(f"  {cat:5s} {n:3d} of {len(by_cat[cat]):3d} available")
    print(f"\nwrote {SUBSET_FILE}\nwrote {CONDITIONS_FILE}")
    print("\nThe subset is now FIXED. Every condition uses these same recordings,\n"
          "which is what makes the paired comparison valid. Do not regenerate it\n"
          "mid-study - doing so invalidates every comparison already run.")


def load_subset():
    if not SUBSET_FILE.exists():
        sys.exit("run --init first")
    return json.loads(SUBSET_FILE.read_text())["cases"]


def load_conditions():
    if not CONDITIONS_FILE.exists():
        sys.exit("run --init first")
    raw = json.loads(CONDITIONS_FILE.read_text())
    # Keys beginning with "_" are comments and section markers in the conditions
    # file, not conditions. Without this they would each be run as an experiment
    # with no edits - silently producing duplicates of the baseline.
    return {k: v for k, v in raw.items()
            if not k.startswith("_") and isinstance(v, dict)}


# ==================================================================== run ===
def apply_settings(edits):
    """Rewrite the pipeline's SETTINGS block, returning the original text."""
    original = PIPELINE.read_text()
    text = original
    for key, value in edits.items():
        pat = re.compile(rf"^{re.escape(key)}\s*=.*$", flags=re.M)
        if not pat.search(text):
            PIPELINE.write_text(original)
            sys.exit(
                f"setting '{key}' does not exist in {PIPELINE.name}.\n"
                f"Conditions can only vary settings the pipeline actually defines.\n"
                f"Run  python add_settings.py  to add the extra decoding and VAD\n"
                f"settings, or remove this condition from conditions.json."
            )
        text = pat.sub(f"{key} = {value}", text, count=1)
    PIPELINE.write_text(text)
    return original


def cmd_run(name):
    conds = load_conditions()
    if name not in conds:
        sys.exit(f"unknown condition '{name}'. Known: {', '.join(conds)}")
    cases = load_subset()
    outdir = STUDY / "results" / name
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*68}\ncondition: {name}")
    print(f"edits: {conds[name] or 'none (baseline)'}")
    print(f"{len(cases)} recordings\n{'='*68}")

    original = apply_settings(conds[name])
    try:
        for i, case in enumerate(cases, 1):
            xlsx = outdir / f"{case}_large_v3_review.xlsx"
            if xlsx.exists():
                print(f"[{i}/{len(cases)}] {case} - already done")
                continue
            SCRATCH.mkdir(parents=True, exist_ok=True)
            work = SCRATCH / f"{case}.mp3"
            # copyfile, never copy: chmod is not permitted on Windows mounts and
            # shutil.copy fails AFTER writing the content.
            shutil.copyfile(AUDIO_DIR / f"{case}.mp3", work)
            print(f"[{i}/{len(cases)}] {case}", flush=True)
            r = subprocess.run([sys.executable, str(PIPELINE), str(work)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print("    FAILED:", r.stderr.strip().splitlines()[-1:])
            produced = SCRATCH / f"{case}_large_v3_review.xlsx"
            if produced.exists():
                shutil.copyfile(produced, xlsx)
            for junk in SCRATCH.glob(f"{case}*"):
                junk.unlink(missing_ok=True)
    finally:
        PIPELINE.write_text(original)           # always restore, even on Ctrl-C
        print("\npipeline settings restored")

    score_condition(name)


def score_condition(name):
    outdir = STUDY / "results" / name
    rows = []
    for xlsx in sorted(outdir.glob("*_large_v3_review.xlsx")):
        case = xlsx.stem.split("_large_v3")[0]
        ref = REF_DIR / f"{case}.txt"
        if not ref.exists():
            continue
        m = score_pair(parse_reference(ref), parse_result(xlsx))
        if m:
            rows.append({"condition": name, "case": case,
                         "category": re.match(r"^([A-Z]+)", case).group(1), **m})
    if not rows:
        print("nothing scored")
        return
    df = pd.DataFrame(rows)
    df.to_csv(STUDY / "results" / f"{name}.csv", index=False)

    log = pd.concat([pd.read_csv(LOG_FILE), df]) if LOG_FILE.exists() else df
    log = log.drop_duplicates(subset=["condition", "case"], keep="last")
    log.to_csv(LOG_FILE, index=False)

    print(f"\n{name}: n={len(df)}  "
          f"WER {df.wer.mean()*100:.2f}% (SD {df.wer.std()*100:.2f})  "
          f"WDER {df.wder.mean()*100:.2f}% (SD {df.wder.std()*100:.2f})")


def cmd_run_all():
    for name in load_conditions():
        outdir = STUDY / "results" / name
        done = len(list(outdir.glob("*_large_v3_review.xlsx"))) if outdir.exists() else 0
        if done >= len(load_subset()):
            print(f"skipping {name} - already complete")
            continue
        cmd_run(name)


# ================================================================ analyse ===
def cmd_analyse():
    if not LOG_FILE.exists():
        sys.exit("no results yet")
    log = pd.read_csv(LOG_FILE)
    if "baseline" not in log.condition.unique():
        sys.exit("run the baseline condition first - every comparison is against it")

    base = log[log.condition == "baseline"].set_index("case")
    print(f"\n{'='*78}\nPAIRED COMPARISON AGAINST BASELINE\n{'='*78}")
    print(f"baseline: n={len(base)}  WER {base.wer.mean()*100:.2f}%  "
          f"WDER {base.wder.mean()*100:.2f}%")
    print(f"\nbetween-recording SD is {base.wer.std()*100:.2f} points \u2014 which is why")
    print("comparisons must be paired. The column that matters is 'mean diff'.\n")

    hdr = f"{'condition':18s} {'n':>3s} {'WER':>7s} {'mean diff':>10s} {'95% CI':>16s} {'p':>8s}  verdict"
    print(hdr); print("-"*len(hdr))

    for cond in sorted(log.condition.unique()):
        if cond == "baseline":
            continue
        cur = log[log.condition == cond].set_index("case")
        shared = base.index.intersection(cur.index)
        if len(shared) < 3:
            print(f"{cond:18s} {len(shared):3d}  too few paired recordings")
            continue
        d = (cur.loc[shared, "wer"] - base.loc[shared, "wer"]) * 100
        t, p = scipy_stats.ttest_rel(cur.loc[shared, "wer"], base.loc[shared, "wer"])
        ci = 1.96 * d.std() / (len(d) ** 0.5)
        # A difference is only claimed when the CI excludes zero AND p < 0.05.
        if p < 0.05 and abs(d.mean()) > ci:
            verdict = "BETTER" if d.mean() < 0 else "worse"
        else:
            verdict = "no effect"
        print(f"{cond:18s} {len(shared):3d} {cur.loc[shared,'wer'].mean()*100:6.2f}% "
              f"{d.mean():+9.2f}  [{d.mean()-ci:+6.2f},{d.mean()+ci:+6.2f}] "
              f"{p:8.4f}  {verdict}")

    print(f"\n{'='*78}")
    print("'no effect' means the difference is indistinguishable from noise on this")
    print("sample - not that the parameter was untested. Report these as null results.")
    print(f"\nfull data: {LOG_FILE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()

    if a.init:
        cmd_init()
    elif a.list:
        cases = load_subset()
        print(f"subset: {len(cases)} recordings (seed {SEED})")
        for name, edits in load_conditions().items():
            d = STUDY / "results" / name
            done = len(list(d.glob("*_large_v3_review.xlsx"))) if d.exists() else 0
            print(f"  {name:18s} {done:3d}/{len(cases)}  {edits or 'baseline'}")
    elif a.run:
        cmd_run(a.run)
    elif a.run_all:
        cmd_run_all()
    elif a.analyse:
        cmd_analyse()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
