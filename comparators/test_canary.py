#!/usr/bin/env python3
"""
test_canary.py - swap Whisper for NVIDIA Canary-1b-v2, holding everything else fixed.

WHY THIS EXISTS
    Eleven Whisper-side changes have been tested on this audio and all eleven were
    worse or neutral. Every one of them tuned Whisper. This is the first test of a
    different ASR model entirely.

    Canary-1b-v2 scores 8.9% WER on AMI where whisper-large-v3-turbo scores 16.13%.
    AMI is multi-speaker, single distant microphone, spontaneous - the closest
    public benchmark to an OSCE room recording. Canary is also an encoder-decoder
    trained differently from Whisper, so its failure modes differ: notably it does
    not hallucinate through silence the way Whisper does.

WHAT IS HELD CONSTANT
    Diarization is NOT re-run. This reuses the pyannote turns already cached in
    <stem>_diarization.json, and the same merge + dominant-speaker row building
    that single_large_v3.py uses (copied verbatim). So the ONLY thing that differs
    from the 15.43% baseline run is which model produced the words. If the score
    moves, the ASR moved it.

WHAT THIS DOES NOT TOUCH
    single_large_v3.py is not modified or imported. This writes its own xlsx and
    its own words JSON. A failure here cannot affect the working pipeline.

USAGE
    python test_canary.py --stem ${ASR_STUDY_ROOT}/Video/Male_Female

    Then score it exactly as any other run:
    python score_run.py <stem>_canary_review.xlsx --ref <the reviewed sheet> \\
        --name MF_canary -p model=canary-1b-v2 -p diarizer=pyannote

REALISTIC EXPECTATION
    The 8.9% AMI figure is British/European meeting speech. These recordings are
    UAE-accented clinical dialogue, so the gap will likely shrink. A good outcome
    is 12-14% against Whisper's 15.43%; a drop to 9% would be surprising. And no
    ASR model recovers the passages the researcher marked "skipped by all models"
    or masked by the door closing - those are gone whichever model reads them.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from row_assignment import assign_words_to_intervals

MODEL_NAME = "nvidia/canary-1b-v2"

# Copied verbatim from single_large_v3.py so row building is identical.
MERGE_GAP_SECONDS = 1.50
MIN_INTERVAL_DURATION = 0.15
SPEAKER_MAP = {"SPEAKER_00": "Speaker A", "SPEAKER_01": "Speaker B"}


# ------------------------------------ verbatim from single_large_v3.py -------
def overlap_duration(s1, e1, s2, e2):
    return max(0.0, min(e1, e2) - max(s1, s2))


def merge_diarization_intervals(diar, merge_gap, min_duration):
    if not diar:
        return []
    diar = sorted(diar, key=lambda x: (x["start"], x["end"]))
    merged = []
    current = dict(start=diar[0]["start"], end=diar[0]["end"],
                   speaker=diar[0]["speaker"])
    for seg in diar[1:]:
        if seg["speaker"] == current["speaker"] and \
                (seg["start"] - current["end"]) <= merge_gap:
            current["end"] = max(current["end"], seg["end"])
        else:
            if current["end"] - current["start"] >= min_duration:
                merged.append(current)
            current = dict(start=seg["start"], end=seg["end"],
                           speaker=seg["speaker"])
    if current["end"] - current["start"] >= min_duration:
        merged.append(current)
    return merged


def assign_overlap(words, diar):
    out = []
    for w in words:
        best_spk, best_ov = "UNKNOWN", 0.0
        for seg in diar:
            ov = overlap_duration(w["start"], w["end"], seg["start"], seg["end"])
            if ov > best_ov:
                best_ov, best_spk = ov, seg["speaker"]
        out.append(dict(start=w["start"], end=w["end"], speaker=best_spk,
                        text=w["text"]))
    return out


def build_rows(shared_diar, assigned):
    rows = []
    interval_words, _ = assign_words_to_intervals(assigned, shared_diar)
    for iv, sel in zip(shared_diar, interval_words):
        s, e = iv["start"], iv["end"]
        if not sel:
            continue
        totals = {}
        for w in sel:
            totals[w["speaker"]] = totals.get(w["speaker"], 0.0) + \
                overlap_duration(w["start"], w["end"], s, e)
        rows.append(dict(start=s, end=e,
                         speaker=max(totals, key=totals.get),
                         transcript=" ".join(w["text"].strip()
                                             for w in sel).strip()))
    return rows


def hhmmss(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    return f"{h:02d}:{m:02d}:{sec % 60:06.3f}"


# ------------------------------------------------------------------ canary --
def transcribe_canary(wav_path, batch_size=1):
    """
    Canary handles long audio itself: dynamic chunking with 1-second overlap is
    enabled automatically for a single file, so the 12.5-minute recording does not
    need manual splitting.

    timestamps=True is required - without word timings there is nothing to align
    against the diarization turns. Note that NeMo issue #14877 reports timestamp
    extraction failing on some builds; if that happens the error is caught below
    and reported plainly rather than half-succeeding.
    """
    from nemo.collections.asr.models import EncDecMultiTaskModel

    print(f"Loading {MODEL_NAME} (first run downloads ~4 GB)...")
    model = EncDecMultiTaskModel.from_pretrained(MODEL_NAME)
    model.eval()

    cfg = model.cfg.decoding
    cfg.beam.beam_size = 1          # NVIDIA's documented default for Canary
    model.change_decoding_strategy(cfg)

    print("Transcribing (dynamic chunking, this takes a few minutes)...")
    t0 = time.time()
    results = model.transcribe(
        audio=[str(wav_path)],
        batch_size=batch_size,
        source_lang="en",
        target_lang="en",   # same as source = transcribe, not translate
        pnc="yes",          # must be a STRING - the boolean True is rejected
        timestamps=True,
    )
    elapsed = time.time() - t0
    print(f"Transcription time: {elapsed:.1f} sec")

    r = results[0]
    words = []
    ts = getattr(r, "timestamp", None)
    if not ts or "word" not in ts:
        raise RuntimeError(
            "Canary returned no word timestamps. This is NeMo issue #14877 on some "
            "builds. Without word timings the output cannot be aligned to the "
            "diarization turns, so this test cannot proceed.\n"
            "Try: pip install -U nemo_toolkit['asr']"
        )
    for w in ts["word"]:
        words.append({"start": float(w["start"]), "end": float(w["end"]),
                      "text": " " + str(w["word"]).strip()})

    del model
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    return words, elapsed


# --------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True,
                    help="path without extension, e.g. .../Video/Male_Female")
    args = ap.parse_args()

    stem = Path(args.stem)
    wav = stem.with_suffix(".wav")
    diar_p = stem.with_name(f"{stem.name}_diarization.json")

    if not wav.exists():
        sys.exit(f"Missing {wav} - run single_large_v3.py on this video first.")
    if not diar_p.exists():
        sys.exit(f"Missing {diar_p} - run single_large_v3.py on this video first.")

    diar = json.load(open(diar_p))
    if isinstance(diar, dict):
        diar = diar.get("segments", diar.get("diarization", []))
    print(f"Reusing {len(diar)} pyannote turns (diarization NOT re-run)")

    words, elapsed = transcribe_canary(wav)
    print(f"Words produced: {len(words)}")
    print("  (Whisper's best run on this file produced 1922)")

    json.dump(words, open(stem.with_name(f"{stem.name}_canary_words.json"), "w"))

    merged = merge_diarization_intervals(diar, MERGE_GAP_SECONDS,
                                         MIN_INTERVAL_DURATION)
    rows = build_rows(merged, assign_overlap(words, merged))
    print(f"Turns {len(diar)} -> merged {len(merged)} -> rows {len(rows)}")

    df = pd.DataFrame([{
        "Start": hhmmss(r["start"]),
        "End": hhmmss(r["end"]),
        "Speaker": SPEAKER_MAP.get(r["speaker"], r["speaker"]),
        "large-v3 transcript": r["transcript"],   # column name kept for score_run
        "Researcher-verified speaker": "",
        "Researcher-corrected transcript": "",
    } for r in rows if r["transcript"]])

    out = stem.with_name(f"{stem.name}_canary_review.xlsx")
    df.to_excel(out, index=False)
    print(f"\nWrote {len(df)} rows -> {out}")
    print("\nNow score it:")
    print(f"  python score_run.py {out} \\\n"
          f"    --ref <your reviewed sheet> \\\n"
          f"    --name canary -p model=canary-1b-v2 -p diarizer=pyannote")


if __name__ == "__main__":
    main()
