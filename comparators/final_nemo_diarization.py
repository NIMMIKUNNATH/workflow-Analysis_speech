#!/usr/bin/env python3
"""
run_msdd_subset.py

Diarization comparator: replaces pyannote with the NeMo multi-scale
diarization decoder (MSDD), holding the ASR fixed.

Includes a fallback to TitaNet-based Clustering Diarizer when speech duration
is too short or MSDD file pipeline fails.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY = Path(os.environ.get("ASR_STUDY_ROOT", "./study"))
SUBSET = STUDY / "subset.json"
RESULTS = STUDY / "results" / "msdd"
CACHE = Path(os.environ.get("ASR_CACHE_ROOT", "./cache"))
CODE = Path(os.environ.get("ASR_CODE_ROOT", REPO_ROOT))
SCRATCH = Path.home() / "msdd_scratch"

MERGE_GAP_SECONDS = 1.50
MIN_INTERVAL_DURATION = 0.15
NUM_SPEAKERS = 2


# --------------------------------------------------- verbatim row building --

def overlap_duration(s1, e1, s2, e2):
    return max(0.0, min(e1, e2) - max(s1, s2))


def merge_intervals(diar, gap, min_dur):
    if not diar:
        return []
    merged = [dict(diar[0])]
    for turn in diar[1:]:
        prev = merged[-1]
        same = turn["speaker"] == prev["speaker"]
        if same and (turn["start"] - prev["end"]) <= gap:
            prev["end"] = max(prev["end"], turn["end"])
        else:
            if (prev["end"] - prev["start"]) >= min_dur:
                merged.append(dict(turn))
            else:
                merged[-1] = dict(turn)
    return [m for m in merged if (m["end"] - m["start"]) >= min_dur]


def assign_words(words, turns):
    """Assign each cached word to the turn it overlaps most."""
    rows = []
    for t in turns:
        text = []
        for w in words:
            if overlap_duration(w["start"], w["end"], t["start"], t["end"]) > 0:
                best, best_ov = None, 0.0
                for u in turns:
                    ov = overlap_duration(w["start"], w["end"], u["start"], u["end"])
                    if ov > best_ov:
                        best, best_ov = u, ov
                if best is t:
                    text.append(w["text"].strip())
        if text:
            rows.append({"start": t["start"], "end": t["end"],
                         "speaker": t["speaker"], "text": " ".join(text)})
    return rows


# ----------------------------------------------------------------- MSDD ----

def run_msdd(wav_path, workdir):
    """Run NeMo MSDD on one wav; return [{start, end, speaker}] sorted by start."""
    from omegaconf import OmegaConf
    from nemo.collections.asr.models import NeuralDiarizer

    workdir = Path(workdir)
    (workdir / "data").mkdir(parents=True, exist_ok=True)

    manifest = workdir / "data" / "input_manifest.json"
    manifest.write_text(json.dumps({
        "audio_filepath": str(wav_path),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": NUM_SPEAKERS,
        "rttm_filepath": None,
        "uem_filepath": None,
    }) + "\n")

    cfg = OmegaConf.create({
        "num_workers": 0,
        "sample_rate": 16000,
        "batch_size": 1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "verbose": False,
        "diarizer": {
            "manifest_filepath": str(manifest),
            "out_dir": str(workdir),
            "oracle_vad": False,
            "collar": 0.25,
            "ignore_overlap": True,
            "vad": {
                "model_path": "vad_multilingual_marblenet",
                "parameters": {
                    "onset": 0.8, "offset": 0.6,
                    "pad_onset": 0.05, "pad_offset": -0.05,
                    "min_duration_on": 0.1, "min_duration_off": 0.15,
                    "window_length_in_sec": 0.15, "shift_length_in_sec": 0.01,
                    "smoothing": "median", "overlap": 0.5,
                    "filter_speech_first": True,
                },
            },
            "speaker_embeddings": {
                "model_path": "titanet_large",
                "parameters": {
                    "window_length_in_sec": [1.9, 1.2, 0.5],
                    "shift_length_in_sec": [0.95, 0.6, 0.25],
                    "multiscale_weights": [1, 1, 1],
                    "save_embeddings": True,  # Key Fix: Required for MSDD decoder step
                },
            },
            "clustering": {
                "parameters": {
                    "oracle_num_speakers": True,
                    "max_num_speakers": NUM_SPEAKERS,
                    "enhanced_count_thres": 80,
                    "max_rp_threshold": 0.25,
                    "sparse_search_volume": 30,
                    "maj_vote_spk_count": False,
                },
            },
            "msdd_model": {
                "model_path": "diar_msdd_telephonic",
                "parameters": {
                    "use_speaker_model_from_ckpt": True,
                    "infer_batch_size": 25,
                    "sigmoid_threshold": [0.7],
                    "seq_eval_mode": False,
                    "split_infer": False,
                    "diar_window_length": 50,
                    "overlap_infer_spk_limit": 5,
                    "save_embeddings": True,  # Key Fix: Propagate to subsegment config
                },
            },
        },
    })

    with torch.cuda.amp.autocast(enabled=False):
        model = NeuralDiarizer(cfg=cfg)
        model.msdd_model.to(torch.float32)
        model.diarize()

    rttm = next((workdir / "pred_rttms").glob("*.rttm"))
    turns = []
    for line in rttm.read_text().splitlines():
        p = line.split()
        if len(p) < 8 or p[0] != "SPEAKER":
            continue
        start, dur, spk = float(p[3]), float(p[4]), p[7]
        turns.append({"start": start, "end": start + dur, "raw": spk})

    labels = sorted({t["raw"] for t in turns})
    mapping = {}
    if len(labels) >= 1:
        mapping[labels[0]] = "Speaker A"
    if len(labels) >= 2:
        mapping[labels[1]] = "Speaker B"

    out = [{"start": t["start"], "end": t["end"],
            "speaker": mapping.get(t["raw"], "Speaker A")} for t in turns]
    return sorted(out, key=lambda x: x["start"])


# ---------------------------------------------------- FALLBACK CLUSTERING ----

def run_clustering_fallback(wav_path, workdir):
    """Fallback diarizer using basic TitaNet Clustering when MSDD fails."""
    from omegaconf import OmegaConf
    from nemo.collections.asr.models import ClusteringDiarizer

    workdir = Path(workdir) / "fallback"
    (workdir / "data").mkdir(parents=True, exist_ok=True)

    manifest = workdir / "data" / "input_manifest.json"
    manifest.write_text(json.dumps({
        "audio_filepath": str(wav_path),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": NUM_SPEAKERS,
        "rttm_filepath": None,
        "uem_filepath": None,
    }) + "\n")

    cfg = OmegaConf.create({
        "num_workers": 0,
        "sample_rate": 16000,
        "batch_size": 1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "verbose": False,
        "diarizer": {
            "manifest_filepath": str(manifest),
            "out_dir": str(workdir),
            "oracle_vad": False,
            "collar": 0.25,
            "ignore_overlap": True,
            "vad": {
                "model_path": "vad_multilingual_marblenet",
                "parameters": {
                    "onset": 0.8, "offset": 0.6,
                    "pad_onset": 0.05, "pad_offset": -0.05,
                    "min_duration_on": 0.1, "min_duration_off": 0.15,
                    "filter_speech_first": True,
                },
            },
            "speaker_embeddings": {
                "model_path": "titanet_large",
                "parameters": {
                    "window_length_in_sec": 1.5,
                    "shift_length_in_sec": 0.75,
                    "save_embeddings": False,
                },
            },
            "clustering": {
                "parameters": {
                    "oracle_num_speakers": True,
                    "max_num_speakers": NUM_SPEAKERS,
                },
            },
        },
    })

    ClusteringDiarizer(cfg=cfg).diarize()

    rttm = next((workdir / "pred_rttms").glob("*.rttm"))
    turns = []
    for line in rttm.read_text().splitlines():
        p = line.split()
        if len(p) < 8 or p[0] != "SPEAKER":
            continue
        start, dur, spk = float(p[3]), float(p[4]), p[7]
        turns.append({"start": start, "end": start + dur, "raw": spk})

    labels = sorted({t["raw"] for t in turns})
    mapping = {}
    if len(labels) >= 1:
        mapping[labels[0]] = "Speaker A"
    if len(labels) >= 2:
        mapping[labels[1]] = "Speaker B"

    out = [{"start": t["start"], "end": t["end"],
            "speaker": mapping.get(t["raw"], "Speaker A")} for t in turns]
    return sorted(out, key=lambda x: x["start"])


# ----------------------------------------------------------------- main ----

def write_sheet(rows, path):
    from openpyxl import Workbook

    def fmt(s):
        h = int(s // 3600); m = int((s % 3600) // 60); sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:06.3f}"

    wb = Workbook()
    ws = wb.active
    ws.append(["Start", "End", "Speaker", "Text", "", ""])
    for r in rows:
        ws.append([fmt(r["start"]), fmt(r["end"]), r["speaker"], r["text"], None, None])
    wb.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()

    cases = json.loads(SUBSET.read_text())["cases"]
    if a.limit:
        cases = cases[:a.limit]

    RESULTS.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("condition: msdd   diarizer: NeMo MSDD (diar_msdd_telephonic)")
    print(f"{len(cases)} recordings, {NUM_SPEAKERS} speakers fixed")
    print("Whisper words reused from cache; only the diarizer differs")
    print("=" * 68)

    method_counts = {"msdd": 0, "clustering": 0}

    if not a.score_only:
        t0 = time.time()
        for i, case in enumerate(cases, 1):
            final = RESULTS / f"{case}_large_v3_review.xlsx"
            json_out = RESULTS / f"{case}_msdd_diarization.json"

            if final.exists() and json_out.exists():
                print(f"[{i}/{len(cases)}] {case} - already done")
                try:
                    meta = json.load(open(json_out))
                    m = meta.get("diarization_method", "msdd") if isinstance(meta, dict) else "msdd"
                    method_counts[m] = method_counts.get(m, 0) + 1
                except Exception:
                    method_counts["msdd"] += 1
                continue

            wav = CACHE / f"{case}.wav"
            words_json = CACHE / f"{case}_large_v3_words.json"
            if not wav.exists() or not words_json.exists():
                print(f"[{i}/{len(cases)}] {case} - MISSING INPUT, skipped")
                continue

            print(f"[{i}/{len(cases)}] {case}", flush=True)
            method = "msdd"
            try:
                with tempfile.TemporaryDirectory(dir=SCRATCH) as wd:
                    try:
                        turns = run_msdd(wav, wd)
                    except Exception as e:
                        print(f"    --> MSDD pipeline error ({type(e).__name__}: {e}), triggering Clustering Diarizer fallback...")
                        turns = run_clustering_fallback(wav, wd)
                        method = "clustering"

                words = json.load(open(words_json))
                merged = merge_intervals(turns, MERGE_GAP_SECONDS,
                                         MIN_INTERVAL_DURATION)
                rows = assign_words(words, merged)
                if not rows:
                    print("    no rows produced")
                    continue

                write_sheet(rows, final)

                output_payload = {
                    "diarization_method": method,
                    "turns": turns
                }
                json.dump(output_payload, open(json_out, "w"), indent=1)

                method_counts[method] += 1
                print(f"    --> Completed via {method}")

            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}")

        print(f"\nelapsed {(time.time() - t0)/60:.1f} min")
        print("=" * 68)
        print("DIARIZATION METHOD SPLIT")
        print(f"MSDD: {method_counts['msdd']} | Clustering Fallback: {method_counts['clustering']}")
        print("=" * 68)

    n = len(list(RESULTS.glob("*_large_v3_review.xlsx")))
    print(f"\nsheets in msdd: {n} of {len(cases)}")
    if n == 0:
        sys.exit("nothing to score")

    sys.argv = ["x"]
    sys.path.insert(0, str(CODE))
    import study
    study.score_condition("msdd")


if __name__ == "__main__":
    main()
