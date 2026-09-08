#!/usr/bin/env python3
"""
run_min_interval_272.py
-----------------------
Runs the minimum-retained-interval sensitivity analysis on all 272 recordings
and saves hypothesis transcripts for insertion-side negation analysis.

Reference configuration is transcribed from Supplementary Table S2 and Methods
3.4. VERIFY THESE against your original run before trusting the output --
they are my reading of the manuscript, not your code.

  model                    large-v3
  compute_type             int8_float16
  beam_size                5
  patience                 1.0
  length_penalty           1.0
  repetition_penalty       1.0
  no_repeat_ngram_size     0
  no_speech_threshold      0.60
  log_prob_threshold       -1.0
  compression_ratio_thr    2.4
  condition_on_prev_text   True
  temperature              (0, 0.2, 0.4, 0.6, 0.8, 1.0)   fallback schedule
  VAD                      on, threshold 0.50,
                           min_silence_duration_ms 2000, speech_pad_ms 400
  merge gap                1.00 s
  min retained interval    0.10 s (baseline); swept over 0 / 0.10 / 0.25 / 0.40

Diarization runs ONCE per recording and is cached. Minimum retained interval is
post-processing over turns, so all four levels reuse one transcription and one
RTTM. That is what makes 272 recordings feasible.

  pip install faster-whisper pyannote.audio soundfile

export HF_TOKEN="replace-with-authorized-token"
python run_min_interval_272.py --audio /path/to/audio \
    --out /path/to/output --limit 5
  python run_min_interval_272.py --audio "..." --out ... --limit 40  # dev subset check
  python run_min_interval_272.py --audio "..." --out ...             # all 272
"""

import argparse
import datetime
import glob
import json
import os
import time
import traceback

from row_assignment import assign_words_to_intervals

REFERENCE = dict(
    model="large-v3",
    language="en",
    compute_type="int8_float16",
    beam_size=5,
    patience=1.0,
    length_penalty=1.0,
    repetition_penalty=1.0,
    no_repeat_ngram_size=0,
    no_speech_threshold=0.60,
    log_prob_threshold=-1.0,
    compression_ratio_threshold=2.4,
    condition_on_previous_text=True,
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    vad_filter=True,
    vad_threshold=0.50,
    vad_min_silence_ms=2000,
    vad_speech_pad_ms=400,
    merge_gap=1.00,
)

# matched to code/frozen/single_large_v3.py
DIAR_MODEL = "pyannote/speaker-diarization-community-1"
NUM_SPEAKERS = 2

_WHISPER = None
_DIAR = None


def load_audio(path, target_sr=16000):
    """
    Decode to mono float32 at target_sr WITHOUT torchcodec, which is broken in
    this environment. Returns (numpy_1d, sr). soundfile handles wav/flac; for
    anything else we fall back to PyAV, which faster-whisper already depends on.
    """
    import numpy as np
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        data = data.mean(axis=1)
    except Exception:
        import av
        cont = av.open(path)
        stream = cont.streams.audio[0]
        sr = stream.rate
        chunks = []
        for frame in cont.decode(stream):
            arr = frame.to_ndarray()
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            chunks.append(arr.astype("float32"))
        cont.close()
        data = np.concatenate(chunks) if chunks else np.zeros(0, "float32")
        if data.size and abs(data).max() > 1.5:      # integer PCM
            data = data / 32768.0

    if sr != target_sr and data.size:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(sr), int(target_sr))
        data = resample_poly(data, target_sr // g, sr // g).astype("float32")
        sr = target_sr
    return data, sr


def get_whisper(model_name, compute_type, device):
    global _WHISPER
    if _WHISPER is None:
        from faster_whisper import WhisperModel
        print(f"[load] faster-whisper {model_name} ({compute_type}) on {device}")
        _WHISPER = WhisperModel(model_name, device=device, compute_type=compute_type)
    return _WHISPER


def get_diarizer(device):
    global _DIAR
    if _DIAR is None:
        import torch
        from pyannote.audio import Pipeline
        tok = os.environ.get("HF_TOKEN")
        if not tok:
            raise SystemExit(
                "HF_TOKEN is unset or still the placeholder.\n"
                "  Get one at https://huggingface.co/settings/tokens and accept\n"
                "  the model terms at huggingface.co/pyannote/speaker-diarization-3.1\n"
                "  then set the HF_TOKEN environment variable for this process")
        print(f"[load] pyannote {DIAR_MODEL}")
        try:                                    # pyannote.audio >= 4.x
            _DIAR = Pipeline.from_pretrained(DIAR_MODEL, token=tok)
        except TypeError:                       # pyannote.audio 3.x
            _DIAR = Pipeline.from_pretrained(DIAR_MODEL, use_auth_token=tok)
        if device == "cuda":
            _DIAR.to(torch.device("cuda"))
    return _DIAR


def diarize(audio_path, rttm_out, device, wav=None, sr=16000):
    import torch
    pipe = get_diarizer(device)
    if wav is None:
        wav, sr = load_audio(audio_path)
    # in-memory waveform bypasses the broken torchcodec decoder
    result = pipe({"waveform": torch.from_numpy(wav).unsqueeze(0),
                   "sample_rate": sr},
                  num_speakers=NUM_SPEAKERS)

    # pyannote.audio 4.x returns DiarizeOutput; 3.x returns an Annotation
    # frozen/single_large_v3.py uses result.exclusive_speaker_diarization,
    # in which each instant is assigned to exactly one speaker. Using
    # speaker_diarization instead changes the turn structure and therefore
    # changes which intervals the minimum-interval filter removes.
    ann = result
    for attr in ("exclusive_speaker_diarization", "speaker_diarization",
                 "diarization", "annotation"):
        if hasattr(result, attr):
            ann = getattr(result, attr)
            break
    if not hasattr(ann, "itertracks"):
        raise RuntimeError(
            f"could not extract an Annotation from {type(result).__name__}; "
            f"attributes: {[a for a in dir(result) if not a.startswith('_')][:20]}")

    turns = []
    for seg, _, spk in ann.itertracks(yield_label=True):
        turns.append(dict(start=float(seg.start), end=float(seg.end),
                          speaker=str(spk)))

    # write RTTM ourselves -- 4.x moved write_rttm off the result object
    if not turns:
        raise RuntimeError("diarization produced no turns")

    # write atomically: temp file then rename, so an interrupted run can never
    # leave a truncated RTTM that a later run would silently trust
    uri = os.path.splitext(os.path.basename(audio_path))[0]
    tmp = rttm_out + ".tmp"
    with open(tmp, "w") as fh:
        for t in turns:
            fh.write(f"SPEAKER {uri} 1 {t['start']:.3f} "
                     f"{t['end'] - t['start']:.3f} <NA> <NA> "
                     f"{t['speaker']} <NA> <NA>\n")
    os.replace(tmp, rttm_out)
    return turns


def rttm_is_valid(path):
    """A cached RTTM is only usable if it actually contains SPEAKER lines.
    A previous crash could have truncated the file to zero bytes."""
    try:
        if os.path.getsize(path) == 0:
            return False
        for line in open(path):
            if line.startswith("SPEAKER"):
                return True
    except OSError:
        return False
    return False


def load_rttm(path):
    turns = []
    for line in open(path):
        p = line.split()
        if len(p) > 7 and p[0] == "SPEAKER":
            turns.append(dict(start=float(p[3]),
                              end=float(p[3]) + float(p[4]),
                              speaker=p[7]))
    return sorted(turns, key=lambda t: t["start"])


def transcribe(audio_path, cfg, device, wav=None):
    """Full-audio transcription with word timestamps. Returns list of words."""
    model = get_whisper(cfg["model"], cfg["compute_type"], device)
    src = wav if wav is not None else audio_path
    segments, _info = model.transcribe(
        src,
        language=cfg["language"],
        beam_size=cfg["beam_size"],
        patience=cfg["patience"],
        length_penalty=cfg["length_penalty"],
        repetition_penalty=cfg["repetition_penalty"],
        no_repeat_ngram_size=cfg["no_repeat_ngram_size"],
        temperature=cfg["temperature"],
        no_speech_threshold=cfg["no_speech_threshold"],
        log_prob_threshold=cfg["log_prob_threshold"],
        compression_ratio_threshold=cfg["compression_ratio_threshold"],
        condition_on_previous_text=cfg["condition_on_previous_text"],
        vad_filter=cfg["vad_filter"],
        vad_parameters=dict(threshold=cfg["vad_threshold"],
                            min_silence_duration_ms=cfg["vad_min_silence_ms"],
                            speech_pad_ms=cfg["vad_speech_pad_ms"]),
        word_timestamps=True,
    )
    words = []
    for seg in segments:
        # guard the RES0100 failure mode: segment decoded with no tokens
        if not getattr(seg, "words", None):
            if (seg.text or "").strip():
                words.append(dict(start=float(seg.start), end=float(seg.end),
                                  word=seg.text, degraded=True))
            continue
        for w in seg.words:
            if w.start is None or w.end is None:
                continue
            # NOT stripped: frozen/single_large_v3.py stores word.word verbatim
            # and build_rows joins with "". faster-whisper puts the leading
            # space inside the token, so stripping here would concatenate the
            # whole row into a single unbroken string.
            words.append(dict(start=float(w.start), end=float(w.end),
                              word=w.word))
    return words


def merge_turns(turns, merge_gap):
    """Merge consecutive same-speaker turns separated by < merge_gap seconds."""
    if not turns:
        return []
    out = [dict(turns[0])]
    for t in turns[1:]:
        last = out[-1]
        if t["speaker"] == last["speaker"] and (t["start"] - last["end"]) < merge_gap:
            last["end"] = max(last["end"], t["end"])
        else:
            out.append(dict(t))
    return out


def overlap_duration(s1, e1, s2, e2):
    return max(0.0, min(e1, e2) - max(s1, s2))


def assign_overlap(words, diar):
    """Verbatim from frozen/single_large_v3.py: each word gets the speaker of
    the interval it overlaps most."""
    out = []
    for w in words:
        best_spk, best_ov = "UNKNOWN", 0.0
        for seg in diar:
            ov = overlap_duration(w["start"], w["end"], seg["start"], seg["end"])
            if ov > best_ov:
                best_ov, best_spk = ov, seg["speaker"]
        out.append(dict(start=w["start"], end=w["end"],
                        speaker=best_spk, text=w["word"]))
    return out


def assemble(words, turns, min_interval, merge_gap):
    """Build rows after assigning each eligible word to one interval."""
    merged = merge_turns(turns, merge_gap)
    kept = [t for t in merged if (t["end"] - t["start"]) >= min_interval]
    if not kept:
        return ""

    assigned = assign_overlap(words, kept)
    interval_words, _ = assign_words_to_intervals(assigned, kept)

    lines = []
    for iv, sel in zip(kept, interval_words):
        s, e = iv["start"], iv["end"]
        if not sel:
            continue
        totals = {}
        for w in sel:
            totals[w["speaker"]] = totals.get(w["speaker"], 0.0) + \
                overlap_duration(w["start"], w["end"], s, e)
        spk = max(totals, key=totals.get)
        text = "".join(w["text"] for w in sel).strip()
        if text:
            lines.append(f"{spk}: {text}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", nargs="+", type=float,
                    default=[0.0, 0.10, 0.25, 0.40])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None,
                    help="JSON list of recording ids (e.g. dev40.json)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ext", default=".wav")
    ap.add_argument("--model", default=None, help="override reference model")
    ap.add_argument("--shard", type=int, default=0,
                    help="this worker's index, 0-based")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="total number of parallel workers")
    ap.add_argument("--compute-type", default=None,
                    help="override compute_type (e.g. float16)")
    args = ap.parse_args()

    cfg = dict(REFERENCE)
    if args.model:
        cfg["model"] = args.model
    if args.compute_type:
        cfg["compute_type"] = args.compute_type

    root = args.out
    for sub in ["rttm", "words", "logs"]:
        os.makedirs(os.path.join(root, sub), exist_ok=True)

    def cond_name(lv):
        return {0.0: "min_interval_00", 0.10: "baseline",
                0.25: "min_interval_25", 0.40: "min_interval_40"}.get(
                    lv, f"min_interval_{lv:g}")

    for lv in args.levels:
        os.makedirs(os.path.join(root, cond_name(lv)), exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.audio, f"*{args.ext}")))
    if len(files) < 2:                     # audio is probably in subfolders
        rec = sorted(glob.glob(os.path.join(args.audio, "**", f"*{args.ext}"),
                               recursive=True))
        if len(rec) > len(files):
            print(f"[scan] {len(files)} at top level, {len(rec)} recursively "
                  f"-- using recursive")
            files = rec
    if not files:
        for alt in (".mp3", ".m4a", ".flac", ".wav"):
            alt_files = sorted(glob.glob(os.path.join(args.audio, "**", f"*{alt}"),
                                         recursive=True))
            if alt_files:
                print(f"[scan] no {args.ext} found; {len(alt_files)} {alt} files "
                      f"-- using {alt}")
                files = alt_files
                break
    if not files:
        raise SystemExit(f"no audio found under {args.audio}")
    if args.ids:
        want = json.load(open(args.ids))
        want = set(want if isinstance(want, list)
                   else (want.get("dev") or want.get("ids") or []))
        files = [f for f in files
                 if os.path.splitext(os.path.basename(f))[0] in want]
    if args.limit:
        files = files[:args.limit]

    if args.num_shards > 1:
        total = len(files)
        # interleave rather than block-split: consultation length varies, so
        # every shard gets a comparable mix of long and short recordings
        files = files[args.shard::args.num_shards]
        print(f"[shard] worker {args.shard} of {args.num_shards}: "
              f"{len(files)} of {total} recordings")

    print(f"[plan] {len(files)} recordings x {len(args.levels)} levels")
    print(f"[cfg ] {json.dumps({k: v for k, v in cfg.items()}, default=str)}")

    manifest, failures = [], []
    t_start = time.time()

    for i, apath in enumerate(files, 1):
        rid = os.path.splitext(os.path.basename(apath))[0]
        rttm = os.path.join(root, "rttm", f"{rid}.rttm")
        wjson = os.path.join(root, "words", f"{rid}.json")
        t0 = time.time()
        try:
            need_diar = not (os.path.exists(rttm) and rttm_is_valid(rttm))
            if os.path.exists(rttm) and not rttm_is_valid(rttm):
                print(f"      stale/empty rttm for {rid} -- regenerating")
                os.remove(rttm)
            need_asr = not os.path.exists(wjson)
            wav = None
            if need_diar or need_asr:
                wav, sr = load_audio(apath)
                if wav.size == 0:
                    raise RuntimeError("decoded 0 audio samples")

            turns = (diarize(apath, rttm, args.device, wav, sr)
                     if need_diar else load_rttm(rttm))

            if need_asr:
                words = transcribe(apath, cfg, args.device, wav)
                if not words:
                    raise RuntimeError("transcription produced no words")
                tmp = wjson + ".tmp"
                json.dump(words, open(tmp, "w"))
                os.replace(tmp, wjson)
            else:
                words = json.load(open(wjson))
                if not words:
                    raise RuntimeError(f"cached {wjson} is empty -- delete it")

            for lv in args.levels:
                out_txt = os.path.join(root, cond_name(lv), f"{rid}.txt")
                if os.path.exists(out_txt):
                    continue
                text = assemble(words, turns, lv, cfg["merge_gap"])
                if not text.strip():
                    raise RuntimeError(
                        f"assembly produced an empty transcript at "
                        f"min_interval={lv} ({len(words)} words, "
                        f"{len(turns)} turns)")
                open(out_txt, "w", encoding="utf-8").write(text)

            manifest.append(dict(recording=rid, status="ok",
                                 n_words=len(words), n_turns=len(turns),
                                 seconds=round(time.time() - t0, 1)))
        except Exception as exc:                                   # noqa: BLE001
            failures.append(dict(recording=rid, error=repr(exc)))
            open(os.path.join(root, "logs", f"{rid}.err"), "w").write(
                traceback.format_exc())
            manifest.append(dict(recording=rid, status="failed",
                                 seconds=round(time.time() - t0, 1)))
            print(f"  [{i}/{len(files)}] {rid} FAILED: {exc}")
            continue

        el = time.time() - t_start
        print(f"  [{i}/{len(files)}] {rid}  {time.time()-t0:6.1f}s"
              f"  elapsed {el/60:6.1f}m  eta {(el/i)*(len(files)-i)/60:6.1f}m")

    meta = dict(generated=datetime.datetime.now().isoformat(timespec="seconds"),
                config=cfg, levels=args.levels,
                n_ok=sum(1 for m in manifest if m["status"] == "ok"),
                n_failed=len(failures), failures=failures,
                recordings=manifest)
    mname = ("run_manifest.json" if args.num_shards == 1
             else f"run_manifest_shard{args.shard}.json")
    json.dump(meta, open(os.path.join(root, mname), "w"), indent=2, default=str)

    print(f"\n[done] ok {meta['n_ok']}  failed {meta['n_failed']}"
          f"  total {(time.time()-t_start)/60:.1f} min")
    print(f"[out ] {root}")
    if failures:
        print("\nFailures (report, do not silently drop -- condition-specific "
              "missingness is itself a finding):")
        for f in failures:
            print(f"  {f['recording']}: {f['error']}")


if __name__ == "__main__":
    main()
