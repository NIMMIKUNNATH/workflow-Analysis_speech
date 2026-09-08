import os
import sys
import gc
import json
import time
import subprocess
from pathlib import Path

import soundfile as sf
import torch

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from openpyxl import Workbook
from openpyxl.styles import Font, Alignment

try:
    from .row_assignment import assign_words_to_intervals
except ImportError:
    from row_assignment import assign_words_to_intervals


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16"

LANGUAGE = "en"
BEAM_SIZE = 5
VAD_FILTER = True
WORD_TIMESTAMPS = True
INITIAL_PROMPT = None

NUM_SPEAKERS = 2

MERGE_GAP_SECONDS = 1.50
MIN_INTERVAL_DURATION = 0.15

# --- exposed for the parameter study -------------------------------------
# These are Whisper's own defaults, written out explicitly so the study can
# vary them. Adding them does NOT change behaviour: the pipeline was already
# using these values implicitly.
NO_SPEECH_THRESHOLD = 0.6    # faster-whisper default; above this AND low logprob, a window is skipped
LOG_PROB_THRESHOLD = -1.0    # faster-whisper default; below this a segment is retried at higher temperature
COMPRESSION_RATIO_THRESHOLD = 3.0
CONDITION_ON_PREVIOUS_TEXT = True    # Whisper default; feeds each segment's text forward as context
PATIENCE = 1.0    # 1.0 is conventional beam search
LENGTH_PENALTY = 1.0    # 1.0 is simple length normalisation
VAD_THRESHOLD = 0.30    # frozen selection, was 0.5 (Silero default)
VAD_MIN_SILENCE_MS = 2000    # Silero default; silences shorter than this do not split speech
VAD_SPEECH_PAD_MS = 400    # Silero default padding either side of a detected speech region
REPETITION_PENALTY = 1.0    # faster-whisper default; >1 discourages repeating tokens
NO_REPEAT_NGRAM_SIZE = 0    # faster-whisper default; 0 disables n-gram blocking

# --- exposed for the TPE conditions --------------------------------------
TEMPERATURE = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]    # faster-whisper default schedule
VAD_MIN_SPEECH_MS = 250    # Silero default; speech shorter than this is discarded
BEST_OF = 5    # faster-whisper default; active only when sampling


# ============================================================
# HELPERS
# ============================================================

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def overlap_duration(start1, end1, start2, end2):
    start = max(start1, start2)
    end = min(end1, end2)

    return max(0.0, end - start)


def get_audio_duration(wav_path):
    info = sf.info(str(wav_path))
    return float(info.duration)


# ============================================================
# AUDIO EXTRACTION
# ============================================================

def extract_audio(video_path, wav_path):
    print("\n[1/5] Extracting mono 16-kHz audio...")

    start = time.time()

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    elapsed = time.time() - start

    print(f"Audio extraction time: {elapsed:.1f} sec")

    return elapsed


# ============================================================
# DIARIZATION
# ============================================================

def run_diarization(wav_path):
    print("\n[2/5] Running speaker diarization...")

    if "HF_TOKEN" not in os.environ:
        print("ERROR: HF_TOKEN is not set.")
        sys.exit(1)

    start = time.time()

    audio, sample_rate = sf.read(str(wav_path))

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    waveform = torch.from_numpy(audio).float().unsqueeze(0)

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=os.environ["HF_TOKEN"],
    )

    pipeline.to(torch.device("cuda"))

    result = pipeline(
        {
            "waveform": waveform,
            "sample_rate": sample_rate,
        },
        num_speakers=NUM_SPEAKERS,
    )

    elapsed = time.time() - start

    annotation = result.exclusive_speaker_diarization

    raw_labels = sorted(
        set(
            speaker
            for _, _, speaker
            in annotation.itertracks(yield_label=True)
        )
    )

    speaker_mapping = {}

    if len(raw_labels) >= 1:
        speaker_mapping[raw_labels[0]] = "Speaker A"

    if len(raw_labels) >= 2:
        speaker_mapping[raw_labels[1]] = "Speaker B"

    diar = []

    for turn, _, speaker in annotation.itertracks(yield_label=True):
        diar.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker_mapping.get(
                    speaker,
                    speaker,
                ),
            }
        )

    print(f"Diarization time: {elapsed:.1f} sec")

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return diar, elapsed


# ============================================================
# MERGE ADJACENT SAME-SPEAKER INTERVALS
# ============================================================

def merge_diarization_intervals(diar):
    if not diar:
        return []

    diar = sorted(
        diar,
        key=lambda x: (x["start"], x["end"]),
    )

    merged = []

    current = {
        "start": diar[0]["start"],
        "end": diar[0]["end"],
        "speaker": diar[0]["speaker"],
    }

    for segment in diar[1:]:
        gap = segment["start"] - current["end"]

        same_speaker = (
            segment["speaker"] == current["speaker"]
        )

        if same_speaker and gap <= MERGE_GAP_SECONDS:
            current["end"] = max(
                current["end"],
                segment["end"],
            )
        else:
            if (
                current["end"] - current["start"]
                >= MIN_INTERVAL_DURATION
            ):
                merged.append(current)

            current = {
                "start": segment["start"],
                "end": segment["end"],
                "speaker": segment["speaker"],
            }

    if (
        current["end"] - current["start"]
        >= MIN_INTERVAL_DURATION
    ):
        merged.append(current)

    return merged


# ============================================================
# TRANSCRIPTION
# ============================================================

def transcribe_large_v3(wav_path):
    print("\n[3/5] Running large-v3 transcription...")

    start = time.time()

    model = WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )

    segments, info = model.transcribe(
        str(wav_path),
        language=LANGUAGE,
        beam_size=BEAM_SIZE,
        vad_filter=VAD_FILTER,
        word_timestamps=WORD_TIMESTAMPS,
        initial_prompt=INITIAL_PROMPT,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
        log_prob_threshold=LOG_PROB_THRESHOLD,
        compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
        condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
        patience=PATIENCE,
        temperature=TEMPERATURE,
        best_of=BEST_OF,
        length_penalty=LENGTH_PENALTY,
        repetition_penalty=REPETITION_PENALTY,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        vad_parameters=dict(threshold=VAD_THRESHOLD,
                            min_speech_duration_ms=VAD_MIN_SPEECH_MS,
                            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
                            speech_pad_ms=VAD_SPEECH_PAD_MS),
    )

    # --- ALIGNMENT FALLBACK ---------------------------------------------

    # faster-whisper 1.2.1 raises IndexError in find_alignment() when a

    # decoded segment has no tokens and word_timestamps=True. Retry once

    # without word timestamps so the recording completes; the file is

    # flagged so the coarser timestamp basis can be disclosed.

    ALIGNMENT_FALLBACK_USED = False

    try:

        segments = list(segments)

    except IndexError as e:

        print(f'    WARNING: word-timestamp alignment failed ({e});'

              f' retrying without word timestamps', flush=True)

        ALIGNMENT_FALLBACK_USED = True

        segments, _ = model.transcribe(

            str(wav_path),

            language=LANGUAGE,

            beam_size=BEAM_SIZE,

            vad_filter=VAD_FILTER,

            word_timestamps=False,

            initial_prompt=INITIAL_PROMPT,

            no_speech_threshold=NO_SPEECH_THRESHOLD,

            log_prob_threshold=LOG_PROB_THRESHOLD,

            compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,

            condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,

            patience=PATIENCE,

            temperature=TEMPERATURE,

            best_of=BEST_OF,

            length_penalty=LENGTH_PENALTY,

            repetition_penalty=REPETITION_PENALTY,

            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,

            vad_parameters=dict(threshold=VAD_THRESHOLD,

                                min_speech_duration_ms=VAD_MIN_SPEECH_MS,

                                min_silence_duration_ms=VAD_MIN_SILENCE_MS,

                                speech_pad_ms=VAD_SPEECH_PAD_MS),

        )

        segments = list(segments)

    # --- END ALIGNMENT FALLBACK -----------------------------------------
    words = []

    for segment in segments:
        if not segment.words:
            continue

        for word in segment.words:
            if word.start is None or word.end is None:
                continue

            words.append(
                {
                    "start": float(word.start),
                    "end": float(word.end),
                    "text": word.word,
                }
            )

    elapsed = time.time() - start

    print(f"Words produced: {len(words)}")
    print(f"Transcription time: {elapsed:.1f} sec")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return words, elapsed


# ============================================================
# WORD-TO-SPEAKER ASSIGNMENT
# ============================================================

def assign_overlap(words, diar):
    assigned = []

    for word in words:
        best_speaker = "UNKNOWN"
        best_overlap = 0.0

        for segment in diar:
            overlap = overlap_duration(
                word["start"],
                word["end"],
                segment["start"],
                segment["end"],
            )

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = segment["speaker"]

        assigned.append(
            {
                "start": word["start"],
                "end": word["end"],
                "speaker": best_speaker,
                "text": word["text"],
            }
        )

    return assigned


# ============================================================
# BUILD EXCEL ROWS
# ============================================================

def words_in_interval(words, start, end):
    selected = []

    for word in words:
        overlap = overlap_duration(
            word["start"],
            word["end"],
            start,
            end,
        )

        if overlap > 0:
            selected.append(word)

    return selected


def join_words(words):
    if not words:
        return ""

    return "".join(
        word["text"]
        for word in words
    ).strip()


def dominant_speaker(words, start, end):
    if not words:
        return ""

    totals = {}

    for word in words:
        speaker = word["speaker"]

        overlap = overlap_duration(
            word["start"],
            word["end"],
            start,
            end,
        )

        totals[speaker] = (
            totals.get(speaker, 0.0)
            + overlap
        )

    return max(
        totals,
        key=totals.get,
    )


def build_rows(shared_diar, assigned_words):
    rows = []
    interval_words, _ = assign_words_to_intervals(assigned_words, shared_diar)

    for interval, selected in zip(shared_diar, interval_words):
        start = interval["start"]
        end = interval["end"]

        text = join_words(selected)

        model_speaker = dominant_speaker(
            selected,
            start,
            end,
        )

        rows.append(
            {
                "start": start,
                "end": end,
                "speaker": model_speaker,
                "transcript": text,
            }
        )

    return rows


# ============================================================
# SAVE JSON
# ============================================================

def save_json(data, path):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# SAVE EXCEL
# ============================================================

def save_excel(
    rows,
    excel_path,
    video_name,
    duration,
    extraction_time,
    diarization_time,
    transcription_time,
    total_time,
):
    print("\n[4/5] Creating Excel file...")

    wb = Workbook()

    ws = wb.active
    ws.title = "Transcript Review"

    headers = [
        "Start",
        "End",
        "Speaker",
        "large-v3 transcript",
        "Researcher-verified speaker",
        "Researcher-corrected transcript",
    ]

    ws.append(headers)

    for cell in ws[1]:
        cell.font = Font(bold=True)

    for item in rows:
        ws.append(
            [
                format_time(item["start"]),
                format_time(item["end"]),
                item["speaker"],
                item["transcript"],
                "",
                "",
            ]
        )

    widths = {
        "A": 15,
        "B": 15,
        "C": 18,
        "D": 65,
        "E": 28,
        "F": 65,
    }

    for col, width in widths.items():
        ws.column_dimensions[col].width = width

    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = Alignment(
                vertical="top",
                wrap_text=True,
            )

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    summary = wb.create_sheet("Run Summary")

    summary.append(
        [
            "Parameter",
            "Value",
        ]
    )

    for cell in summary[1]:
        cell.font = Font(bold=True)

    summary_rows = [
        ["Video", video_name],
        ["Model", MODEL_NAME],
        ["Device", torch.cuda.get_device_name(0)],
        ["Compute type", COMPUTE_TYPE],
        ["Language", LANGUAGE],
        ["Beam size", BEAM_SIZE],
        ["VAD filter", VAD_FILTER],
        ["Word timestamps", WORD_TIMESTAMPS],
        ["Initial prompt", str(INITIAL_PROMPT)],
        ["Number of speakers", NUM_SPEAKERS],
        ["Duration (sec)", round(duration, 1)],
        ["Duration (min)", round(duration / 60, 2)],
        ["Audio extraction time (sec)", round(extraction_time, 1)],
        ["Diarization time (sec)", round(diarization_time, 1)],
        ["Transcription time (sec)", round(transcription_time, 1)],
        ["Total processing time (sec)", round(total_time, 1)],
        ["Real-time factor", round(total_time / duration, 3)],
        ["Speaker A role", ""],
        ["Speaker B role", ""],
    ]

    for row in summary_rows:
        summary.append(row)

    summary.column_dimensions["A"].width = 38
    summary.column_dimensions["B"].width = 55

    wb.save(excel_path)

    print(f"Excel saved: {excel_path}")


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(video_path):
    print("\n======================================")
    print(f"PROCESSING ONE VIDEO: {video_path.name}")
    print("======================================")

    print("\nGPU:")
    print(f"  {torch.cuda.get_device_name(0)}")

    total_start = time.time()

    stem = video_path.stem

    wav_path = video_path.with_name(
        f"{stem}.wav"
    )

    diar_json = video_path.with_name(
        f"{stem}_diarization.json"
    )

    words_json = video_path.with_name(
        f"{stem}_large_v3_words.json"
    )

    excel_path = video_path.with_name(
        f"{stem}_large_v3_review.xlsx"
    )

    if excel_path.exists():
        print(
            f"\nWARNING: {excel_path.name} already exists."
        )
        print(
            "This script will not overwrite a completed result."
        )
        return

    extraction_time = extract_audio(
        video_path,
        wav_path,
    )

    duration = get_audio_duration(
        wav_path
    )

    print(
        f"Recording duration: "
        f"{duration / 60:.1f} minutes"
    )

    diar, diarization_time = run_diarization(
        wav_path
    )

    merged_diar = merge_diarization_intervals(
        diar
    )

    words, transcription_time = transcribe_large_v3(
        wav_path
    )

    print("\n[4/5] Assigning speakers...")

    assigned_words = assign_overlap(
        words,
        merged_diar,
    )

    rows = build_rows(
        merged_diar,
        assigned_words,
    )

    save_json(
        merged_diar,
        diar_json,
    )

    save_json(
        assigned_words,
        words_json,
    )

    total_time = time.time() - total_start

    save_excel(
        rows,
        excel_path,
        video_path.name,
        duration,
        extraction_time,
        diarization_time,
        transcription_time,
        total_time,
    )

    print("\n[5/5] Complete.")

    print("\n======================================")
    print("PROCESSING SUMMARY")
    print("======================================")

    print(
        f"Recording duration: "
        f"{duration / 60:.1f} min"
    )

    print(
        f"Audio extraction: "
        f"{extraction_time:.1f} sec"
    )

    print(
        f"Diarization: "
        f"{diarization_time:.1f} sec"
    )

    print(
        f"Transcription: "
        f"{transcription_time:.1f} sec"
    )

    print(
        f"Total processing: "
        f"{total_time:.1f} sec"
    )

    print(
        f"Real-time factor: "
        f"{total_time / duration:.3f}"
    )

    print("\nOutput:")
    print(excel_path)


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) != 2:
        print("\nUsage:")
        print(
            "python single_large_v3.py "
            "Male_Female.mp4"
        )
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        sys.exit(1)

    if "HF_TOKEN" not in os.environ:
        print("ERROR: HF_TOKEN is not set.")
        sys.exit(1)

    video_path = Path(sys.argv[1])

    if not video_path.exists():
        print(
            f"ERROR: File not found: "
            f"{video_path}"
        )
        sys.exit(1)

    process_video(video_path)


if __name__ == "__main__":
    main()
