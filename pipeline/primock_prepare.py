#!/usr/bin/env python3
"""
primock_prepare.py - turn PriMock57 into something the pipeline can consume, and
build the timestamped reference the Fareez corpus could not provide.

WHY THIS DATASET MATTERS FOR THIS PROJECT
    Three limitations of the Fareez benchmark are answered here:

    1. TRUE DER BECOMES COMPUTABLE. Fareez transcripts carry no time alignment,
       so only word-level diarization error rate (WDER) could be reported. Every
       PriMock57 utterance has xmin and xmax, so reference speaker intervals
       exist and DER can be computed properly.

    2. THE ROLE RULE GETS A HELD-OUT TEST. The lexical doctor/patient rule was
       built by inspecting Fareez transcripts and evaluated on the same
       recordings - that is resubstitution accuracy, not performance. PriMock57
       is unseen data, so it is the first honest test of the frozen rule.

    3. A SECOND POPULATION. UK primary care rather than Canadian OSCE, with
       published accent distributions.

WHAT IT DOES
    PriMock57 ships each speaker on a SEPARATE audio file and a separate
    TextGrid. That is ground truth, but it is not the condition the pipeline is
    meant to operate in - a pipeline handed pre-separated speakers has nothing to
    diarize. So this merges the two channels into a single mixed track, exactly
    as a room microphone would capture, and writes the reference intervals to
    RTTM for scoring.

    The merge is the honest experiment: the pipeline must re-derive who spoke
    when, and can be scored against a known answer.

OUTPUTS
    <out>/audio/<consultation>.wav       merged mono 16 kHz
    <out>/rttm/<consultation>.rttm       reference speaker intervals
    <out>/text/<consultation>.txt        D:/P: transcript, for WER scoring

USAGE
    python primock_prepare.py --root /path/to/primock57 --out ${ASR_PRIMOCK_ROOT}
"""
import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Transcriber tags defined in the dataset README.
#   <UNIN/>   an unintelligible audio section
#   <UNSURE>  the transcriber was unsure of the transcription
#
# UNIN regions have no recoverable ground truth, so words there cannot be scored
# fairly - the tag is stripped and the (empty) text simply contributes nothing.
# UNSURE text IS retained: the transcriber produced a best guess, and discarding
# it would silently remove exactly the hard audio a benchmark should include.
TAG_UNIN = re.compile(r"<UNIN\s*/?>", re.I)
TAG_UNSURE = re.compile(r"</?UNSURE>", re.I)
OTHER_TAGS = re.compile(r"<[^>]+>")


def parse_textgrid(path):
    """
    Parse a Praat TextGrid into (xmin, xmax, text) tuples.

    Handles both the long format (explicit "intervals [n]:" blocks with named
    fields) and the short format (bare values on consecutive lines). PriMock57
    ships long format, but short-format files appear in the wild often enough
    that failing on them would be a nuisance.
    """
    raw = Path(path).read_bytes()
    # TextGrid may be UTF-8 or UTF-16 depending on the Praat version that wrote it
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        txt = raw.decode("utf-16", errors="replace")
    else:
        txt = raw.decode("utf-8", errors="replace")

    intervals = []

    # long format: xmin = 1.23 / xmax = 4.56 / text = "..."
    long_pat = re.compile(
        r'xmin\s*=\s*([\d.]+)\s*\n\s*xmax\s*=\s*([\d.]+)\s*\n\s*text\s*=\s*"((?:[^"]|"")*)"',
        re.S)
    for m in long_pat.finditer(txt):
        intervals.append((float(m.group(1)), float(m.group(2)),
                          m.group(3).replace('""', '"')))

    if not intervals:
        # short format: bare numbers and quoted strings in triples
        short_pat = re.compile(r'^\s*([\d.]+)\s*\n\s*([\d.]+)\s*\n\s*"((?:[^"]|"")*)"',
                               re.M | re.S)
        for m in short_pat.finditer(txt):
            intervals.append((float(m.group(1)), float(m.group(2)),
                              m.group(3).replace('""', '"')))

    cleaned = []
    for xmin, xmax, text in intervals:
        t = TAG_UNIN.sub(" ", text)
        t = TAG_UNSURE.sub(" ", t)
        t = OTHER_TAGS.sub(" ", t)
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            cleaned.append((xmin, xmax, t))
    return cleaned


def merge_audio(doc_wav, pat_wav, out_wav):
    """
    Mix the two speaker channels into one mono track.

    amix with normalize=0 preserves each speaker's original level; normalising
    would duck whichever speaker is louder and change the acoustic conditions the
    pipeline sees. 16 kHz mono matches what the pipeline expects.
    """
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-i", str(doc_wav), "-i", str(pat_wav),
           "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=longest:normalize=0[a]",
           "-map", "[a]", "-ac", "1", "-ar", "16000", "-b:a", "128k", str(out_wav)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, r.stderr.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="the cloned primock57 repo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="process only N (for a trial)")
    a = ap.parse_args()

    root = Path(a.root)
    audio_dir, tg_dir = root / "audio", root / "transcripts"
    if not audio_dir.is_dir() or not tg_dir.is_dir():
        sys.exit(f"expected {audio_dir} and {tg_dir}")

    out = Path(a.out)
    for sub in ("audio", "rttm", "text"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # group the per-speaker files by consultation
    consults = defaultdict(dict)
    for f in tg_dir.glob("*.TextGrid"):
        m = re.match(r"(.+)_(doctor|patient)$", f.stem)
        if m:
            consults[m.group(1)][m.group(2)] = f

    both = {k: v for k, v in consults.items() if "doctor" in v and "patient" in v}
    print(f"{len(both)} consultations with both speaker transcripts")
    if len(both) != len(consults):
        missing = set(consults) - set(both)
        print(f"  incomplete, skipped: {sorted(missing)}")

    names = sorted(both)
    if a.limit:
        names = names[:a.limit]

    ok = 0
    for i, name in enumerate(names, 1):
        doc_tg = parse_textgrid(both[name]["doctor"])
        pat_tg = parse_textgrid(both[name]["patient"])
        if not doc_tg or not pat_tg:
            print(f"[{i}/{len(names)}] {name}: empty transcript - skipped")
            continue

        # --- reference RTTM, for true DER --------------------------------
        segs = ([(s, e, "doctor") for s, e, _ in doc_tg] +
                [(s, e, "patient") for s, e, _ in pat_tg])
        segs.sort()
        with open(out / "rttm" / f"{name}.rttm", "w") as f:
            for s, e, spk in segs:
                f.write(f"SPEAKER {name} 1 {s:.3f} {e-s:.3f} "
                        f"<NA> <NA> {spk} <NA> <NA>\n")

        # --- D:/P: transcript, for WER -----------------------------------
        # Interleaved by start time so the word sequence matches the audio, which
        # is what an alignment-based WER needs.
        utts = ([(s, "D", t) for s, _, t in doc_tg] +
                [(s, "P", t) for s, _, t in pat_tg])
        utts.sort()
        with open(out / "text" / f"{name}.txt", "w", encoding="utf-8") as f:
            for _, spk, t in utts:
                f.write(f"{spk}: {t}\n")

        # --- merged audio -------------------------------------------------
        wav_out = out / "audio" / f"{name}.mp3"
        if wav_out.exists():
            print(f"[{i}/{len(names)}] {name}: audio already merged")
            ok += 1
            continue
        d_wav = audio_dir / f"{name}_doctor.wav"
        p_wav = audio_dir / f"{name}_patient.wav"
        if not d_wav.exists() or not p_wav.exists():
            print(f"[{i}/{len(names)}] {name}: audio missing "
                  f"(did git lfs pull finish?) - skipped")
            continue
        if d_wav.stat().st_size < 1000:
            print(f"[{i}/{len(names)}] {name}: audio is an LFS pointer, "
                  f"not a file - run 'git lfs pull'")
            continue

        good, err = merge_audio(d_wav, p_wav, wav_out)
        dur = max(e for _, e, _ in doc_tg + pat_tg)
        print(f"[{i}/{len(names)}] {name}: {len(doc_tg)}+{len(pat_tg)} utterances, "
              f"{dur/60:.1f} min" + ("" if good else f"  FFMPEG FAILED: {err[:60]}"))
        ok += good

    print(f"\n{ok} consultations prepared -> {out}")
    print(f"  audio: {out}/audio   (merged mono 16 kHz)")
    print(f"  rttm : {out}/rttm    (reference speaker intervals, for true DER)")
    print(f"  text : {out}/text    (D:/P: transcripts, for WER)")
    print("\nNext: run the pipeline over the merged audio, then score with")
    print("score_primock.py, which computes WER, WDER and DER.")


if __name__ == "__main__":
    main()
