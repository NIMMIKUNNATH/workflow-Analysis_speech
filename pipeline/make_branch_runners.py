#!/usr/bin/env python3
"""
make_branch_runners.py
----------------------
Creates run_primock_ruleA.py and run_primock_ruleC.py from single_large_v3.py,
patched to the two constrained branch selections reported in Table 4.

  Rule A (Branch 1, negation-improvement)   large-v3, compression threshold 2.0
  Rule C (Branch 2, non-degradation screen) large-v2, beam size 3

Everything else is held at the frozen reference values. PriMock57
post-processing follows run_primock_trial35.py: merge gap 1.50 s and minimum
retained interval 0.15 s. single_large_v3.py is not modified.

WHY THIS MATTERS
    The existing PriMock57 result evaluates large-v3 TPE trial 35, a
    WER-selected configuration that is neither branch winner. These runs test
    whether the two constrained selections transfer to a corpus that took no
    part in selection. Because they are launched after all development results
    were seen, they are a declared post-hoc external evaluation, not a
    prespecified confirmation.

    Report whatever they show. Trial 35 changed PriMock57 WER by 0.12 points
    and the clinician/patient error direction reversed between corpora, so a
    null result for either winner is a plausible and reportable outcome.

Run from the `pipeline` directory:  python3 make_branch_runners.py
Then: python3 run_primock_ruleA.py /path/to/prepared-audio.wav
"""

import re
import sys
from pathlib import Path

SRC = Path("single_large_v3.py")

# (constant, expected current value, new value)
COMMON = [
    # PriMock57 post-processing, matching run_primock_trial35.py
    ("MERGE_GAP_SECONDS", "1.00", "1.50"),
    ("MIN_INTERVAL_DURATION", "0.10", "0.15"),
]

RUNNERS = {
    "run_primock_ruleA.py": dict(
        tag="primock_ruleA",
        # Branch 1 selection: large-v3 + compression threshold 2.0
        patch=[("COMPRESSION_RATIO_THRESHOLD", "2.4", "2.0")],
    ),
    "run_primock_ruleC.py": dict(
        tag="primock_ruleC",
        # Branch 2 selection: large-v2 + beam 3
        patch=[('MODEL_NAME', '"large-v3"', '"large-v2"'),
               ("BEAM_SIZE", "5", "3")],
    ),
}


def patch_constant(text, name, old, new, path_label):
    """Replace `NAME = old` with `NAME = new`, once, verifying the old value."""
    # no \b: it does not match after a closing quote, so string constants
    # such as MODEL_NAME = "large-v3" would fail the guard spuriously
    pattern = re.compile(rf"^({re.escape(name)}\s*=\s*){re.escape(old)}(?=\s|$|#)",
                         re.MULTILINE)
    found = pattern.search(text)
    if not found:
        cur = re.search(rf"^{re.escape(name)}\s*=\s*(.+)$", text, re.MULTILINE)
        cur = cur.group(1).strip() if cur else "<not found>"
        sys.exit(f"{path_label}: expected {name} = {old}, found {cur}. "
                 f"Refusing to patch blindly.")
    return pattern.sub(rf"\g<1>{new}", text, count=1)


def main():
    if not SRC.exists():
        sys.exit(f"{SRC} not found -- run this from the code directory.")
    base = SRC.read_text()

    for dst_name, spec in RUNNERS.items():
        dst = Path(dst_name)
        if dst.exists():
            print(f"skip {dst_name}: already exists (delete it to rebuild)")
            continue

        s = base
        for name, old, new in COMMON + spec["patch"]:
            s = patch_constant(s, name, old, new, dst_name)

        # distinct output naming so these cannot overwrite existing sheets
        tag = spec["tag"]
        if "TRIAL_TAG" in s:
            s = re.sub(r'^TRIAL_TAG\s*=\s*.+$', f'TRIAL_TAG = "{tag}"',
                       s, count=1, flags=re.MULTILINE)
        else:
            s = s.replace(
                'f"{stem}_large_v3_review.xlsx"',
                f'f"{{stem}}_{tag}_review.xlsx"')
            s = s.replace(
                'f"{stem}_large_v3_words.json"',
                f'f"{{stem}}_{tag}_words.json"')
            s = s.replace(
                'f"{stem}_diarization.json"',
                f'f"{{stem}}_{tag}_diarization.json"')

        # PriMock57 audio is already mono 16 kHz; skip the ffmpeg extraction
        s = s.replace(
            "    extraction_time = extract_audio(\n        video_path,\n"
            "        wav_path,\n    )",
            "    # PriMock57 audio is already mono 16 kHz -- no extraction needed\n"
            "    if wav_path.resolve() == video_path.resolve():\n"
            "        extraction_time = 0.0\n"
            "    else:\n"
            "        extraction_time = extract_audio(video_path, wav_path)")

        dst.write_text(s)
        applied = ", ".join(f"{n}={v}" for n, _, v in COMMON + spec["patch"])
        print(f"wrote {dst_name}")
        print(f"      {applied}")
        print(f"      output tag: _{tag}_review.xlsx")

    print("\nVerify before running:")
    for d in RUNNERS:
        if Path(d).exists():
            print(f"  grep -n 'MODEL_NAME\\|BEAM_SIZE\\|COMPRESSION_RATIO\\|"
                  f"MERGE_GAP_SECONDS\\|MIN_INTERVAL_DURATION' {d}")


if __name__ == "__main__":
    main()
