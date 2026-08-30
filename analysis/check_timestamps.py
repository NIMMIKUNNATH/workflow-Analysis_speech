#!/usr/bin/env python3
"""
check_timestamps.py - settle, from the data itself, whether DER is computable.

DER requires the REFERENCE to say when each speaker spoke. This scans every
reference transcript for any time-like pattern (00:01:23, [1:23], 1.5 --> 3.2,
and similar) and reports what fraction carry one.

If the answer is zero, WDER is the correct metric and that is a fact about the
published dataset, not a limitation of the pipeline. If some carry times, DER is
available on that subset and worth pursuing.

USAGE
    python check_timestamps.py --ref-dir "/path/to/Clean Transcripts"
"""
import argparse, re, sys
from pathlib import Path

# Any plausible time marker: hh:mm:ss(.ms), [m:ss], 12.34 --> 56.78, <00:01:02>
TIME_PATTERNS = [
    re.compile(r"\d{1,2}:\d{2}:\d{2}"),
    re.compile(r"\[\s*\d{1,3}:\d{2}"),
    re.compile(r"\d+\.\d+\s*-->\s*\d+\.\d+"),
    re.compile(r"<\s*\d{1,2}:\d{2}"),
    re.compile(r"^\s*\d+\.\d+\s+\d+\.\d+", re.M),
]
TURN = re.compile(r"^\s*([DP])\s*:", re.I)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True)
    a = ap.parse_args()

    files = sorted(Path(a.ref_dir).glob("*.txt"))
    if not files:
        sys.exit(f"no .txt files in {a.ref_dir}")

    with_time, no_turns, empty, ok = [], [], [], 0
    for f in files:
        txt = f.read_text(encoding="utf-8", errors="replace")
        if any(p.search(txt) for p in TIME_PATTERNS):
            with_time.append(f.stem)
        n_turns = len(TURN.findall(txt))
        if not txt.strip():
            empty.append((f.stem, f.stat().st_size))
        elif n_turns == 0:
            no_turns.append((f.stem, f.stat().st_size, txt.strip()[:70]))
        else:
            ok += 1

    print(f"\n{'='*70}\nREFERENCE TRANSCRIPT CHECK - {len(files)} files\n{'='*70}")
    print(f"\nfiles containing ANY time marker : {len(with_time)} of {len(files)}")
    if with_time:
        print("  ", ", ".join(with_time[:15]))
        print("\n  -> DER IS computable on these. Worth extracting.")
    else:
        print("  -> none. DER is NOT computable from this reference.")
        print("     WDER (El Shafey et al. 2019) is the correct metric, and this")
        print("     is a property of the published dataset, not of the pipeline.")

    print(f"\nfiles with D:/P: turn structure  : {ok} of {len(files)}")

    if no_turns:
        print(f"\nfiles WITHOUT D:/P: markers      : {len(no_turns)}")
        for stem, size, head in no_turns:
            print(f"   {stem:10s} {size:>7,} bytes | starts: {head!r}")
        print("\n   These are why some recordings scored 0 words. Inspect the")
        print("   format - a parser fix may recover them.")
    if empty:
        print(f"\ntruly empty files: {[s for s,_ in empty]}")

if __name__ == "__main__":
    main()
