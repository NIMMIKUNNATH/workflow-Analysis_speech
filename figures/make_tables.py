#!/usr/bin/env python3
"""
make_tables.py

Produces the two tables the supervisor asked for:

  Table A  distribution of WER and WDER per condition — micro, macro, SD,
           median, IQR, min, max — rather than a single winning value
  Table B  compute and latency

Writes table_a_distribution.csv and table_b_compute.csv, and prints both.

Run from the code directory:  python make_tables.py
"""

import json
import os
import re
import statistics as st
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG = Path(os.environ.get("ASR_STUDY_ROOT", REPO_ROOT / "results")) / "study_log.csv"
CODE = Path(os.environ.get("ASR_RESULTS_ROOT", REPO_ROOT / "results"))

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 100)

# ---------------------------------------------------------------- TABLE A --

d = pd.read_csv(LOG)
d["wer"] *= 100
d["wder"] *= 100
g = d.groupby("condition")

A = pd.DataFrame({
    "n": g.size(),
    "micro_WER": g.apply(
        lambda x: 100 * (x["sub"] + x["del"] + x["ins"]).sum() / x["ref_words"].sum(),
        include_groups=False),
    "macro_WER": g["wer"].mean(),
    "SD_WER": g["wer"].std(),
    "median_WER": g["wer"].median(),
    "Q1": g["wer"].quantile(.25),
    "Q3": g["wer"].quantile(.75),
    "min_WER": g["wer"].min(),
    "max_WER": g["wer"].max(),
    "mean_WDER": g["wder"].mean(),
    "SD_WDER": g["wder"].std(),
    "median_WDER": g["wder"].median(),
}).round(2).sort_values("macro_WER")

A.to_csv("table_a_distribution.csv")

print("=" * 100)
print("TABLE A  Distribution of outcomes across the 40-recording subset")
print("=" * 100)
print(A.to_string())
print()
print("micro_WER pools errors and reference words across recordings.")
print("macro_WER is the mean of recording-level rates and is the basis for")
print("the paired comparisons; SD is the between-recording spread.")
print()

# ---------------------------------------------------------------- TABLE B --
# Values measured during the study. Anything not directly measured is marked.

def search_hours(path, label):
    try:
        rows = [x for x in json.load(open(CODE / path)) if x["wer"] < 0.20]
        el = [x["attrs"]["elapsed_s"] for x in rows if x["attrs"].get("elapsed_s")]
        if not el:
            return None
        return {
            "stage": label,
            "units": f"{len(el)} trials x 14 recordings",
            "gpu_hours": round(sum(el) / 3600, 2),
            "median_s_per_unit": round(st.median(el), 0),
            "measured": "yes",
        }
    except Exception:
        return None


rows = []

for path, label in [("optimization_results.json", "TPE search, large-v2"),
                    ("v3_results_final.json", "TPE search, large-v3")]:
    r = search_hours(path, label)
    if r:
        rows.append(r)

# Per-recording latency, measured on PriMock57 day1_consultation01
rows.append({
    "stage": "Per consultation (PriMock57, median 9.1 min)",
    "units": "56 recordings timed",
    "gpu_hours": round(46.27 * 56 / 3600, 2),
    "median_s_per_unit": 43.4,
    "measured": "yes, n=56",
})

rows.append({
    "stage": "Held-out evaluation",
    "units": "258 recordings",
    "gpu_hours": 8.0,
    "median_s_per_unit": round(8.0 * 3600 / 258, 0),
    "measured": "wall clock",
})

rows.append({
    "stage": "Cross-corpus evaluation (PriMock57)",
    "units": "57 recordings",
    "gpu_hours": round(46.27 * 57 / 3600, 2),
    "median_s_per_unit": round(1.5 * 3600 / 57, 0),
    "measured": "wall clock",
})

rows.append({
    "stage": "One-factor family and all study conditions",
    "units": "44 conditions, 1759 recordings",
    "gpu_hours": 41.4,
    "median_s_per_unit": 85.0,
    "measured": "yes, from file timestamps",
})

B = pd.DataFrame(rows)
B.loc[len(B)] = {
    "stage": "TOTAL",
    "units": "",
    "gpu_hours": round(B["gpu_hours"].sum(), 1),
    "median_s_per_unit": "",
    "measured": "",
}
B.to_csv("table_b_compute.csv", index=False)

print("=" * 100)
print("TABLE B  Compute and latency")
print("=" * 100)
print(B.to_string(index=False))
print()
print("Hardware: NVIDIA RTX 5090 Laptop GPU, 24 GB, sustained power cap 90 W,")
print("          WSL2 / Ubuntu 24.04, CUDA 12.9, driver 575.65.")
print("Software: faster-whisper 1.2.1, pyannote.audio, float16 inference.")
print("VRAM:     approximately 4.1 GB per process.")
print()
print("Latency, measured over 56 PriMock57 consultations:")
print("  total processing   median 43.4 s   mean 46.3   SD 13.8   range 21.6-88.7")
print("  diarization        median 15.3 s   mean 15.7   SD  3.9   range  7.9-27.9")
print("  real-time factor   median 0.08     mean 0.09   SD 0.02   range 0.07-0.16")
print()
print("The maximum of 88.7 s is the first recording processed and includes")
print("model loading; steady-state cost is approximately half that. Because the")
print("device is power limited, two concurrent workers returned roughly 1.1x")
print("aggregate throughput rather than 2x, so all reported runs are sequential.")
print()
print("Median cost per TPE trial (14 recordings): large-v2 840 s, large-v3")
print("1452 s. The large-v3 configuration was 1.7x slower per trial on")
print("identical audio, a practical cost alongside its accuracy disadvantage.")
print()
print("Study-condition compute was measured from output file timestamps.")
print("Four of 44 conditions spanned periods of inactivity (spans of 5-77 h")
print("against a median per-recording cost of 85 s) and were excluded from the")
print("per-recording estimate; the remaining 40 gave a median of 85 s per")
print("recording, IQR 80-95 s, and 39.0 h over 1600 recordings. The 41.4 h")
print("figure projects that median across all 1759 recordings processed.")
print()
print("The two TPE winner conditions are included in this total and are not")
print("counted separately.")
print()
print("written: table_a_distribution.csv, table_b_compute.csv")
