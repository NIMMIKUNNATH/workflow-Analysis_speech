"""Configurable roots. Set these in the environment before running anything.

    export ASR_DATA_ROOT=/path/to/fareez
    export ASR_STUDY_ROOT=/path/to/study
    export ASR_CACHE_ROOT=/path/to/transcript-cache
    export ASR_PRIMOCK_ROOT=/path/to/primock57_prepared
"""

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("ASR_DATA_ROOT", "./data"))
STUDY_ROOT = Path(os.environ.get("ASR_STUDY_ROOT", "./study"))
CACHE_ROOT = Path(os.environ.get("ASR_CACHE_ROOT", "./cache"))
PRIMOCK_ROOT = Path(os.environ.get("ASR_PRIMOCK_ROOT", "./primock57"))
CODE_ROOT = Path(os.environ.get("ASR_CODE_ROOT", Path(__file__).parent))

AUDIO_DIR = DATA_ROOT / "Audio Recordings"
REF_DIR = DATA_ROOT / "Clean Transcripts"
