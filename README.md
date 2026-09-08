# Clinical ASR configuration selection

Public reproducibility repository for the manuscript **Multi-Outcome
Configuration Selection for Clinical Speech Recognition: Word Error Rate Can
Conceal Negation and Speaker-Attribution Trade-offs**.

It contains the study code, frozen manifests, aggregate results,
post-freeze sensitivity analyses, validation tests, and pipeline hashes. It
does not redistribute audio, reference
transcripts, model weights, caches, access tokens, or local clinical
recordings.

## Repository layout

- `pipeline/`: transcription, diarization, PriMock57 preparation, and the
  paired one-factor study driver.
- `search/`: Optuna/TPE searches for Whisper large-v2 and large-v3.
- `scoring/`: WER, clinical-error, speaker-attribution, and corpus scorers.
- `analysis/`: paired comparisons, selection rules, sensitivity analyses,
  surrogate analyses, and audit utilities.
- `analysis/post_freeze/`: declared post-freeze sensitivity analyses.
- `comparators/`: Canary-1b-v2 and the documented NeMo MSDD attempt.
- `figures/`: scripts used to generate manuscript figures and tables.
- `manifests/`: frozen split, condition registry, lexicon, requirements, and
  SHA-256 records.
- `results/`: the study log, completed optimisation summaries, and aggregate
  per-consultation clinical metrics.
- `validation/`: the synthetic WER, WDER, SA-WER, and DER validation suite and
  its 56 passing metric-level comparisons across 19 cases.
- `tests/`: focused tests for the insertion-inclusive negation audit.

## Environment

The installable Python environment is recorded in `requirements.txt`.
`manifests/requirements.txt` is the immutable environment record covered by
the analytic-freeze hash; its trailing version-summary lines are retained as
provenance and are not pip requirement entries. The reported GPU runs used
WSL2, CUDA 12.9, and one NVIDIA RTX 5090 Laptop GPU. A CUDA-capable environment
is required for transcription and diarization; saved-result analyses can run
without a GPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Pyannote model access requires a Hugging Face token supplied at run time:

```bash
export HF_TOKEN="replace-with-authorized-token"
```

Never commit a token or `.env` file. The token used during development should
be rotated before this repository is made public.

## Data and path configuration

Set only the roots needed by the command being run:

```bash
export ASR_DATA_ROOT=/path/to/fareez
export ASR_STUDY_ROOT=/path/to/study
export ASR_CACHE_ROOT=/path/to/transcript-cache
export ASR_PRIMOCK_ROOT=/path/to/primock57-prepared
export ASR_CODE_ROOT=/path/to/repository/pipeline
export ASR_RESULTS_ROOT=/path/to/repository/results
export ASR_CLINICAL_RESULTS_ROOT=/path/to/repository/results/clinical
export ASR_STUDY_LOG=/path/to/repository/results/study_log.csv
```

The Fareez and PriMock57 corpora are not included. Obtain them from their
original publications and follow their access and licensing terms. The
committed result CSVs contain aggregate error counts/rates and public corpus
case identifiers, not audio or transcript text.

## Reproduce saved-result analyses

From the repository root:

```bash
python analysis/final_select_from_log.py
python analysis/final_paired_compare.py --help
python analysis/final_minimax.py --help
python figures/make_figures.py
python figures/make_response_figures.py
python tests/test_row_assignment.py
python tests/test_negation_insertion_audit.py
python validation/synthetic_validation.py \
  --module scoring.final_score_primock \
  --out validation/synthetic_validation_reproduced.csv
```

Use `--help` for script-specific arguments. Full transcription and search runs
require the corpora, model access, CUDA environment, and roots above.

## Reproducibility records

- `manifests/FROZEN_AT.txt` records the analytic-freeze timestamp.
- `manifests/HASHES.txt` records the frozen scorer, lexicon, registry, and
  environment hashes.
- `manifests/pipeline_hashes.txt` distinguishes the original and post-freeze
  pipeline snapshots.
- `manifests/subset.json`, `conditions.json`, `conditions_all.json`,
  `dev_split.json`, and `lexicon_v1.json` record the evaluated design.
- `results/post_freeze/HASHES_post_freeze.txt` records hashes for the
  post-freeze analysis and validation files.
- `validation/synthetic_validation.py` contains the executable 19-case metric
  suite; `validation/synthetic_validation.csv` records the retained results.

The post-freeze pipeline exposes decoder parameters that were previously
implicit and includes a faster-whisper 1.2.1 word-alignment fallback. The
pipeline hashes must be used to determine which snapshot produced each run.

## Known limitations

- One `tpe_trial32` recording was excluded from matched comparisons after a
  word-alignment failure; those comparisons use 39 recordings.
- The NeMo MSDD comparator did not reach neural-refinement output and is not a
  scored diarization comparator.
- The one-factor and TPE stages used different negation cue sets; their rates
  are reported stage-specifically and must not be pooled.
- The repository does not make the source datasets redistributable and does
  not make the system suitable for clinical deployment.
- Independent clinician adjudication of the clinical-cue definitions has not
  been completed; the released signals are reproducible measurement
  definitions, not clinically validated ground truth or measures of harm.

## Citation and licence

The final citation, DOI, authors, and software licence will be added when the
manuscript metadata and release route are fixed. Until then, this private
repository is supplied for collaborator/reviewer access only; no software
licence is granted.
