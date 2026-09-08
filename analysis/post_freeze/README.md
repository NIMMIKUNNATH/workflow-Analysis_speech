# Post-freeze analyses

These scripts reproduce sensitivity analyses added after the 26 August 2026
analytic freeze. They do not modify the frozen scorer, lexicon, condition
registry, or primary results. Their hashes are recorded separately in
`results/post_freeze/HASHES_post_freeze.txt`.

## Included analyses

| Script | Purpose |
|---|---|
| `selection_validity_bootstrap.py` | Recording-level bootstrap stability of the two configuration-selection rules |
| `equivalence_sensitivity.py` | Sensitivity to disabling negation-equivalence normalization |
| `negation_insertion_audit.py` | Insertion-inclusive negation scoring |
| `negation_paired_bootstrap.py` | Paired confidence intervals for the insertion-inclusive audit |

The associated regeneration utilities are in `pipeline/`:

- `run_min_interval_272.py` regenerates the 0.10-s and 0.40-s minimum-interval
  sensitivity runs.
- `make_branch_runners.py` creates the two PriMock57 branch-evaluation runners
  from `pipeline/single_large_v3.py` after verifying each patched constant.

The synthetic metric-validation suite is in `validation/`. Tests for the
insertion audit are in `tests/`.

## Requirements

Saved-result analyses require NumPy, pandas, and jiwer. Regeneration additionally
requires faster-whisper, pyannote.audio, soundfile, a CUDA-capable environment,
and authorized access to the gated pyannote model. Supply model credentials at
run time through the `HF_TOKEN` environment variable; never store credentials
in the repository.

## Examples

Run commands from the repository root. Replace example paths with local paths.

```bash
python analysis/post_freeze/selection_validity_bootstrap.py \
  --root /path/to/derived-outcomes --subset manifests/dev_split_v2.json \
  --outer 2000 --inner 400 --out selection_validity.csv

python analysis/post_freeze/equivalence_sensitivity.py \
  --clin /path/to/clinical-results --reference baseline \
  --out equivalence_sensitivity.csv

python analysis/post_freeze/negation_insertion_audit.py \
  --refs /path/to/references --hyps /path/to/hypotheses \
  --lexicon manifests/lexicon_v1.json \
  --conditions baseline min_interval_40 --out audit_272.csv

python analysis/post_freeze/negation_paired_bootstrap.py \
  --per-recording negation_insertion_per_recording.csv --reference baseline

python tests/test_negation_insertion_audit.py
python validation/synthetic_validation.py
```

`run_min_interval_272.py` is a corroborating reimplementation rather than the
frozen pipeline. It uses a different diarization checkpoint and may use
different library/audio-decoding versions; its outputs must not replace the
frozen 40-recording results.

## Deliberately excluded

The unadministered clinician-review workbook generator and blank templates are
not included. Independent clinical review remains future work, so no reviewer
decisions, agreement statistics, adjudicated lexicon, or clinician identifiers
belong in this reproducibility release.

Audio, reference transcripts, generated transcripts, model weights, caches,
and access tokens are also excluded.
