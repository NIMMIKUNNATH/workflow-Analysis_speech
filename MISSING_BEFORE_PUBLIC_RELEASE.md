# Missing items before public release

The private repository contains the complete planned code inventory (30 study
scripts), the frozen environment, manifests, hashes, optimisation summaries,
study log, and 57 aggregate clinical-result tables. No required code file in
the release inventory is currently missing.

## Author input still required

- Final author list, order, affiliations, and corresponding-author email.
- Software licence choice.
- Final manuscript citation and DOI or preprint URL.
- Ethics/data-governance wording for each dataset and local recording source.
- Funding statement, competing-interests declaration, CRediT roles, and the
  target journal's generative-AI disclosure.
- Confirmation that the Hugging Face token used during development has been
  rotated before any public release.

## Study work still outstanding

- Publish the synthetic metric-validation suite with known WER, WDER, SA-WER,
  and DER error counts.
- Complete the segment-level minimum-interval mechanism analysis linking
  deleted negations to reference-utterance duration and error type.
- Obtain independent clinical review of the outcome definitions and
  investigator-set non-inferiority margins.
- Add any regenerated results if the clinical lexicon or negation cue set is
  changed after review.

## Intentionally excluded—do not upload

- Raw Fareez or PriMock57 audio and reference transcripts; obtain these from
  their original sources under their own terms.
- Local ceiling-microphone recordings or derived transcripts.
- Model weights, Hugging Face caches, pyannote/NeMo caches, and temporary audio.
- `.env` files, access tokens, shell histories, workstation paths, or personal
  identifiers.
- Unpublished drafts, slides, meeting notes, and the OSCE session package.

## Optional additions later

- Final manuscript and supplementary PDFs after author/ethics declarations are
  resolved.
- Release tag and archived DOI after acceptance or preprint release.
- Continuous-integration checks for syntax and the synthetic validation suite.
