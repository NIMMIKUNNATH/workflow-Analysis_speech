# Remaining items before archival release

The public repository contains the study code, frozen environment, manifests,
hashes, optimisation summaries, study log, aggregate clinical-result tables,
post-freeze sensitivity analyses, and the synthetic metric-validation suite.

## Author input still required

- Final author list, order, affiliations, and corresponding-author email.
- Software licence choice.
- Final manuscript citation and DOI or preprint URL.
- Ethics/data-governance wording for each dataset and local recording source.
- Funding statement, competing-interests declaration, and CRediT roles.
- Final archived release tag and DOI or permanent archive identifier.

## Study work still outstanding

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
