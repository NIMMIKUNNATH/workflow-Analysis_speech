# Post-freeze results summary

Numerical results of the analyses in `analysis/post_freeze/` and `pipeline/`.

## 1. Selection stability (`selection_validity.csv`)

2,000 outer resamples of the 40 development recordings, 400 inner draws, seed
20260902. Both point selections reproduce Table 4.

| Rule | Selection | Frequency | Mean candidates passing |
|---|---|---|---|
| A (negation improvement) | compression threshold 2.0 | 46.1% | 3.3 |
| A | VAD minimum silence 1000 ms | 20.1% | |
| A | large-v3 reference | 14.3% | |
| C (no-detected-deterioration screen) | large-v2 + beam 3 | 33.6% | 26.9 |
| C | four other large-v2 configurations | 51.8% combined | |

## 2. Failure-aware sensitivity

All 42 selection-set configurations have complete 40-recording outcome
matrices; no configuration produced an unscoreable recording. Only TPE trial 32
is affected (RES0100).

| | Mean WER |
|---|---|
| trial 32, 39 scoreable recordings | 13.495% |
| trial 32, RES0100 scored at 100% | 15.658% |
| large-v3 reference | 18.537% |

The penalty costs 2.163 points. Trial 32 remains below the large-v3 reference,
but its apparent advantage over trial 35 contracts to 0.09 percentage points
and is not robust to the missing-output policy.

## 3. Equivalence-convention sensitivity
(`equivalence_sensitivity.csv`, `equivalence_sensitivity_medication.csv`)

Medication: 0 events suppressed in all 42 configurations.
Negation: 787 suppressed (14.3% of matched cue events); uniform increase of
0.66–1.27 pp.

| Configuration | Reported | Equivalence-off |
|---|---|---|
| large-v3 reference | 5.13% | 6.20% |
| minimum interval 0.40 s | 10.17% | 11.03% |

37 of 41 paired comparisons unchanged in sign and in whether the CI excludes
zero. Two changed sign with both CIs spanning zero (beam 3; compression 3.0);
two crossed the detection boundary (no-speech 0.40; repetition penalty 1.10),
both previously marginal, neither a selected candidate.

## 4. Insertion-inclusive scoring, n = 272
(`audit_272.csv`, `negation_insertion_per_recording.csv`,
`negation_paired_bootstrap.csv`)

| Condition | Deleted | Substituted | Hypothesis-only | Reference-anchored | Insertion-inclusive |
|---|---|---|---|---|---|
| large-v3 reference | 302 | 371 | 1,610 | 4.906% | 16.642% |
| minimum interval 0.40 s | 633 | 659 | 1,215 | 9.418% | 18.275% |

Recording-paired mean differences (272 recordings):

| Metric | Difference | 95% CI | d_z | Direction |
|---|---|---|---|---|
| Reference-anchored | +4.60 pp | +3.88 to +5.35 | 0.754 | 169 worse, 2 better |
| Hypothesis-only forms | −2.78 pp | −3.27 to −2.30 | −0.673 | 0 worse, 142 better |
| Insertion-inclusive | +1.82 pp | +1.00 to +2.68 | 0.258 | 120 worse, 72 better |

## 5. PriMock57 external evaluation of the branch selections
(`primock_ruleA.csv`, `primock_ruleC.csv`)

| Configuration | WER | WDER | SA-WER | DER | Role |
|---|---|---|---|---|---|
| large-v3 canonical baseline | 20.05% | 5.90% | 24.02% | 10.02% | 57/57 |
| large-v3 TPE trial 35 | 19.93% | 6.26% | 24.11% | 8.48% | 57/57 |
| compression threshold 2.0 (Rule A) | 19.59% | 6.12% | 23.72% | 8.51% | 57/57 |
| large-v2 + beam 3 (Rule C) | 19.38% | 4.86% | 22.61% | 8.53% | 56/57 |

WER gain over the reference, Fareez versus PriMock57: Rule A 0.82 → 0.46
points; Rule C 4.90 → 0.67 points.

Clinician−patient micro gap: Fareez +2.18; PriMock57 −3.71 (Rule A), −6.17
(Rule C).
