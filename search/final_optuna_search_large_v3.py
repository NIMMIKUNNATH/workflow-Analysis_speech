"""
final_optuna_search.py
======================
TPE search over faster-whisper inference parameters, scored on corpus WER.

Differences from a naive Optuna script, each of which matters:

  * The objective actually runs the model. There is no dummy return.
  * Conditional parameters are only suggested when active (VAD params when
    v=1, best_of when the temperature schedule has fallbacks, patience and
    length_penalty when beam search is on). TPE wastes density on inactive
    dimensions otherwise.
  * Pruning is real: WER is reported after each recording and MedianPruner
    kills trials that are clearly losing, usually after 5-10 of 40 files.
  * Negation error is computed for every trial and stored as a user attribute,
    so the WER-optimal trial can be screened afterwards rather than selected
    blind.
  * Results persist to SQLite. Kill the run and resume with the same
    --study-name.

Read the caveat in select_final() before reporting anything from this.

Usage
-----
    python final_optuna_search.py --n-trials 100 --dev-n 40
    python final_optuna_search.py --resume --n-trials 50      # add more trials
    python final_optuna_search.py --report                    # summarise, no runs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import optuna

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scoring"))
from final_optimize_wer import compute_metrics  # normaliser + WER + negation

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA = Path(os.environ.get("ASR_DATA_ROOT", "./data"))
AUDIO_DIR = DATA / "Audio Recordings"
TRANS_DIR = DATA / "Clean Transcripts"

MODEL_NAME = "large-v3"      # comparison model
COMPUTE_TYPE = "float16"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def strip_speaker_prefixes(text: str) -> str:
    import re
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(D|P|DR|PT|Doctor|Patient)\s*[:\-]\s*", "", line, flags=re.I)
        if line.strip():
            out.append(line.strip())
    return " ".join(out)


def load_items() -> list[tuple[Path, str]]:
    items = []
    for audio in sorted(AUDIO_DIR.glob("*.mp3")):
        t = TRANS_DIR / f"{audio.stem}.txt"
        if t.exists():
            items.append((audio, strip_speaker_prefixes(
                t.read_text(encoding="utf-8", errors="replace"))))
    return items


def split_items(items, dev_n: int, seed: int = 0):
    """
    Dev/holdout split by recording. Cases are grouped by specialty prefix
    (CAR/RES/MSK/...) so the dev split is not accidentally all one specialty.
    """
    rng = random.Random(seed)
    by_prefix: dict[str, list] = {}
    for it in items:
        by_prefix.setdefault(it[0].stem[:3], []).append(it)

    dev, holdout = [], []
    for prefix, group in sorted(by_prefix.items()):
        rng.shuffle(group)
        take = max(1, round(dev_n * len(group) / len(items)))
        dev.extend(group[:take])
        holdout.extend(group[take:])
    return dev, holdout


# --------------------------------------------------------------------------- #
# Search space with conditional activation
# --------------------------------------------------------------------------- #

# Two schedules, not three. The full six-step fallback re-decodes failing
# segments up to six times; combined with beam 8 it ran ~17x slower than
# baseline in the first study and no trial completed in 2.8 hours. No
# deployment would use it, so it is not worth search budget.
TEMP_SCHEDULES = [
    (0.0,),
    (0.0, 0.2, 0.4),
]

# Wall-clock budget per trial, as a multiple of the expected baseline cost.
# A trial exceeding it is pruned rather than allowed to stall the study.
TRIAL_TIME_BUDGET_X = 4.0
SECONDS_PER_RECORDING_BASELINE = 30.0


def suggest_phi(trial: optuna.Trial) -> dict:
    phi: dict = {"model_name": MODEL_NAME, "compute_type": COMPUTE_TYPE}

    # --- VAD block --------------------------------------------------------
    # v is fixed at 1. With VAD off, silence is fed to the decoder and a
    # 12-minute consultation costs several times more for no accuracy gain.
    phi["v"] = 1
    phi["tau"] = trial.suggest_categorical("tau", [0.30, 0.50, 0.70])
    phi["d_min"] = trial.suggest_categorical("d_min", [100, 250, 500])
    phi["d_sil"] = trial.suggest_categorical("d_sil", [250, 500, 1000])
    phi["d_pad"] = trial.suggest_categorical("d_pad", [200, 400, 600])

    # --- decoding block ----------------------------------------------------
    t_idx = trial.suggest_categorical("T_idx", [0, 1])
    phi["T"] = TEMP_SCHEDULES[t_idx]

    phi["b"] = trial.suggest_categorical("b", [1, 3, 5])
    if phi["b"] > 1:
        # patience and length_penalty only affect beam search
        phi["p"] = trial.suggest_categorical("p", [1.0, 1.5, 2.0])
        phi["alpha"] = trial.suggest_categorical("alpha", [0.8, 1.0, 1.2])
    if len(phi["T"]) > 1:
        # best_of only applies when temperature fallback can trigger sampling
        phi["k"] = trial.suggest_categorical("k", [3, 5, 8])

    phi["r"] = trial.suggest_categorical("r", [1.0, 1.1])
    phi["n"] = trial.suggest_categorical("n", [0, 3])

    # --- filtering / context ----------------------------------------------
    phi["q_ns"] = trial.suggest_categorical("q_ns", [0.40, 0.60, 0.75, 0.90])
    phi["q_lp"] = trial.suggest_categorical("q_lp", [-1.5, -1.0, -0.5])
    phi["q_cr"] = trial.suggest_categorical("q_cr", [2.0, 2.4, 3.0])
    phi["h"] = trial.suggest_categorical("h", [0, 1])

    return phi


def to_kwargs(phi: dict) -> dict:
    """Map the phi vector onto faster-whisper's transcribe() signature."""
    kw = dict(
        language="en",
        task="transcribe",
        beam_size=phi["b"],
        temperature=phi["T"],
        repetition_penalty=phi["r"],
        no_repeat_ngram_size=phi["n"],
        no_speech_threshold=phi["q_ns"],
        log_prob_threshold=phi["q_lp"],
        compression_ratio_threshold=phi["q_cr"],
        condition_on_previous_text=bool(phi["h"]),
        vad_filter=bool(phi["v"]),
    )
    if "p" in phi:
        kw["patience"] = phi["p"]
    if "alpha" in phi:
        kw["length_penalty"] = phi["alpha"]
    if "k" in phi:
        kw["best_of"] = phi["k"]
    if phi["v"] == 1:
        kw["vad_parameters"] = {
            "threshold": phi["tau"],
            "min_speech_duration_ms": phi["d_min"],
            "min_silence_duration_ms": phi["d_sil"],
            "speech_pad_ms": phi["d_pad"],
        }
    return kw


# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel
        _MODEL = WhisperModel(MODEL_NAME, device="cuda", compute_type=COMPUTE_TYPE)
    return _MODEL


def make_objective(dev_items):
    budget = (TRIAL_TIME_BUDGET_X * SECONDS_PER_RECORDING_BASELINE
              * len(dev_items))

    def objective(trial: optuna.Trial) -> float:
        import time
        phi = suggest_phi(trial)
        kwargs = to_kwargs(phi)
        model = get_model()
        t0 = time.time()

        refs, hyps = [], []
        for step, (audio, ref) in enumerate(dev_items):
            segments, _ = model.transcribe(str(audio), **kwargs)
            hyps.append(" ".join(s.text.strip() for s in segments))
            refs.append(ref)

            elapsed = time.time() - t0
            if elapsed > budget:
                # This configuration is too slow to be a deployable candidate.
                trial.set_user_attr("aborted", "time budget exceeded")
                trial.set_user_attr("elapsed_s", round(elapsed))
                raise optuna.TrialPruned()

            # Report running WER so the pruner can kill hopeless trials early.
            if step >= 4:
                running = compute_metrics(refs, hyps)["wer"]
                trial.report(running, step)
                if trial.should_prune():
                    trial.set_user_attr("pruned_at", step + 1)
                    raise optuna.TrialPruned()

        m = compute_metrics(refs, hyps)
        trial.set_user_attr("elapsed_s", round(time.time() - t0))
        trial.set_user_attr("negation_error", m["negation_error"])
        trial.set_user_attr("negation_tokens", m["negation_tokens"])
        trial.set_user_attr("del_rate", m["del_rate"])
        trial.set_user_attr("ins_rate", m["ins_rate"])
        trial.set_user_attr("phi", json.dumps(phi, default=str))
        return m["wer"]

    return objective


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def select_final(study: optuna.Study, neg_tolerance: float = 0.005) -> dict:
    """
    CAVEAT, and it is not a small one.

    argmin over N trials on a dev set of n recordings carries an expected
    optimism of roughly sigma * sqrt(2 * ln N), where sigma is the
    trial-to-trial noise in dev WER. At n=40 the paired bootstrap put sigma
    near 1 percentage point, so 100 trials inflates the winning dev WER by
    about 3pp purely by selection. The dev WER printed here is therefore NOT
    reportable. Only the holdout number is.

    The negation screen below uses point estimates, which on 40 recordings
    cannot resolve differences below roughly 1pp. Treat a failed screen as a
    flag to test, not as grounds for rejection.
    """
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        return {"error": "no completed trials"}

    best = min(done, key=lambda t: t.value)
    negs = [t for t in done if not math.isnan(t.user_attrs.get("negation_error", float("nan")))]
    best_neg = min(negs, key=lambda t: t.user_attrs["negation_error"]) if negs else None

    screened = None
    if best_neg is not None:
        anchor = best_neg.user_attrs["negation_error"]
        eligible = [t for t in negs
                    if t.user_attrs["negation_error"] <= anchor + neg_tolerance]
        if eligible:
            s = min(eligible, key=lambda t: t.value)
            screened = {"trial": s.number, "dev_wer": s.value,
                        "negation_error": s.user_attrs["negation_error"],
                        "phi": json.loads(s.user_attrs["phi"])}

    return {
        "n_complete": len(done),
        "n_pruned": sum(1 for t in study.trials
                        if t.state == optuna.trial.TrialState.PRUNED),
        "argmin_wer": {"trial": best.number, "dev_wer": best.value,
                       "negation_error": best.user_attrs.get("negation_error"),
                       "phi": json.loads(best.user_attrs["phi"])},
        "screened": screened,
        "expected_selection_optimism_pp":
            round(1.0 * math.sqrt(2 * math.log(max(2, len(done)))), 2),
    }


def evaluate_phi(phi: dict, items) -> dict:
    kwargs = to_kwargs(phi)
    model = get_model()
    refs, hyps = [], []
    for audio, ref in items:
        segments, _ = model.transcribe(str(audio), **kwargs)
        hyps.append(" ".join(s.text.strip() for s in segments))
        refs.append(ref)
    return compute_metrics(refs, hyps)


# --------------------------------------------------------------------------- #

def build_storage(spec: str):
    """
    'journal:path.log' -> JournalStorage (safe for concurrent workers)
    anything else       -> passed through as an RDB URL

    SQLite raises 'database is locked' once two or more workers write
    concurrently. Journal storage is file-locked and designed for this.
    """
    if spec.startswith("journal:"):
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend
        return JournalStorage(JournalFileBackend(spec.split(":", 1)[1]))
    return spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--dev-n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--study-name", default="fareez_phi_search")
    ap.add_argument("--storage", default="sqlite:///optuna_fareez.db")
    ap.add_argument("--worker-id", type=int, default=0,
                    help="offsets the sampler seed so parallel workers explore "
                         "different regions; the split stays identical")
    ap.add_argument("--report", action="store_true",
                    help="summarise an existing study without running trials")
    ap.add_argument("--holdout", action="store_true",
                    help="evaluate the selected phi once on the holdout split")
    args = ap.parse_args()

    items = load_items()
    dev, holdout = split_items(items, args.dev_n, seed=args.seed)
    print(f"{len(items)} recordings: {len(dev)} dev, {len(holdout)} holdout")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=build_storage(args.storage),
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=args.seed + args.worker_id, n_startup_trials=20),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )

    if not args.report:
        study.optimize(make_objective(dev), n_trials=args.n_trials,
                       show_progress_bar=(args.worker_id == 0))

    result = select_final(study)
    print("\n" + "=" * 60)
    print(json.dumps(result, indent=2, default=str))
    print("=" * 60)
    print(f"\nDev WER above is inflated by roughly "
          f"{result.get('expected_selection_optimism_pp')}pp of selection "
          f"optimism. Do not report it.")

    if args.holdout and "argmin_wer" in result:
        phi = result["argmin_wer"]["phi"]
        print(f"\nEvaluating on {len(holdout)} held-out recordings...")
        m = evaluate_phi(phi, holdout)
        print(json.dumps(m, indent=2))
        print("\nThis is the reportable number.")


if __name__ == "__main__":
    main()
