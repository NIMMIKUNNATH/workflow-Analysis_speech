"""
final_optimize_wer.py
=====================
Parameter optimisation for the faster-whisper + pyannote pipeline, scored on WER.

Design notes
------------
1.  Selection happens on a DEV split only. The holdout is touched exactly once,
    at the end, by report_holdout(). This is the difference between a tuned
    system and a system tuned on its own test set.
2.  Every configuration records secondary metrics (negation error, per-role WER)
    alongside WER, so that a WER-optimal configuration can be inspected against
    the metrics it may be trading away.
3.  select_best() supports a non-degradation screen: minimise WER subject to
    secondary metrics not regressing beyond a tolerance relative to a baseline.
    Unconstrained argmin-WER is available but is not the default.
4.  Results are cached to disk after every evaluation, so a long search can be
    killed and resumed.

Replace the two functions marked  >>> HOOK <<<  with calls into your existing
final_score_fareez_speaker.py / final_score_primock.py scorers so that the
metric definitions here stay identical to the rest of the project.
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import jiwer

try:
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    _NORMALIZER = EnglishTextNormalizer({})
except Exception:  # transformers not present; fall back to a crude normaliser
    import re

    def _NORMALIZER(text: str) -> str:  # type: ignore[misc]
        text = text.lower()
        text = re.sub(r"[^a-z0-9' ]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

NEGATION_CUES = {
    "no", "not", "n't", "never", "none", "nor", "neither", "without",
    "denies", "denied", "deny", "negative", "nothing", "nobody", "cannot",
}


def _tokens(text: str) -> list[str]:
    return _NORMALIZER(text).split()


def compute_metrics(refs: Sequence[str], hyps: Sequence[str]) -> dict[str, float]:
    """
    Corpus-level WER plus a negation-token error rate.

    Negation error rate = fraction of negation cue tokens present in the
    reference that are deleted or substituted in the hypothesis. A pipeline can
    look excellent on WER while losing exactly these tokens, which is why it is
    computed separately rather than folded into the aggregate.
    """
    if len(refs) != len(hyps):
        raise ValueError(f"ref/hyp count mismatch: {len(refs)} vs {len(hyps)}")

    ref_norm = [" ".join(_tokens(r)) for r in refs]
    hyp_norm = [" ".join(_tokens(h)) for h in hyps]

    # Drop empty references — jiwer cannot score them and they silently skew WER.
    pairs = [(r, h) for r, h in zip(ref_norm, hyp_norm) if r.strip()]
    if not pairs:
        raise ValueError("no non-empty references")
    ref_norm, hyp_norm = map(list, zip(*pairs))

    out = jiwer.process_words(ref_norm, hyp_norm)

    neg_total = 0
    neg_wrong = 0
    for ref_seq, alignment in zip(out.references, out.alignments):
        neg_total += sum(1 for t in ref_seq if t in NEGATION_CUES)
        for chunk in alignment:
            if chunk.type == "equal":
                continue
            span = ref_seq[chunk.ref_start_idx:chunk.ref_end_idx]
            neg_wrong += sum(1 for t in span if t in NEGATION_CUES)

    return {
        "wer": float(out.wer),
        "sub_rate": out.substitutions / max(1, sum(len(r) for r in out.references)),
        "del_rate": out.deletions / max(1, sum(len(r) for r in out.references)),
        "ins_rate": out.insertions / max(1, sum(len(r) for r in out.references)),
        "negation_error": (neg_wrong / neg_total) if neg_total else float("nan"),
        "negation_tokens": float(neg_total),
        "n_recordings": float(len(ref_norm)),
    }


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #

@dataclass
class SearchSpace:
    """
    Each entry maps a parameter name to the list of values to try.

    MODEL_KEYS are parameters that require reloading the model. They are kept
    separate so the search can be ordered to minimise reloads — reloading
    large-v2 for every configuration will dominate your wall-clock time.
    """

    grid: dict[str, list[Any]] = field(default_factory=lambda: {
        "model_name":              ["large-v2", "large-v3"],
        "compute_type":            ["float16"],
        "beam_size":               [1, 5],
        "condition_on_previous_text": [False],
        "temperature":             [0.0, (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)],
        "compression_ratio_threshold": [2.4],
        "log_prob_threshold":      [-1.0],
        "no_speech_threshold":     [0.6],
        "vad_filter":              [True],
        "vad_min_silence_ms":      [400, 700, 1000],
        "vad_speech_pad_ms":       [200, 400],
    })

    MODEL_KEYS = ("model_name", "compute_type")

    def configurations(self) -> list[dict[str, Any]]:
        keys = list(self.grid)
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*self.grid.values())]
        # Group by model so each model is loaded once.
        combos.sort(key=lambda c: tuple(str(c.get(k)) for k in self.MODEL_KEYS))
        return combos

    def sample(self, n: int, seed: int = 0) -> list[dict[str, Any]]:
        rng = random.Random(seed)
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for _ in range(n * 20):
            if len(out) >= n:
                break
            cfg = {k: rng.choice(v) for k, v in self.grid.items()}
            key = config_key(cfg)
            if key not in seen:
                seen.add(key)
                out.append(cfg)
        out.sort(key=lambda c: tuple(str(c.get(k)) for k in self.MODEL_KEYS))
        return out


def config_key(cfg: dict[str, Any]) -> str:
    """Stable, hashable identity for a configuration (used as the cache key)."""
    return json.dumps(cfg, sort_keys=True, default=str)


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #

class ModelCache:
    """Holds at most one loaded model to stay inside 24 GB of VRAM."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._key: str | None = None
        self._model: Any = None

    def get(self, model_name: str, compute_type: str):
        from faster_whisper import WhisperModel

        key = f"{model_name}|{compute_type}"
        if key != self._key:
            self._model = None  # release before allocating the replacement
            self._model = WhisperModel(model_name, device=self.device, compute_type=compute_type)
            self._key = key
        return self._model


def transcribe_one(model, audio_path: Path, cfg: dict[str, Any]) -> str:
    """>>> HOOK <<< Swap for your pipeline's transcribe call if it differs."""
    vad_params = {
        "min_silence_duration_ms": cfg["vad_min_silence_ms"],
        "speech_pad_ms": cfg["vad_speech_pad_ms"],
    }
    segments, _info = model.transcribe(
        str(audio_path),
        language="en",
        task="transcribe",
        beam_size=cfg["beam_size"],
        temperature=cfg["temperature"],
        condition_on_previous_text=cfg["condition_on_previous_text"],
        compression_ratio_threshold=cfg["compression_ratio_threshold"],
        log_prob_threshold=cfg["log_prob_threshold"],
        no_speech_threshold=cfg["no_speech_threshold"],
        vad_filter=cfg["vad_filter"],
        vad_parameters=vad_params,
    )
    return " ".join(s.text.strip() for s in segments)


# --------------------------------------------------------------------------- #
# Optimiser
# --------------------------------------------------------------------------- #

class ParameterOptimizer:
    def __init__(
        self,
        dev_items: Sequence[tuple[Path, str]],
        cache_path: Path = Path("wer_search_cache.json"),
        device: str = "cuda",
        metric_fn: Callable[[Sequence[str], Sequence[str]], dict[str, float]] = compute_metrics,
    ) -> None:
        self.dev_items = list(dev_items)      # (audio_path, reference_transcript)
        self.cache_path = Path(cache_path)
        self.models = ModelCache(device)
        self.metric_fn = metric_fn
        self.results: dict[str, dict[str, Any]] = {}
        if self.cache_path.exists():
            self.results = json.loads(self.cache_path.read_text())

    def _persist(self) -> None:
        self.cache_path.write_text(json.dumps(self.results, indent=2, default=str))

    def evaluate(self, cfg: dict[str, Any], items=None) -> dict[str, float]:
        items = self.dev_items if items is None else items
        key = config_key(cfg)
        if items is self.dev_items and key in self.results:
            return self.results[key]["metrics"]

        model = self.models.get(cfg["model_name"], cfg["compute_type"])
        hyps, refs = [], []
        for audio_path, reference in items:
            hyps.append(transcribe_one(model, audio_path, cfg))
            refs.append(reference)

        metrics = self.metric_fn(refs, hyps)
        if items is self.dev_items:
            self.results[key] = {"config": cfg, "metrics": metrics}
            self._persist()
        return metrics

    def search(self, configs: Iterable[dict[str, Any]], verbose: bool = True) -> list[dict[str, Any]]:
        configs = list(configs)
        for i, cfg in enumerate(configs, 1):
            metrics = self.evaluate(cfg)
            if verbose:
                print(
                    f"[{i}/{len(configs)}] WER={metrics['wer']:.4f} "
                    f"neg={metrics['negation_error']:.4f}  {cfg}",
                    flush=True,
                )
        return [self.results[config_key(c)] for c in configs]


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

def select_best(
    records: Sequence[dict[str, Any]],
    primary: str = "wer",
    baseline_config: dict[str, Any] | None = None,
    constraints: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Minimise `primary` subject to non-degradation constraints.

    constraints maps a secondary metric to the absolute tolerance by which it
    may exceed the baseline's value, e.g. {"negation_error": 0.005} permits at
    most half a percentage point of extra negation error in exchange for
    whatever WER gain the configuration offers.

    Passing constraints=None reproduces plain argmin-WER. That result is worth
    reporting, but selecting on it alone is what produced the 0.40 s interval
    configuration that halved WER gains against doubled negation error.
    """
    eligible = list(records)

    if constraints:
        if baseline_config is None:
            raise ValueError("constraints require a baseline_config")
        b_key = config_key(baseline_config)
        base = next((r for r in records if config_key(r["config"]) == b_key), None)
        if base is None:
            raise ValueError("baseline_config was not evaluated in this search")
        eligible = [
            r for r in records
            if all(
                r["metrics"].get(m, float("inf")) <= base["metrics"][m] + tol
                for m, tol in constraints.items()
            )
        ]

    if not eligible:
        return {"selected": None, "reason": "constraint screen returned the empty set",
                "n_candidates": len(records)}

    best = min(eligible, key=lambda r: r["metrics"][primary])
    return {
        "selected": best["config"],
        "metrics": best["metrics"],
        "n_eligible": len(eligible),
        "n_candidates": len(records),
    }


def paired_bootstrap(
    refs: Sequence[str],
    hyps_a: Sequence[str],
    hyps_b: Sequence[str],
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap over recordings for WER(a) - WER(b)."""
    rng = random.Random(seed)
    n = len(refs)
    idx = list(range(n))
    observed = compute_metrics(refs, hyps_a)["wer"] - compute_metrics(refs, hyps_b)["wer"]

    deltas = []
    for _ in range(n_boot):
        s = [rng.choice(idx) for _ in range(n)]
        r = [refs[i] for i in s]
        d = compute_metrics(r, [hyps_a[i] for i in s])["wer"] - \
            compute_metrics(r, [hyps_b[i] for i in s])["wer"]
        deltas.append(d)
    deltas.sort()

    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[int(0.975 * n_boot)]
    p = 2 * min(
        sum(1 for d in deltas if d >= 0) / n_boot,
        sum(1 for d in deltas if d <= 0) / n_boot,
    )
    return {"delta_wer": observed, "ci_low": lo, "ci_high": hi, "p_two_sided": min(1.0, p)}


def report_holdout(opt: ParameterOptimizer, cfg: dict[str, Any],
                   holdout_items: Sequence[tuple[Path, str]]) -> dict[str, float]:
    """Call this exactly once, on the single configuration you have committed to."""
    return opt.evaluate(cfg, items=holdout_items)


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # dev_items / holdout_items: [(Path("audio.wav"), "reference transcript"), ...]
    # Split speaker-disjointly, not recording-disjointly.
    dev_items: list[tuple[Path, str]] = []
    holdout_items: list[tuple[Path, str]] = []

    space = SearchSpace()
    opt = ParameterOptimizer(dev_items, cache_path=Path("wer_search_cache.json"))

    records = opt.search(space.configurations())

    baseline = {**space.configurations()[0], "model_name": "large-v2"}

    unconstrained = select_best(records)
    screened = select_best(
        records,
        baseline_config=baseline,
        constraints={"negation_error": 0.005},
    )

    print("\nargmin WER          :", json.dumps(unconstrained, indent=2, default=str))
    print("\nnon-degradation     :", json.dumps(screened, indent=2, default=str))

    if screened.get("selected"):
        print("\nholdout             :",
              json.dumps(report_holdout(opt, screened["selected"], holdout_items), indent=2))
