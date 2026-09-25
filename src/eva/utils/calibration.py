"""Calibration scoring for judge metrics.

Adapted from "Calibration as a First-Class Criterion in LLM Evaluation"
(arXiv:2609.26489), which argues that every performance metric should be paired
with a calibration score reporting whether confidence is empirically meaningful.

EVA's judges emit no explicit confidence (no logprobs, no self-reported score),
so confidence is derived from self-consistency across repeat runs: the share of
runs that agree with the modal verdict. Correctness is measured against human
labels, e.g. the ``expected_rating`` fields of the judge validation datasets in
``docs/metrics/judge_validation_datasets/``, letting calibration be reported
alongside the accuracy / macro-F1 numbers from that protocol.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eva.models.results import RecordMetrics
from eva.utils.pass_at_k import parse_trial_record_id

N_BINS = 10


@dataclass(frozen=True)
class ReliabilityBin:
    """One equal-width confidence bin of the reliability curve."""

    lower: float
    upper: float
    mean_confidence: float | None
    empirical_accuracy: float | None
    count: int


@dataclass(frozen=True)
class TrialAgreement:
    """Modal prediction and agreement share across a record's repeat runs."""

    prediction: Any
    confidence: float
    n_runs: int


@dataclass(frozen=True)
class CalibrationReport:
    """Calibration summary to report alongside a performance metric."""

    n: int
    n_bins: int
    mean_confidence: float | None
    accuracy: float | None
    ece: float | None
    brier: float | None
    # Signed gap between mean confidence and empirical accuracy:
    # positive means the judge is overconfident, negative underconfident.
    confidence_gap: float | None
    bins: list[ReliabilityBin]

    def summary_fields(self, decimals: int = 4) -> dict[str, float | int | None]:
        """Flat summary dict shaped like the bootstrap ``*_fields`` helpers."""
        return {
            "calibration_n": self.n,
            "mean_confidence": _round_or_none(self.mean_confidence, decimals),
            "judge_accuracy": _round_or_none(self.accuracy, decimals),
            "ece": _round_or_none(self.ece, decimals),
            "brier": _round_or_none(self.brier, decimals),
            "confidence_gap": _round_or_none(self.confidence_gap, decimals),
        }


def _round_or_none(value: float | None, decimals: int) -> float | None:
    """Round ``value``, passing ``None`` through untouched."""
    return round(value, decimals) if value is not None else None


def _validated_pair(
    confidences: Sequence[float],
    correctnesses: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce the two calibration inputs to equal-length float arrays."""
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correctnesses, dtype=float)
    if conf.shape != corr.shape:
        raise ValueError(
            f"confidences ({conf.size}) and correctnesses ({corr.size}) must have the same length"
        )
    return conf, corr


def _bin_indices(conf: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-width bin assignment on [0, 1], with 1.0 falling into the last bin."""
    return np.maximum(np.minimum((conf * n_bins).astype(int), n_bins - 1), 0)


def reliability_bins(
    confidences: Sequence[float],
    correctnesses: Sequence[float],
    n_bins: int = N_BINS,
) -> list[ReliabilityBin]:
    """Equal-width reliability bins over [0, 1].

    Args:
        confidences: One confidence score per example, in [0, 1].
        correctnesses: One binary correctness judgment (0/1) per example.
        n_bins: Number of equal-width confidence bins.

    Returns:
        One ``ReliabilityBin`` per bin (empty bins included, with ``None`` stats),
        for plotting reliability curves.
    """
    conf, corr = _validated_pair(confidences, correctnesses)
    indices = _bin_indices(conf, n_bins)
    bins: list[ReliabilityBin] = []
    for b in range(n_bins):
        mask = indices == b
        count = int(mask.sum())
        bins.append(
            ReliabilityBin(
                lower=b / n_bins,
                upper=(b + 1) / n_bins,
                mean_confidence=float(conf[mask].mean()) if count else None,
                empirical_accuracy=float(corr[mask].mean()) if count else None,
                count=count,
            )
        )
    return bins


def expected_calibration_error(
    confidences: Sequence[float],
    correctnesses: Sequence[float],
    n_bins: int = N_BINS,
) -> float | None:
    """Expected Calibration Error: bin-weighted |accuracy - confidence| gap.

    Args:
        confidences: One confidence score per example, in [0, 1].
        correctnesses: One binary correctness judgment (0/1) per example.
        n_bins: Number of equal-width confidence bins.

    Returns:
        ECE in [0, 1], or ``None`` for empty input.
    """
    conf, corr = _validated_pair(confidences, correctnesses)
    if conf.size == 0:
        return None
    indices = _bin_indices(conf, n_bins)
    ece = 0.0
    for b in range(n_bins):
        mask = indices == b
        count = int(mask.sum())
        if count == 0:
            continue
        ece += (count / conf.size) * abs(float(corr[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def brier_score(
    confidences: Sequence[float],
    correctnesses: Sequence[float],
) -> float | None:
    """Brier score: mean squared error between confidence and correctness.

    Args:
        confidences: One confidence score per example, in [0, 1].
        correctnesses: One binary correctness judgment (0/1) per example.

    Returns:
        Brier score in [0, 1] (lower is better), or ``None`` for empty input.
    """
    conf, corr = _validated_pair(confidences, correctnesses)
    if conf.size == 0:
        return None
    return float(np.mean((conf - corr) ** 2))


def calibration_report(
    confidences: Sequence[float],
    correctnesses: Sequence[float],
    n_bins: int = N_BINS,
) -> CalibrationReport:
    """Full calibration summary for one (confidence, correctness) sample.

    Args:
        confidences: One confidence score per example, in [0, 1].
        correctnesses: One binary correctness judgment (0/1) per example.
        n_bins: Number of equal-width confidence bins for ECE and the bins.

    Returns:
        ``CalibrationReport`` with ECE, Brier, the signed confidence gap, and
        the reliability bins.
    """
    conf, corr = _validated_pair(confidences, correctnesses)
    n = int(conf.size)
    return CalibrationReport(
        n=n,
        n_bins=n_bins,
        mean_confidence=float(conf.mean()) if n else None,
        accuracy=float(corr.mean()) if n else None,
        ece=expected_calibration_error(conf, corr, n_bins),
        brier=brier_score(conf, corr),
        confidence_gap=float(conf.mean() - corr.mean()) if n else None,
        bins=reliability_bins(conf, corr, n_bins),
    )


def repeat_agreement_confidence(per_record_predictions: Sequence[Sequence[Any]]) -> list[TrialAgreement]:
    """Modal prediction and agreement share for each record's repeat runs.

    Confidence here is self-consistency: the share of repeat predictions that
    agree with the modal one. Records with no predictions are skipped. Ties
    break toward the first prediction encountered.

    Args:
        per_record_predictions: One sequence of repeat-run predictions per record.

    Returns:
        One ``TrialAgreement`` per non-empty record, in input order.
    """
    agreements: list[TrialAgreement] = []
    for predictions in per_record_predictions:
        if not predictions:
            continue
        counts: Counter[Any] = Counter(predictions)
        modal, modal_count = max(counts.items(), key=lambda kv: kv[1])
        agreements.append(
            TrialAgreement(prediction=modal, confidence=modal_count / len(predictions), n_runs=len(predictions))
        )
    return agreements


def judge_calibration_from_repeats(
    per_record_predictions: Sequence[Sequence[Any]],
    expected_ratings: Sequence[Any],
    n_bins: int = N_BINS,
) -> CalibrationReport:
    """Calibration report for the judge-validation protocol.

    Confidence is the share of repeat runs agreeing with the modal rating;
    correctness is whether that modal rating matches the human ``expected_rating``.
    Pairs with the accuracy / macro-F1 numbers reported in
    ``docs/metrics/judge_validation_datasets/README.md``.

    Args:
        per_record_predictions: Judge ratings from each repeat run, per record.
        expected_ratings: Human label per record, aligned with the predictions.
        n_bins: Number of equal-width confidence bins.

    Returns:
        ``CalibrationReport`` over the records with at least one prediction.

    Raises:
        ValueError: If the two sequences have different lengths.
    """
    if len(per_record_predictions) != len(expected_ratings):
        raise ValueError(
            f"per_record_predictions ({len(per_record_predictions)}) and "
            f"expected_ratings ({len(expected_ratings)}) must have the same length"
        )
    labeled = [
        (predictions, expected)
        for predictions, expected in zip(per_record_predictions, expected_ratings)
        if predictions
    ]
    agreements = repeat_agreement_confidence([predictions for predictions, _ in labeled])
    confidences = [agreement.confidence for agreement in agreements]
    correctnesses = [float(agreement.prediction == expected) for agreement, (_, expected) in zip(agreements, labeled)]
    return calibration_report(confidences, correctnesses, n_bins)


def load_validation_labels(dataset_path: str | Path) -> dict[str, Any]:
    """Load human ``expected_rating`` labels from a judge validation dataset.

    Reads one JSON object per line. Records flagged ``invalid`` or missing an
    ``expected_rating`` are skipped, so the returned ids are exactly the
    labeled, validated subset.

    Args:
        dataset_path: Path to a ``docs/metrics/judge_validation_datasets/*.jsonl`` file.

    Returns:
        Dict mapping record id to its human rating.
    """
    labels: dict[str, Any] = {}
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("invalid") or record.get("expected_rating") is None:
                continue
            labels[str(record["id"])] = record["expected_rating"]
    return labels


def metric_trial_agreement(
    all_metrics: dict[str, RecordMetrics],
    metric_name: str,
    threshold: float = 0.5,
) -> dict[str, TrialAgreement]:
    """Per-scenario modal verdict and agreement share for one metric.

    Groups a run's records by base scenario id (``parse_trial_record_id``) and
    treats each trial's binary verdict (``get_score`` >= ``threshold``) as one
    repeat prediction. Trials where the metric is missing or errored are
    excluded, mirroring how pass@k treats invalid trials.

    Args:
        all_metrics: Dict mapping record ID to RecordMetrics, as produced by a run.
        metric_name: Judge metric whose verdicts to aggregate.
        threshold: Score threshold separating pass from fail verdicts.

    Returns:
        Dict mapping base scenario id to its ``TrialAgreement``.
    """
    grouped: dict[str, list[bool]] = {}
    for record_id, record_metrics in all_metrics.items():
        val = record_metrics.get_score(metric_name)
        if val is None:
            continue
        base_id, _ = parse_trial_record_id(record_id)
        grouped.setdefault(base_id, []).append(val >= threshold)
    agreements: dict[str, TrialAgreement] = {}
    for base_id, verdicts in grouped.items():
        runs = repeat_agreement_confidence([verdicts])
        if runs:
            agreements[base_id] = runs[0]
    return agreements


def judge_calibration_from_records(
    all_metrics: dict[str, RecordMetrics],
    metric_name: str,
    human_labels: dict[str, bool],
    threshold: float = 0.5,
    n_bins: int = N_BINS,
) -> CalibrationReport:
    """Calibration report for one judge metric over a multi-trial run.

    Combines ``metric_trial_agreement`` with human pass/fail labels keyed by
    base scenario id: confidence comes from cross-trial verdict agreement,
    correctness from agreement with the human label. Scenarios without a human
    label are excluded — calibration is computed on the labeled subset only.

    Args:
        all_metrics: Dict mapping record ID to RecordMetrics, as produced by a run.
        metric_name: Judge metric to score.
        human_labels: Human pass/fail verdict per base scenario id.
        threshold: Score threshold separating pass from fail verdicts.
        n_bins: Number of equal-width confidence bins.

    Returns:
        ``CalibrationReport`` over the labeled scenarios.
    """
    agreements = metric_trial_agreement(all_metrics, metric_name, threshold)
    confidences: list[float] = []
    correctnesses: list[float] = []
    for base_id, agreement in agreements.items():
        label = human_labels.get(base_id)
        if label is None:
            continue
        confidences.append(agreement.confidence)
        correctnesses.append(float(agreement.prediction == label))
    return calibration_report(confidences, correctnesses, n_bins)
