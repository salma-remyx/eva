"""Unit tests for judge calibration scoring."""

from pathlib import Path

import pytest

from eva.models.results import MetricScore, RecordMetrics
from eva.utils.calibration import (
    brier_score,
    calibration_report,
    expected_calibration_error,
    judge_calibration_from_records,
    judge_calibration_from_repeats,
    load_validation_labels,
    metric_trial_agreement,
    reliability_bins,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VALIDATION_DIR = _REPO_ROOT / "docs" / "metrics" / "judge_validation_datasets"


def _make_trial_record(record_id: str, score: float | None, *, error: str | None = None) -> RecordMetrics:
    """Build a RecordMetrics holding one judge score, like a run's metrics.json."""
    return RecordMetrics(
        record_id=record_id,
        metrics={
            "faithfulness": MetricScore(
                name="faithfulness",
                score=score,
                normalized_score=score,
                error=error,
            )
        },
    )


class TestExpectedCalibrationError:
    def test_perfectly_calibrated(self):
        assert expected_calibration_error([1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]) == 0.0

    def test_fully_overconfident_half_wrong(self):
        # Confidence 1.0 everywhere but only half correct: ECE collapses to the accuracy gap.
        ece = expected_calibration_error([1.0] * 4, [1.0, 1.0, 0.0, 0.0])
        assert ece == pytest.approx(0.5)

    def test_empty_input_returns_none(self):
        assert expected_calibration_error([], []) is None

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="must have the same length"):
            expected_calibration_error([0.5], [])


class TestBrierScore:
    def test_known_values(self):
        assert brier_score([0.8, 0.8], [1.0, 0.0]) == pytest.approx(0.34)

    def test_perfect(self):
        assert brier_score([1.0, 0.0], [1.0, 0.0]) == 0.0

    def test_empty_input_returns_none(self):
        assert brier_score([], []) is None


class TestReliabilityBins:
    def test_bins_cover_all_examples(self):
        bins = reliability_bins([0.05, 0.95, 0.55], [1.0, 0.0, 1.0], n_bins=10)
        assert len(bins) == 10
        assert sum(b.count for b in bins) == 3
        occupied = [b for b in bins if b.count]
        assert [(b.lower, b.upper) for b in occupied] == [(0.0, 0.1), (0.5, 0.6), (0.9, 1.0)]
        assert occupied[1].mean_confidence == pytest.approx(0.55)
        assert occupied[1].empirical_accuracy == pytest.approx(1.0)

    def test_empty_bins_have_none_stats(self):
        bins = reliability_bins([], [], n_bins=4)
        assert all(b.count == 0 and b.mean_confidence is None and b.empirical_accuracy is None for b in bins)


class TestCalibrationReport:
    def test_overconfident_report(self):
        report = calibration_report([1.0, 1.0], [1.0, 0.0])
        assert report.n == 2
        assert report.accuracy == pytest.approx(0.5)
        assert report.confidence_gap == pytest.approx(0.5)
        assert report.ece == pytest.approx(0.5)
        assert report.brier == pytest.approx(0.5)

    def test_summary_fields_rounding(self):
        fields = calibration_report([2 / 3, 2 / 3], [1.0, 1.0]).summary_fields()
        assert set(fields) == {"calibration_n", "mean_confidence", "judge_accuracy", "ece", "brier", "confidence_gap"}
        assert fields["calibration_n"] == 2
        assert fields["mean_confidence"] == pytest.approx(0.6667)
        assert fields["confidence_gap"] == pytest.approx(-0.3333)


class TestJudgeCalibrationFromRepeats:
    def test_self_consistent_correct_judge_is_perfectly_calibrated(self):
        report = judge_calibration_from_repeats([[3, 3, 3], [1, 1, 1], [2, 2, 2]], [3, 1, 2])
        assert report.n == 3
        assert report.ece == 0.0
        assert report.brier == 0.0
        assert report.accuracy == 1.0

    def test_flip_flopping_judge_is_underconfident(self):
        # Both modal verdicts are right, so accuracy is 1.0 while confidence is 2/3.
        report = judge_calibration_from_repeats([[1, 1, 2], [3, 3, 1]], [1, 3])
        assert report.accuracy == 1.0
        assert report.mean_confidence == pytest.approx(2 / 3)
        assert report.confidence_gap == pytest.approx(-1 / 3)
        assert report.ece == pytest.approx(1 / 3)
        assert report.brier == pytest.approx(1 / 9)

    def test_confidently_wrong_judge_is_overconfident(self):
        report = judge_calibration_from_repeats([[1, 1, 1], [1, 1, 1]], [1, 0])
        assert report.confidence_gap == pytest.approx(0.5)
        assert report.ece == pytest.approx(0.5)

    def test_records_without_predictions_are_skipped(self):
        report = judge_calibration_from_repeats([[1, 1, 1], []], [1, 1])
        assert report.n == 1

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="must have the same length"):
            judge_calibration_from_repeats([[1]], [1, 2])


class TestMetricTrialAgreement:
    def test_groups_trials_by_base_scenario(self):
        records = {
            "1.1.1/trial_0": _make_trial_record("1.1.1/trial_0", 1.0),
            "1.1.1/trial_1": _make_trial_record("1.1.1/trial_1", 1.0),
            "1.1.1/trial_2": _make_trial_record("1.1.1/trial_2", 0.0),
            # Flat trial naming (backward compat) groups with the same base id.
            "1.1.2_trial_0": _make_trial_record("1.1.2_trial_0", 0.0),
            "1.1.2_trial_1": _make_trial_record("1.1.2_trial_1", 0.0),
        }
        agreements = metric_trial_agreement(records, "faithfulness")
        assert set(agreements) == {"1.1.1", "1.1.2"}
        assert agreements["1.1.1"].prediction is True
        assert agreements["1.1.1"].confidence == pytest.approx(2 / 3)
        assert agreements["1.1.1"].n_runs == 3
        assert agreements["1.1.2"].prediction is False
        assert agreements["1.1.2"].confidence == 1.0

    def test_errored_and_missing_trials_are_excluded(self):
        records = {
            "1.1.3/trial_0": _make_trial_record("1.1.3/trial_0", None, error="judge call failed"),
            "1.1.3/trial_1": _make_trial_record("1.1.3/trial_1", 1.0),
            # No faithfulness metric at all.
            "1.1.4/trial_0": RecordMetrics(record_id="1.1.4/trial_0", metrics={}),
        }
        agreements = metric_trial_agreement(records, "faithfulness")
        assert set(agreements) == {"1.1.3"}
        assert agreements["1.1.3"].n_runs == 1


class TestJudgeCalibrationFromRecords:
    def test_calibration_over_labeled_subset_only(self):
        records = {
            "1.1.1/trial_0": _make_trial_record("1.1.1/trial_0", 1.0),
            "1.1.1/trial_1": _make_trial_record("1.1.1/trial_1", 1.0),
            "1.1.1/trial_2": _make_trial_record("1.1.1/trial_2", 0.0),
            "1.1.2/trial_0": _make_trial_record("1.1.2/trial_0", 0.0),
            "1.1.2/trial_1": _make_trial_record("1.1.2/trial_1", 0.0),
            # Unlabeled scenario: must not count towards calibration.
            "1.1.5/trial_0": _make_trial_record("1.1.5/trial_0", 1.0),
        }
        labels = {"1.1.1": True, "1.1.2": False}
        report = judge_calibration_from_records(records, "faithfulness", labels)
        assert report.n == 2
        assert report.accuracy == 1.0
        assert report.mean_confidence == pytest.approx(5 / 6)
        assert report.confidence_gap == pytest.approx(-1 / 6)
        assert report.ece == pytest.approx(1 / 6)

    def test_wrong_verdicts_score_as_incorrect(self):
        records = {
            "1.1.1/trial_0": _make_trial_record("1.1.1/trial_0", 1.0),
            "1.1.1/trial_1": _make_trial_record("1.1.1/trial_1", 1.0),
            "1.1.1/trial_2": _make_trial_record("1.1.1/trial_2", 0.0),
            "1.1.2/trial_0": _make_trial_record("1.1.2/trial_0", 0.0),
            "1.1.2/trial_1": _make_trial_record("1.1.2/trial_1", 0.0),
        }
        report = judge_calibration_from_records(records, "faithfulness", {"1.1.1": False, "1.1.2": True})
        assert report.accuracy == 0.0
        assert report.ece == pytest.approx(5 / 6)


class TestLoadValidationLabels:
    def test_faithfulness_dataset(self):
        labels = load_validation_labels(_VALIDATION_DIR / "faithfulness.jsonl")
        assert len(labels) == 137
        assert set(labels.values()) == {1, 2, 3}

    def test_invalid_records_are_excluded(self):
        # 136 records, 2 flagged invalid in the progression dataset.
        labels = load_validation_labels(_VALIDATION_DIR / "conversation_progression.jsonl")
        assert len(labels) == 134
        assert set(labels.values()) == {1, 2, 3}
