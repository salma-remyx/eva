"""Tests for the LLM-as-a-Verifier scoring primitive.

Covers :func:`eva.metrics.verifier_scoring.expectation_score_from_logprobs`,
which converts a completion's scoring-token logprob distribution into a
continuous expected-rating score. Provider logprob objects are exercised both as
objects (litellm default) and as dicts, and the leading-token scan, chosen-token
fallback, and empty-input cases are all covered.
"""

from types import SimpleNamespace

import pytest

from eva.metrics.verifier_scoring import expectation_score_from_logprobs


def _entry(token: str, logprob: float, top_logprobs: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(token=token, logprob=logprob, top_logprobs=top_logprobs or [])


def _top(token: str, logprob: float) -> SimpleNamespace:
    return SimpleNamespace(token=token, logprob=logprob)


def test_equal_mass_over_full_scale_gives_scale_midpoint():
    entries = [_entry("2", -0.3, [_top("1", 0.0), _top("2", 0.0), _top("3", 0.0)])]

    dist = expectation_score_from_logprobs(SimpleNamespace(content=entries), (1, 3))

    assert dist is not None
    assert dist.expectation == pytest.approx(2.0)
    assert dist.probabilities == pytest.approx({1: 1 / 3, 2: 1 / 3, 3: 1 / 3})
    assert dist.from_top_logprobs is True


def test_all_mass_on_top_rating():
    entries = [_entry("3", 0.0, [_top("3", 0.0)])]

    dist = expectation_score_from_logprobs(SimpleNamespace(content=entries), (1, 3))

    assert dist is not None
    assert dist.expectation == pytest.approx(3.0)
    assert dist.probabilities == {3: 1.0}


def test_splits_evenly_between_two_ratings():
    entries = [_entry("3", 0.0, [_top("2", 0.0), _top("3", 0.0)])]

    dist = expectation_score_from_logprobs(SimpleNamespace(content=entries), (1, 3))

    assert dist is not None
    assert dist.probabilities == pytest.approx({2: 0.5, 3: 0.5})
    assert dist.expectation == pytest.approx(2.5)


def test_probability_weighted_by_logprob_magnitude():
    entries = [_entry("3", 0.0, [_top("1", -5.0), _top("3", 0.0)])]

    dist = expectation_score_from_logprobs(SimpleNamespace(content=entries), (1, 3))

    assert dist is not None
    assert dist.probabilities[3] > dist.probabilities[1]
    assert 2.7 < dist.expectation < 3.0


def test_uses_chosen_token_logprob_when_top_logprobs_absent():
    entries = [_entry("2", -0.5, [])]

    dist = expectation_score_from_logprobs(SimpleNamespace(content=entries), (1, 3))

    assert dist is not None
    assert dist.expectation == pytest.approx(2.0)
    assert dist.probabilities == {2: 1.0}
    assert dist.from_top_logprobs is False


def test_scans_past_non_rating_leading_token():
    # The model emitted a leading quote before the rating digit.
    entries = [_entry("'", -5.0), _entry("3", -0.1, [_top("3", -0.1)])]

    dist = expectation_score_from_logprobs(SimpleNamespace(content=entries), (1, 3))

    assert dist is not None
    assert dist.scoring_token == "3"
    assert dist.expectation == pytest.approx(3.0)


def test_accepts_dict_shaped_provider_entries():
    entries = [
        {
            "token": "3",
            "logprob": -0.1,
            "top_logprobs": [{"token": "3", "logprob": -0.1}, {"token": "2", "logprob": -3.0}],
        }
    ]

    dist = expectation_score_from_logprobs({"content": entries}, (1, 3))

    assert dist is not None
    assert dist.expectation == pytest.approx(3.0, abs=0.1)


def test_returns_none_when_no_scale_token_present():
    entries = [_entry("Sure", -0.1), _entry(",", -0.2)]

    assert expectation_score_from_logprobs(SimpleNamespace(content=entries), (1, 3)) is None


def test_returns_none_for_missing_or_empty_logprobs():
    assert expectation_score_from_logprobs(None, (1, 3)) is None
    assert expectation_score_from_logprobs(SimpleNamespace(content=[]), (1, 3)) is None
    assert expectation_score_from_logprobs({}, (1, 3)) is None
