"""Unit tests for the disclosure-sharding condition suite.

Covers the two production contracts the suite is anchored on: dataset records
(``EvaluationRecord`` load/save) and the user-simulator goal resolution path
(``eva.utils.culture.resolve_user_goal``), plus the cross-condition delta
statistics that report the incremental-disclosure cost.
"""

import copy
from pathlib import Path

import pytest

from analysis.disclosure_sharding.condition_deltas import cross_condition_deltas, normalize_rows, summarize_conditions
from analysis.disclosure_sharding.disclosure_sharding import (
    condition_variants,
    goal_shards,
    parse_condition,
    split_into_shards,
    variant_id,
)
from eva.models.record import EvaluationRecord, GroundTruth
from eva.utils.culture import resolve_user_goal

AIRLINE_GOAL = (
    "You want to move your AUS to LAX flight from March 20 to March 25, arriving by 4:00 PM Pacific, "
    "keeping the total rebooking cost under $120, and keeping your window seat."
)

DATASET_PATH = Path(__file__).resolve().parents[3] / "data" / "airline_dataset.json"


def make_record(record_id: str = "1.1.2", goal: str = AIRLINE_GOAL) -> EvaluationRecord:
    """Create an airline-style record with a multi-constraint user goal."""
    return EvaluationRecord(
        id=record_id,
        user_goal={
            "high_level_user_goal": goal,
            "decision_tree": {
                "must_have_criteria": ["New travel date is 2026-03-25 for AUS→LAX."],
                "escalation_behavior": [],
                "nice_to_have_criteria": [],
                "negotiation_behavior": ["Confirm the total cost before booking."],
                "resolution_condition": "Rebooking confirmed",
                "failure_condition": "No suitable flight",
                "edge_cases": [],
            },
            "information_required": {"confirmation_number": "ABC123"},
            "starting_utterance": None,
        },
        user_config={"name": "Robert White", "gender": "man", "user_persona_id": 2, "user_persona": "direct"},
        current_date_time="2026-03-19 09:00:00",
        scenario_context={"dataset": "airline"},
        ground_truth=GroundTruth(expected_scenario_db={"seat": "window"}),
        starting_utterances={"en": "Hi, I need to change my flight to March 25."},
    )


class TestSplitIntoShards:
    """Deterministic clause-level sharding of goal statements."""

    def test_splits_on_clause_boundaries(self) -> None:
        """Commas separating disclosure-worthy constraints become shard boundaries."""
        shards = split_into_shards(AIRLINE_GOAL)
        assert shards == [
            "You want to move your AUS to LAX flight from March 20 to March 25",
            "arriving by 4:00 PM Pacific",
            "keeping the total rebooking cost under $120",
            "keeping your window seat.",
        ]

    def test_merges_fragments_too_short_to_disclose(self) -> None:
        """Clauses too short to stand alone are folded into the previous shard."""
        shards = split_into_shards("You want to book a flight, first class, to Tokyo.")
        assert shards == ["You want to book a flight, first class, to Tokyo."]

    def test_caps_shard_count_by_merging_the_tail(self) -> None:
        """The finest tail clauses merge until max_shards is respected, opener intact."""
        goal = (
            "You need a hotel, near the convention center, with a gym, under $200 a night, for three nights, in March."
        )
        shards = split_into_shards(goal, max_shards=3)
        assert len(shards) == 3
        assert shards[0] == "You need a hotel"

    def test_splits_single_clause_on_and(self) -> None:
        """A clause with no commas falls back to a guarded and-split."""
        shards = split_into_shards("You want a window seat and you need to arrive before noon.")
        assert shards == ["You want a window seat", "you need to arrive before noon."]


class TestConditionVariants:
    """Variant construction against the EvaluationRecord contract."""

    def test_variants_rewrite_only_the_disclosure_surface(self) -> None:
        """Variants differ from the source record only in id and goal statement."""
        record = make_record()
        original = record.model_dump()

        variants = condition_variants(record)

        assert [v.id for v in variants] == [f"1.1.2::{c}" for c in ("full", "concat", "sharded")]
        assert record.model_dump() == original, "source record must not be mutated"

        by_condition = {parse_condition(v.id)[1]: v for v in variants}
        assert by_condition["full"].user_goal["high_level_user_goal"] == AIRLINE_GOAL

        shards = split_into_shards(AIRLINE_GOAL)
        for condition in ("concat", "sharded"):
            goal_text = by_condition[condition].user_goal["high_level_user_goal"]
            assert all(shard in goal_text for shard in shards)

        # Everything except the id and the goal statement is carried over verbatim.
        for variant in variants:
            expected = copy.deepcopy(original)
            expected["id"] = variant.id
            expected["user_goal"]["high_level_user_goal"] = variant.user_goal["high_level_user_goal"]
            assert variant.model_dump() == expected

    def test_sharded_goal_stages_disclosure(self) -> None:
        """The sharded goal instructs one-requirement-per-turn disclosure."""
        sharded = condition_variants(make_record(), conditions=("sharded",))[0]
        goal_text = sharded.user_goal["high_level_user_goal"]
        assert goal_text.startswith("Disclose your requirements")
        assert "ONE PER TURN" in goal_text
        assert "Never state the full list up front" in goal_text

    def test_single_shard_goal_only_gets_full_variant(self) -> None:
        """Goals without internal boundaries only produce a full variant."""
        record = make_record(goal="You just want to chat about the weather today.")
        variants = condition_variants(record)
        assert [v.id for v in variants] == ["1.1.2::full"]

    def test_non_dict_goal_is_rejected(self) -> None:
        """Non-dict user goals raise instead of producing broken variants."""
        record = make_record()
        record.user_goal = "plain string goal"
        with pytest.raises(TypeError, match="high_level_user_goal"):
            condition_variants(record)
        with pytest.raises(TypeError, match="high_level_user_goal"):
            goal_shards(record)


class TestProductionContracts:
    """The variants must flow through the existing dataset and simulator-goal paths."""

    def test_dataset_round_trip(self, tmp_path: Path) -> None:
        """Variants survive the production dataset save/load path unchanged."""
        record = make_record()
        variants = condition_variants(record)

        EvaluationRecord.save_dataset(variants, tmp_path / "conditions.json")
        loaded = EvaluationRecord.load_dataset(tmp_path / "conditions.json")

        assert [v.id for v in loaded] == [v.id for v in variants]
        assert [v.user_goal["high_level_user_goal"] for v in loaded] == [
            v.user_goal["high_level_user_goal"] for v in variants
        ]
        assert loaded[0].ground_truth == record.ground_truth

    @pytest.mark.skipif(not DATASET_PATH.exists(), reason="airline dataset not present in this checkout")
    def test_real_dataset_variants_survive_culture_resolution(self) -> None:
        """Sharded variants flow through the runtime goal-resolution path."""
        record = EvaluationRecord.load_dataset(DATASET_PATH)[0]
        sharded = condition_variants(record, conditions=("sharded",))[0]

        resolved = resolve_user_goal(
            sharded.user_goal,
            record.culture_overrides,
            "en",
            None,
            record.starting_utterances,
        )

        assert resolved["starting_utterance"] == record.starting_utterances["en"]
        assert resolved["high_level_user_goal"].startswith("Disclose your requirements")
        assert resolved["decision_tree"] == record.user_goal["decision_tree"]


def make_trial_rows(n_scenarios: int = 6, n_trials: int = 2) -> list[dict[str, object]]:
    """Deterministic trial rows: concat/full at 1.0, sharded at 0.5."""
    rows: list[dict[str, object]] = []
    for i in range(n_scenarios):
        for condition, value in (("full", 1.0), ("concat", 1.0), ("sharded", 0.5)):
            for trial in range(n_trials):
                rows.append(
                    {
                        "system_alias": "test-model",
                        "scenario_id": f"s{i}::{condition}",
                        "metric": "task_completion",
                        "trial": trial,
                        "value": value,
                    }
                )
    return rows


class TestConditionDeltas:
    """Cross-condition statistics for disclosure runs."""

    def test_per_condition_summary(self) -> None:
        """Each condition reports its scenario count, trials, and mean with a CI."""
        rows = normalize_rows(make_trial_rows())
        summary = {row["condition"]: row for row in summarize_conditions(rows, n_boot=500)}
        assert set(summary) == {"full", "concat", "sharded"}
        for condition in summary:
            assert summary[condition]["n_scenarios"] == 6
            assert summary[condition]["n_trials"] == 12
        assert summary["concat"]["mean"] == pytest.approx(1.0)
        assert summary["sharded"]["mean"] == pytest.approx(0.5)

    def test_paired_deltas_report_disclosure_cost(self) -> None:
        """Sharded minus concat is negative and excludes zero; full minus concat is zero."""
        deltas = cross_condition_deltas(normalize_rows(make_trial_rows()), n_boot=500)
        by_pair = {(row["condition_a"], row["condition_b"]): row for row in deltas}

        sharded_drop = by_pair[("sharded", "concat")]
        assert sharded_drop["n_pairs"] == 6
        assert sharded_drop["mean_delta"] == pytest.approx(-0.5)
        assert sharded_drop["delta_ci_high"] < -0.3, "incremental-disclosure cost should exclude zero"

        reformulation = by_pair[("full", "concat")]
        assert reformulation["mean_delta"] == pytest.approx(0.0)

    def test_deltas_are_paired_and_deterministic(self) -> None:
        """Unpaired records are excluded and seeded bootstraps are reproducible."""
        rows = normalize_rows(make_trial_rows())
        # A record scored only under concat must not enter the paired deltas.
        rows.append(
            {
                "system_alias": "test-model",
                "scenario_id": "s99::concat",
                "metric": "task_completion",
                "trial": 0,
                "value": 0.0,
            }
        )
        deltas = cross_condition_deltas(rows, n_boot=500)
        assert all(row["n_pairs"] == 6 for row in deltas)
        assert deltas == cross_condition_deltas(rows, n_boot=500), "seeded bootstrap must be reproducible"

    def test_normalize_rows_drops_rows_without_condition_suffix(self) -> None:
        """Rows from non-variant records are dropped."""
        rows = make_trial_rows(n_scenarios=1, n_trials=1)
        rows.append(
            {
                "system_alias": "test-model",
                "scenario_id": "plain_record",
                "metric": "task_completion",
                "trial": 0,
                "value": 1.0,
            }
        )
        assert len(normalize_rows(rows)) == 3

    def test_normalize_rows_detects_alternate_columns(self) -> None:
        """record_id and alias columns are auto-detected."""
        rows = [
            {
                "alias": "test-model",
                "record_id": "s0::full",
                "metric": "task_completion",
                "trial": 0,
                "value": 1,
            }
        ]
        normalized = normalize_rows(rows)
        assert normalized[0]["system_alias"] == "test-model"
        assert normalized[0]["scenario_id"] == "s0::full"
        assert normalized[0]["value"] == 1.0

    def test_parse_condition_round_trip(self) -> None:
        """Variant ids parse back to base and condition."""
        assert parse_condition(variant_id("1.1.2", "sharded")) == ("1.1.2", "sharded")
        assert parse_condition("1.1.2") == ("1.1.2", None)
        assert parse_condition("a::b::not-a-condition") == ("a::b::not-a-condition", None)
