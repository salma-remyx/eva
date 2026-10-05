"""Tests for the rank-stability audit wired into the cross-run comparison.

Covers the joint cluster bootstrap itself and the wiring in
``apps.analysis.render_cross_run_comparison``, which is driven end-to-end on
synthetic run directories with the Streamlit app dependencies stubbed out.
"""

from __future__ import annotations

import contextlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from apps.rank_stability import FIRM_RANK_HOLD, compute_rank_stability, pairwise_frame, stability_frame


def _rows(system_values: dict[str, list[float]], metric: str = "EVA-A_mean") -> list[dict]:
    """Build cross-run per-record rows (system, record, metric) from value lists."""
    return [
        {"system": system, "record": str(i), metric: value}
        for system, values in system_values.items()
        for i, value in enumerate(values)
    ]


class TestComputeRankStability:
    """The joint cluster bootstrap behind the audit."""

    def test_separated_systems_hold_rank_and_are_firm(self):
        rows = _rows({"alpha": [0.9, 0.91, 0.92, 0.9], "beta": [0.5, 0.51, 0.52, 0.5], "gamma": [0.1, 0.11, 0.12, 0.1]})
        result = compute_rank_stability(rows, "EVA-A_mean")

        assert result is not None
        assert result.systems == ["alpha", "beta", "gamma"]
        assert result.observed_ranks == {"alpha": 1, "beta": 2, "gamma": 3}
        # Every record orders alpha > beta > gamma, so no replicate can reshuffle.
        assert result.rank_hold == {"alpha": 1.0, "beta": 1.0, "gamma": 1.0}
        assert result.pairwise[("alpha", "gamma")] == 1.0
        assert result.pairwise[("gamma", "alpha")] == 0.0
        assert result.n_records == 4

        frame = stability_frame(result)
        assert set(frame["Verdict"]) == {"firm"}
        assert frame.iloc[0]["System"] == "alpha"

    def test_overlapping_systems_are_flagged_unstable(self):
        # alpha and beta trade places record by record and tie on the mean.
        rows = _rows({"alpha": [0.9, 0.1] * 6, "beta": [0.5] * 12})
        result = compute_rank_stability(rows, "EVA-A_mean")

        assert result is not None
        assert result.rank_hold["alpha"] < FIRM_RANK_HOLD
        assert 0.05 < result.pairwise[("alpha", "beta")] < 0.95
        verdicts = dict(zip(stability_frame(result)["System"], stability_frame(result)["Verdict"]))
        assert verdicts["alpha"] == "unstable"

    def test_lower_is_better_flips_the_ranking(self):
        rows = _rows({"alpha": [0.9, 0.85], "beta": [0.1, 0.15]})

        higher = compute_rank_stability(rows, "EVA-A_mean", higher_is_better=True)
        lower = compute_rank_stability(rows, "EVA-A_mean", higher_is_better=False)

        assert higher is not None and lower is not None
        assert higher.systems == ["alpha", "beta"]
        assert lower.systems == ["beta", "alpha"]
        assert lower.pairwise[("beta", "alpha")] == 1.0

    def test_records_missing_for_one_system_are_dropped(self):
        rows = _rows({"alpha": [0.9, 0.8, 0.7], "beta": [0.5, 0.4, 0.3]})
        rows.append({"system": "gamma", "record": "0", "EVA-A_mean": 0.1})  # only one shared record

        assert compute_rank_stability(rows, "EVA-A_mean") is None

        rows += [
            {"system": "gamma", "record": "1", "EVA-A_mean": 0.2},
            {"system": "gamma", "record": "2", "EVA-A_mean": 0.1},
        ]
        result = compute_rank_stability(rows, "EVA-A_mean")
        assert result is not None
        assert result.n_records == 3  # gamma's orphan record was not part of the sample
        assert set(result.means) == {"alpha", "beta", "gamma"}

    def test_needs_two_systems_sharing_two_records(self):
        assert compute_rank_stability(_rows({"alpha": [0.9, 0.8]}), "EVA-A_mean") is None
        assert compute_rank_stability(_rows({"alpha": [0.9], "beta": [0.1]}), "EVA-A_mean") is None
        assert compute_rank_stability(_rows({"alpha": [0.9, None], "beta": [0.1, 0.2]}), "EVA-A_mean") is None

    def test_bootstrap_is_deterministic_for_a_fixed_seed(self):
        rows = _rows({"alpha": [0.9, 0.1] * 6, "beta": [0.5] * 12})

        first = compute_rank_stability(rows, "EVA-A_mean", seed=7)
        second = compute_rank_stability(rows, "EVA-A_mean", seed=7)

        assert first is not None and second is not None
        assert first.rank_hold == second.rank_hold
        assert first.pairwise == second.pairwise


class TestStabilityFrames:
    """The frames rendered from a RankStability result."""

    def test_pairwise_frame_is_labelled_with_diagonal_blank(self):
        rows = _rows({"alpha": [0.9, 0.8], "beta": [0.2, 0.1]})
        result = compute_rank_stability(rows, "EVA-A_mean")

        assert result is not None
        frame = pairwise_frame(result)
        assert list(frame.index) == ["alpha", "beta"]
        assert list(frame.columns) == ["alpha", "beta"]
        assert np.isnan(frame.loc["alpha", "alpha"])
        assert frame.loc["alpha", "beta"] == 1.0


class _StreamlitStub(types.ModuleType):
    """Minimal streamlit stand-in: widgets return their defaults, dataframes recorded."""

    def __init__(self) -> None:
        super().__init__("streamlit")
        self.session_state: dict = {}
        self.dataframe_calls: list = []
        self.sidebar = types.SimpleNamespace(toggle=self.toggle, selectbox=self.selectbox, multiselect=self.multiselect)
        self.column_config = types.SimpleNamespace(Column=lambda *a, **k: None, LinkColumn=lambda *a, **k: None)
        self.components = types.ModuleType("streamlit.components")
        self.components.v1 = types.ModuleType("streamlit.components.v1")

    def multiselect(self, label, options, default=None, **kwargs):
        return list(default) if default is not None else []

    def selectbox(self, label, options, index=0, **kwargs):
        return options[index] if options else None

    def toggle(self, label, value=False, **kwargs):
        return value

    def container(self, **kwargs):
        return contextlib.nullcontext(None)

    def cache_data(self, *args, **kwargs):
        # audio_plots decorates loaders with st.cache_data at import time.
        def decorator(func):
            return func

        return decorator

    def dataframe(self, obj, **kwargs):
        self.dataframe_calls.append(obj)

    def __getattr__(self, name):
        # markdown / caption / plotly_chart / download_button / space / ... are no-ops.
        return lambda *args, **kwargs: None


def _install_app_stub_modules(monkeypatch: pytest.MonkeyPatch) -> _StreamlitStub:
    """Stub the app-only dependencies so ``apps.analysis`` imports without the apps extras."""
    st_stub = _StreamlitStub()
    monkeypatch.setitem(sys.modules, "streamlit", st_stub)
    monkeypatch.setitem(sys.modules, "streamlit.components", st_stub.components)
    monkeypatch.setitem(sys.modules, "streamlit.components.v1", st_stub.components.v1)

    def _module(name: str, **attributes) -> types.ModuleType:
        module = types.ModuleType(name)
        for attr, value in attributes.items():
            setattr(module, attr, value)
        return module

    class _Trace:
        def __init__(self, *args, **kwargs):
            pass

    class _Figure:
        def __init__(self, *args, **kwargs):
            pass

        def add_trace(self, *args, **kwargs):
            pass

        def update_layout(self, *args, **kwargs):
            pass

    plotly = _module("plotly")
    plotly.express = _module("plotly.express")

    class _ColorScale(types.SimpleNamespace):
        # analysis.py concatenates px.colors.qualitative.* palettes at import time.
        def __getattr__(self, name):
            return ["#636efa"] * 12

    plotly.express.colors = types.SimpleNamespace(qualitative=_ColorScale(), sequential=_ColorScale())
    plotly.graph_objects = _module("plotly.graph_objects", Figure=_Figure, Bar=_Trace, Heatmap=_Trace)
    plotly.subplots = _module("plotly.subplots", make_subplots=lambda *a, **k: None)
    for name, module in {
        "plotly": plotly,
        "plotly.express": plotly.express,
        "plotly.graph_objects": plotly.graph_objects,
        "plotly.subplots": plotly.subplots,
        "diff_viewer": _module("diff_viewer", diff_viewer=lambda *a, **k: None),
        "librosa": _module("librosa"),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return st_stub


def _write_run_dir(base: Path, run_name: str, eva_a_values: list[float]) -> Path:
    """Create a run directory whose records carry one metric plus the EVA-A_mean composite."""
    run_dir = base / run_name
    for record_id, value in enumerate(eva_a_values):
        record_dir = run_dir / "records" / str(record_id)
        record_dir.mkdir(parents=True)
        metrics = {
            "record_id": str(record_id),
            "metrics": {
                "conversation_progression": {
                    "name": "conversation_progression",
                    "score": value,
                    "normalized_score": value,
                }
            },
            "aggregate_metrics": {"EVA-A_mean": value},
        }
        (record_dir / "metrics.json").write_text(json.dumps(metrics))
    return run_dir


class TestRenderCrossRunComparison:
    """End-to-end drive of the Streamlit page on synthetic run directories."""

    def test_cross_run_comparison_renders_rank_stability(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        st_stub = _install_app_stub_modules(monkeypatch)
        import apps.analysis as analysis  # imported late so the stubs above are in place

        n_records = 12
        wiggle = [0.0, 0.01, 0.02]
        runs = {"systemA": 0.90, "systemB": 0.55, "systemC": 0.10}
        run_dirs = [
            _write_run_dir(
                tmp_path, f"2026-01-01_10-00-00.00000{i + 1}_{name}", [base + wiggle[j % 3] for j in range(n_records)]
            )
            for i, (name, base) in enumerate(runs.items())
        ]

        analysis.render_cross_run_comparison(run_dirs)

        frames = [getattr(call, "data", call) for call in st_stub.dataframe_calls]
        stability = [f for f in frames if "Rank Hold" in getattr(f, "columns", [])]
        assert len(stability) == 1, "expected exactly one rank-stability table"
        table = stability[0]
        assert list(table["System"]) == [
            "systemA (2026-01-01_10-00-00.000001)",
            "systemB (2026-01-01_10-00-00.000002)",
            "systemC (2026-01-01_10-00-00.000003)",
        ]
        assert list(table["Rank"]) == [1, 2, 3]
        # The value ranges never overlap across systems, so every replicate keeps the ranking.
        assert list(table["Rank Hold"]) == [1.0, 1.0, 1.0]
        assert set(table["Verdict"]) == {"firm"}

        pairwise = [
            f for f in frames if list(getattr(f, "index", [])) == list(getattr(f, "columns", [])) and len(f) > 1
        ]
        assert pairwise, "expected a pairwise probability table"
        matrix = pairwise[0]
        assert matrix.loc[matrix.index[0], matrix.columns[-1]] == 1.0  # systemA beats systemC in every replicate
