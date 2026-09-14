"""Gold-blind capability-binding preflight for the configured serving route.

Adapted from the capability-binding preflight of IB2 ("IBIB: A Protocol for
Measuring Enterprise AI Systems by Serving Route, Not Model Identifier",
arXiv:2609.10494): before any task reaches a route, verify that the route can
execute the evaluation contract. A benchmark that merely records the serving
route (as EVA does in ``config.json``) can still burn a full run's worth of
conversations on a route that can never produce the requested measurements.

The preflight is gold-blind — it inspects only the run configuration and the
metric registry, never ground truth or scores. Route binding uses each
metric's declared ``supported_pipeline_types`` instead of the paper's live
route probe: the registry already states which pipelines a metric can execute
on, so an unbindable contract is detectable before the first conversation.

Failure semantics follow the paper's classification: a route that cannot
execute any requested metric (or is missing cascade components) fails the
gate and the run aborts before conversations start; individual metrics that
are merely unsupported on this route are excluded and reported, matching how
``MetricsRunner`` already skips them per record.
"""

from dataclasses import dataclass, field
from typing import Any

from eva.metrics.aggregation import EVA_COMPOSITES
from eva.metrics.registry import get_global_registry
from eva.models.config import PipelineType, RunConfig


class RouteContractError(RuntimeError):
    """Raised when the serving route cannot execute the evaluation contract."""


@dataclass
class RoutePreflightResult:
    """Outcome of the capability-binding preflight for one serving route."""

    pipeline_type: PipelineType
    route: dict[str, Any]
    bound_metrics: list[str] = field(default_factory=list)
    unsupported_metrics: list[str] = field(default_factory=list)
    unknown_metrics: list[str] = field(default_factory=list)
    lever_warnings: list[str] = field(default_factory=list)
    partial_composites: dict[str, list[str]] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when the route satisfies every binding predicate."""
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form persisted as ``route_preflight.json``."""
        return {
            "passed": self.passed,
            "pipeline_type": str(self.pipeline_type),
            "route": self.route,
            "bound_metrics": self.bound_metrics,
            "unsupported_metrics": self.unsupported_metrics,
            "unknown_metrics": self.unknown_metrics,
            "lever_warnings": self.lever_warnings,
            "partial_composites": self.partial_composites,
            "failures": self.failures,
        }


def check_route_contract(config: RunConfig) -> RoutePreflightResult:
    """Verify the configured serving route can execute the requested metrics contract.

    Args:
        config: The run configuration (serving route + requested metrics).

    Returns:
        A :class:`RoutePreflightResult` whose ``failures`` list is non-empty
        exactly when the route cannot execute the contract.
    """
    model = config.model
    pipeline_type = model.pipeline_type
    registry = get_global_registry()

    requested = list(config.metrics or [])
    bound: list[str] = []
    unsupported: list[str] = []
    unknown: list[str] = []
    for name in requested:
        metric_class = registry.get(name)
        if metric_class is None:
            unknown.append(name)
        elif pipeline_type in metric_class.supported_pipeline_types:
            bound.append(name)
        else:
            unsupported.append(name)

    failures: list[str] = []
    if requested and not bound:
        failures.append(
            f"no requested metric can execute on pipeline '{pipeline_type}' "
            f"(unsupported: {sorted(unsupported)}, unknown: {sorted(unknown)})"
        )

    if pipeline_type == PipelineType.CASCADE:
        missing = [part for part, value in (("stt", model.stt), ("tts", model.tts), ("llm", model.llm)) if not value]
        if missing:
            failures.append(f"cascade route is missing components: {missing}")

    # Cascade latency levers only bind to the cascade pipeline; on other
    # pipelines they are silently ignored, so the measured route differs from
    # the advertised configuration. Reported as a warning, not a failure.
    active_levers = [
        lever
        for lever, is_set in (
            ("llm_streaming", model.llm_streaming),
            ("pre_tool_speech", model.pre_tool_speech != "off"),
            ("parallel_tool_calls", model.parallel_tool_calls is not None),
        )
        if is_set
    ]
    lever_warnings: list[str] = []
    if active_levers and pipeline_type != PipelineType.CASCADE:
        lever_warnings.append(
            f"latency levers {sorted(active_levers)} do not apply to pipeline '{pipeline_type}' and are ignored"
        )

    # Composites whose components are not all requested cannot be produced by
    # this run — informational, since a metric subset may be intentional.
    requested_set = set(requested)
    partial_composites = {
        comp.name: [m for m in comp.component_metrics if m not in requested_set]
        for comp in EVA_COMPOSITES
        if comp.component_metrics and not set(comp.component_metrics) <= requested_set
    }

    return RoutePreflightResult(
        pipeline_type=pipeline_type,
        route={
            "stt": model.stt,
            "tts": model.tts,
            "llm": model.llm,
            "s2s": model.s2s,
            "audio_llm": model.audio_llm,
            "latency_levers": sorted(active_levers),
        },
        bound_metrics=bound,
        unsupported_metrics=sorted(unsupported),
        unknown_metrics=sorted(unknown),
        lever_warnings=lever_warnings,
        partial_composites=partial_composites,
        failures=failures,
    )
