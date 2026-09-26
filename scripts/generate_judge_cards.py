#!/usr/bin/env python3
"""Generate judge cards for every LLM judge metric and report evidence gaps.

Run this to publish docs/metrics/judge_cards.md and to see which judges are
missing the LLJ Cards reporting items (validation dataset, development record,
pinned decoding parameters, ...).

Usage:
    python scripts/generate_judge_cards.py                # print to stdout
    python scripts/generate_judge_cards.py --out docs/metrics/judge_cards.md
"""

import argparse
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

from typing_extensions import TypedDict

REPO_ROOT = Path(__file__).resolve().parent.parent


def _stub_heavy_imports() -> None:
    """Install a meta-path finder that stubs litellm before any eva imports.

    Same trick as scripts/regen_metric_signatures.py: the judge-card walk only
    reads class definitions and prompt templates, never litellm itself, but
    importing the metric subpackages pulls it in at ~4 s of import time.
    """
    _STUB_PACKAGES = frozenset({"litellm"})

    class _AutoStub(ModuleType):
        """A module stub that satisfies arbitrary attribute and submodule access."""

        def __getattr__(self, name: str) -> "_AutoStub":
            child = _AutoStub(f"{self.__name__}.{name}")
            object.__setattr__(self, name, child)
            sys.modules[child.__name__] = child
            return child

        def __call__(self, *args: object, **kwargs: object) -> "_AutoStub":
            return self

        def __iter__(self):  # type: ignore[override]
            return iter([])

        # Allow use in type union expressions at module level, e.g. `Router | None`
        def __or__(self, other: object) -> object:
            return object

        def __ror__(self, other: object) -> object:
            return object

    class _StubLoader(Loader):
        def create_module(self, spec: ModuleSpec) -> _AutoStub:
            return _AutoStub(spec.name)

        def exec_module(self, module: ModuleType) -> None:
            pass  # _AutoStub handles everything via __getattr__

    class _StubFinder(MetaPathFinder):
        def find_spec(self, fullname: str, path: object, target: object = None) -> ModuleSpec | None:
            if fullname.split(".")[0] in _STUB_PACKAGES:
                return ModuleSpec(fullname, _StubLoader())
            return None

    sys.meta_path.insert(0, _StubFinder())

    # DeploymentTypedDict is inherited by ModelDeployment in eva.models.config,
    # so it must be a real class. Trigger the import so the stub is registered in
    # sys.modules, then replace the attribute with a proper TypedDict.
    import litellm.types.router  # noqa: PLC0415, F401

    class DeploymentTypedDict(TypedDict, total=False):
        pass

    sys.modules["litellm.types.router"].DeploymentTypedDict = DeploymentTypedDict  # type: ignore[attr-defined]


_stub_heavy_imports()

from eva.metrics.judge_cards import audit_judge_cards, build_judge_cards, render_markdown  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="write the cards here instead of stdout")
    args = parser.parse_args()

    cards = build_judge_cards(REPO_ROOT)
    findings = audit_judge_cards(cards)

    document = render_markdown(cards, findings)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(document, encoding="utf-8")
        target = args.out.resolve().relative_to(REPO_ROOT) if args.out.is_relative_to(REPO_ROOT) else args.out
        print(f"Wrote judge cards for {len(cards)} metrics to {target}")
    else:
        print(document, end="")

    total = sum(len(issues) for issues in findings.values())
    print(f"\n{total} reporting gaps across {len(cards)} judges:", file=sys.stderr)
    for name, issues in sorted(findings.items()):
        for finding in issues:
            print(f"  {name}: [{finding.card}] {finding.code} - {finding.message}", file=sys.stderr)


if __name__ == "__main__":
    main()
