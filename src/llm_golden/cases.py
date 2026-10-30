"""Load a suite of golden cases from YAML or JSONL."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from llm_golden.graders import Grader, build_grader

SPLITS = ("train", "validation", "test")
DEFAULT_RATIOS = {"train": 0.6, "validation": 0.2, "test": 0.2}


class SuiteError(ValueError):
    """The suite file is malformed."""


def _digest(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Case:
    id: str
    input: Any
    grader_specs: tuple[dict[str, Any], ...]
    graders: tuple[Grader, ...] = field(compare=False, repr=False)
    tags: tuple[str, ...] = ()
    group: str | None = None
    split: str | None = None

    @property
    def fingerprint(self) -> str:
        """Identifies what the case asks. A changed expectation is a different case."""
        return _digest({"input": self.input, "graders": list(self.grader_specs)})


@dataclass(frozen=True)
class Suite:
    name: str
    cases: tuple[Case, ...]
    salt: str = "v1"
    ratios: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RATIOS))
    path: Path | None = None

    @property
    def fingerprint(self) -> str:
        return _digest(sorted((c.id, c.fingerprint) for c in self.cases))


def _parse_case(raw: Any, where: str) -> Case:
    if not isinstance(raw, dict):
        raise SuiteError(f"{where}: a case must be a mapping")
    unknown = set(raw) - {"id", "input", "graders", "tags", "group", "split"}
    if unknown:
        raise SuiteError(f"{where}: unknown keys {sorted(unknown)}")
    for key in ("id", "input", "graders"):
        if key not in raw:
            raise SuiteError(f"{where}: missing '{key}'")
    specs = raw["graders"]
    if not isinstance(specs, list) or not specs:
        raise SuiteError(f"{where}: 'graders' must be a non-empty list")
    split = raw.get("split")
    if split is not None and split not in SPLITS:
        raise SuiteError(f"{where}: split must be one of {SPLITS}, got {split!r}")
    try:
        graders = tuple(build_grader(spec) for spec in specs)
    except ValueError as exc:
        raise SuiteError(f"{where} (id={raw['id']}): {exc}") from exc
    return Case(
        id=str(raw["id"]),
        input=raw["input"],
        grader_specs=tuple(specs),
        graders=graders,
        tags=tuple(raw.get("tags", ())),
        group=raw.get("group"),
        split=split,
    )


def _check_ratios(ratios: dict[str, float]) -> None:
    if set(ratios) != set(SPLITS) or any(v < 0 for v in ratios.values()):
        raise SuiteError(f"ratios need non-negative values for exactly {SPLITS}")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise SuiteError(f"ratios must sum to 1, got {sum(ratios.values())}")


def build_suite(
    name: str,
    raw_cases: list[Any],
    *,
    salt: str = "v1",
    ratios: dict[str, float] | None = None,
    path: Path | None = None,
) -> Suite:
    ratios = dict(ratios or DEFAULT_RATIOS)
    _check_ratios(ratios)
    cases = tuple(_parse_case(raw, f"case #{i + 1}") for i, raw in enumerate(raw_cases))
    if not cases:
        raise SuiteError("suite has no cases")
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise SuiteError(f"duplicate case id {case.id!r}")
        seen.add(case.id)
    return Suite(name=name, cases=cases, salt=salt, ratios=ratios, path=path)


def load_suite(path: str | Path) -> Suite:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = []
        for number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SuiteError(f"{path}:{number}: {exc.msg}") from exc
        return build_suite(path.stem, rows, path=path)
    if path.suffix not in {".yaml", ".yml"}:
        raise SuiteError(f"{path}: expected .yaml, .yml or .jsonl")
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict) or "cases" not in doc:
        raise SuiteError(f"{path}: expected a mapping with a 'cases' list")
    return build_suite(
        str(doc.get("suite", path.stem)),
        doc["cases"],
        salt=str(doc.get("split_salt", "v1")),
        ratios=doc.get("split_ratios"),
        path=path,
    )
