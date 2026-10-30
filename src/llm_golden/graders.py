"""Graders turn one system output into a pass, fail, error or skip.

Each grader is built from a plain dict so a bad regex or schema fails when the
suite loads, not forty cases into a run.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Protocol

import jsonschema

from llm_golden.judge import JudgeError, LLMJudge

PASS = "pass"
FAIL = "fail"
ERROR = "error"
SKIPPED = "skipped"

# A number with optional sign, thousands separators and exponent. Dot decimal only.
# The lookbehinds keep identifiers such as TKT-4417 from reading as numbers.
_NUMBER = re.compile(
    r"(?<![\w.])(?<!\w-)[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?!\w)"
)
_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


class GraderConfigError(ValueError):
    """A grader spec in the suite file is invalid."""


@dataclass(frozen=True)
class Grade:
    grader: str
    status: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"grader": self.grader, "status": self.status, "detail": self.detail}


class Grader(Protocol):
    name: str

    def grade(self, output: str, *, case_input: Any, judge: LLMJudge | None) -> Grade: ...


def _result(name: str, ok: bool, detail: str) -> Grade:
    return Grade(name, PASS if ok else FAIL, "" if ok else detail)


def _clip(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass(frozen=True)
class ExactGrader:
    expected: str
    ignore_case: bool = False
    normalize_whitespace: bool = False
    name: str = "exact"

    def _canon(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text).strip()
        if self.normalize_whitespace:
            text = " ".join(text.split())
        return text.casefold() if self.ignore_case else text

    def grade(self, output: str, *, case_input: Any, judge: LLMJudge | None) -> Grade:
        ok = self._canon(output) == self._canon(self.expected)
        return _result(self.name, ok, f"expected {self.expected!r}, got {_clip(output)!r}")


@dataclass(frozen=True)
class RegexGrader:
    pattern: re.Pattern[str]
    fullmatch: bool = False
    must_not_match: bool = False
    name: str = "regex"

    def grade(self, output: str, *, case_input: Any, judge: LLMJudge | None) -> Grade:
        text = output.strip()
        match = self.pattern.fullmatch(text) if self.fullmatch else self.pattern.search(text)
        if self.must_not_match:
            found = match.group(0) if match else ""
            return _result(self.name, match is None, f"forbidden pattern matched {found!r}")
        return _result(self.name, match is not None, f"no match for {self.pattern.pattern!r}")


def extract_numbers(text: str) -> list[float]:
    return [float(m.group(0).replace(",", "")) for m in _NUMBER.finditer(text)]


@dataclass(frozen=True)
class NumericGrader:
    expected: float
    abs_tol: float = 1e-9
    rel_tol: float = 0.0
    pick: str = "only"
    name: str = "numeric"

    def grade(self, output: str, *, case_input: Any, judge: LLMJudge | None) -> Grade:
        numbers = extract_numbers(output)
        if not numbers:
            return Grade(self.name, FAIL, f"no number in {_clip(output)!r}")
        if self.pick == "only" and len(numbers) > 1:
            # An answer listing several numbers must not pass because one is right.
            return Grade(self.name, FAIL, f"ambiguous: found {numbers}; set pick: first or last")
        value = numbers[-1] if self.pick == "last" else numbers[0]
        ok = math.isclose(value, self.expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        return _result(self.name, ok, f"expected {self.expected} got {value}")


@dataclass(frozen=True)
class JsonSchemaGrader:
    schema: dict[str, Any]
    allow_fence: bool = False
    name: str = "json_schema"

    def grade(self, output: str, *, case_input: Any, judge: LLMJudge | None) -> Grade:
        text = output.strip()
        if self.allow_fence and (fenced := _FENCE.search(text)):
            text = fenced.group(1)
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            return Grade(self.name, FAIL, f"not JSON: {exc.msg} at char {exc.pos}")
        validator = jsonschema.Draft202012Validator(self.schema)
        errors = sorted(validator.iter_errors(value), key=lambda e: list(map(str, e.path)))
        if not errors:
            return Grade(self.name, PASS)
        first = errors[0]
        where = "/".join(map(str, first.path)) or "<root>"
        return Grade(self.name, FAIL, f"{where}: {first.message}")


@dataclass(frozen=True)
class JudgeGrader:
    criteria: str
    name: str = "judge"

    def grade(self, output: str, *, case_input: Any, judge: LLMJudge | None) -> Grade:
        if judge is None:
            return Grade(self.name, SKIPPED, "no judge configured")
        try:
            verdict = judge.evaluate(criteria=self.criteria, case_input=case_input, output=output)
        except JudgeError as exc:
            # An unreadable verdict is not evidence that the output was bad.
            return Grade(self.name, ERROR, str(exc))
        return Grade(self.name, PASS if verdict.passed else FAIL, verdict.reason)


def _require(spec: dict[str, Any], key: str) -> Any:
    if key not in spec:
        raise GraderConfigError(f"{spec.get('type')} grader needs '{key}'")
    return spec[key]


def _check_keys(spec: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(spec) - allowed - {"type"}
    if unknown:
        raise GraderConfigError(f"{spec['type']} grader: unknown keys {sorted(unknown)}")


def build_grader(spec: dict[str, Any]) -> Grader:
    kind = spec.get("type")
    if kind == "exact":
        _check_keys(spec, {"expected", "ignore_case", "normalize_whitespace"})
        return ExactGrader(
            str(_require(spec, "expected")),
            ignore_case=bool(spec.get("ignore_case", False)),
            normalize_whitespace=bool(spec.get("normalize_whitespace", False)),
        )
    if kind == "regex":
        _check_keys(spec, {"pattern", "ignore_case", "fullmatch", "must_not_match"})
        flags = re.IGNORECASE if spec.get("ignore_case") else 0
        try:
            pattern = re.compile(str(_require(spec, "pattern")), flags)
        except re.error as exc:
            raise GraderConfigError(f"bad regex {spec['pattern']!r}: {exc}") from exc
        return RegexGrader(
            pattern,
            fullmatch=bool(spec.get("fullmatch", False)),
            must_not_match=bool(spec.get("must_not_match", False)),
        )
    if kind == "numeric":
        _check_keys(spec, {"expected", "abs_tol", "rel_tol", "pick"})
        expected = float(_require(spec, "expected"))
        pick = spec.get("pick", "only")
        if pick not in {"only", "first", "last"}:
            raise GraderConfigError(f"numeric pick must be only, first or last, got {pick!r}")
        abs_tol, rel_tol = float(spec.get("abs_tol", 1e-9)), float(spec.get("rel_tol", 0.0))
        if not math.isfinite(expected) or abs_tol < 0 or rel_tol < 0:
            raise GraderConfigError("numeric grader needs a finite expected and tolerances >= 0")
        return NumericGrader(expected, abs_tol=abs_tol, rel_tol=rel_tol, pick=pick)
    if kind == "json_schema":
        _check_keys(spec, {"schema", "allow_fence"})
        schema = _require(spec, "schema")
        try:
            jsonschema.Draft202012Validator.check_schema(schema)
        except jsonschema.SchemaError as exc:
            raise GraderConfigError(f"bad JSON schema: {exc.message}") from exc
        return JsonSchemaGrader(schema, allow_fence=bool(spec.get("allow_fence", False)))
    if kind == "judge":
        _check_keys(spec, {"criteria"})
        return JudgeGrader(str(_require(spec, "criteria")))
    raise GraderConfigError(f"unknown grader type {kind!r}")
