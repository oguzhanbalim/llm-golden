"""Run a system over one split and write a result file that can be compared later."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_golden.cases import Case, Suite
from llm_golden.graders import ERROR, FAIL, PASS, SKIPPED, Grade
from llm_golden.judge import LLMJudge
from llm_golden.splits import TestSetLocked, TestUnlock, cases_in

RESULT_FORMAT = "llm-golden/results/v1"
System = Callable[[Any], str]


def wilson_interval(passed: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% interval for a pass rate. Twenty cases cannot pin a rate to one decimal."""
    if total == 0:
        return None
    p = passed / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _case_status(grades: list[Grade]) -> str:
    statuses = {g.status for g in grades}
    # A definite failure stands even if another grader could not run.
    for status in (FAIL, ERROR, SKIPPED):
        if status in statuses:
            return status
    return PASS


@dataclass(frozen=True)
class CaseResult:
    id: str
    fingerprint: str
    status: str
    tags: tuple[str, ...]
    output: str | None
    grades: tuple[Grade, ...]
    error: str | None = None

    def to_dict(self, *, redact: bool) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": self.id,
            "fingerprint": self.fingerprint,
            "status": self.status,
            "tags": list(self.tags),
        }
        if not redact:
            row["output"] = self.output
            row["grades"] = [g.to_dict() for g in self.grades]
            row["error"] = self.error
        return row


@dataclass(frozen=True)
class RunResult:
    suite: str
    suite_fingerprint: str
    split: str
    system: str
    cases: tuple[CaseResult, ...]

    def count(self, status: str) -> int:
        return sum(1 for c in self.cases if c.status == status)

    @property
    def summary(self) -> dict[str, Any]:
        passed = self.count(PASS)
        graded = len(self.cases) - self.count(SKIPPED)
        interval = wilson_interval(passed, graded)
        return {
            "total": len(self.cases),
            "passed": passed,
            "failed": self.count(FAIL),
            "errors": self.count(ERROR),
            "skipped": self.count(SKIPPED),
            "pass_rate": round(passed / graded, 4) if graded else None,
            "wilson_95": [round(x, 4) for x in interval] if interval else None,
        }

    def to_dict(self) -> dict[str, Any]:
        # Test results keep per-case status for regression checks and drop the
        # outputs and grader details, which are what you would tune against.
        redact = self.split == "test"
        return {
            "format": RESULT_FORMAT,
            "suite": self.suite,
            "suite_fingerprint": self.suite_fingerprint,
            "split": self.split,
            "system": self.system,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "redacted": redact,
            "summary": self.summary,
            "cases": [c.to_dict(redact=redact) for c in self.cases],
        }

    def write(self, path: str | Path) -> None:
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        Path(path).write_text(text + "\n", encoding="utf-8")


def _run_case(system: System, case: Case, judge: LLMJudge | None) -> CaseResult:
    try:
        output = system(case.input)
        if not isinstance(output, str):
            raise TypeError(f"system returned {type(output).__name__}, expected str")
    except Exception as exc:  # The system under test may fail in any way; record it.
        return CaseResult(case.id, case.fingerprint, ERROR, case.tags, None, (), repr(exc))
    grades = [g.grade(output, case_input=case.input, judge=judge) for g in case.graders]
    return CaseResult(
        case.id, case.fingerprint, _case_status(grades), case.tags, output, tuple(grades)
    )


def run_suite(
    system: System,
    suite: Suite,
    *,
    split: str,
    system_name: str = "system",
    judge: LLMJudge | None = None,
    test_unlock: TestUnlock | None = None,
) -> RunResult:
    if split == "test" and test_unlock is None:
        raise TestSetLocked("the test split needs a TestUnlock from TestLedger.unlock()")
    cases = cases_in(suite, split)
    if not cases:
        raise ValueError(f"split {split!r} of suite {suite.name!r} has no cases")
    return RunResult(
        suite=suite.name,
        suite_fingerprint=suite.fingerprint,
        split=split,
        system=system_name,
        cases=tuple(_run_case(system, case, judge) for case in cases),
    )
