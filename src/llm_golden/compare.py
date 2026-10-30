"""Compare two result files case by case.

A case whose definition changed between runs is reported apart. Otherwise relaxing
an expected value would show up as a fix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any

from llm_golden.graders import PASS, SKIPPED
from llm_golden.runner import RESULT_FORMAT


class CompareError(ValueError):
    """The two result files cannot be compared."""


def load_result(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text("utf-8"))
    if not isinstance(data, dict) or data.get("format") != RESULT_FORMAT:
        raise CompareError(f"{path} is not a {RESULT_FORMAT} file")
    return data


def sign_test_p(regressions: int, fixes: int) -> float:
    """Exact two-sided sign test on the cases that flipped."""
    flips = regressions + fixes
    if flips == 0:
        return 1.0
    tail = sum(comb(flips, k) for k in range(min(regressions, fixes) + 1)) / 2**flips
    return min(1.0, 2 * tail)


@dataclass(frozen=True)
class Comparison:
    split: str
    regressions: tuple[str, ...]
    fixes: tuple[str, ...]
    lost_coverage: tuple[str, ...]
    redefined: tuple[str, ...]
    added: tuple[str, ...]
    removed: tuple[str, ...]
    common: int
    base_passed: int
    head_passed: int

    @property
    def p_value(self) -> float:
        return sign_test_p(len(self.regressions), len(self.fixes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "common_cases": self.common,
            "base_passed": self.base_passed,
            "head_passed": self.head_passed,
            "regressions": list(self.regressions),
            "fixes": list(self.fixes),
            "lost_coverage": list(self.lost_coverage),
            "redefined": list(self.redefined),
            "added": list(self.added),
            "removed": list(self.removed),
            "sign_test_p": round(self.p_value, 4),
        }


def compare(base: dict[str, Any], head: dict[str, Any]) -> Comparison:
    if base["split"] != head["split"]:
        raise CompareError(f"splits differ: {base['split']} vs {head['split']}")
    if base["suite"] != head["suite"]:
        raise CompareError(f"suites differ: {base['suite']} vs {head['suite']}")
    old = {c["id"]: c for c in base["cases"]}
    new = {c["id"]: c for c in head["cases"]}
    regressions, fixes, lost, redefined = [], [], [], []
    common = base_passed = head_passed = 0
    for case_id in sorted(old.keys() & new.keys()):
        before, after = old[case_id], new[case_id]
        if before["fingerprint"] != after["fingerprint"]:
            redefined.append(case_id)
            continue
        common += 1
        was_pass, is_pass = before["status"] == PASS, after["status"] == PASS
        base_passed += was_pass
        head_passed += is_pass
        if was_pass and after["status"] == SKIPPED:
            lost.append(case_id)
        elif was_pass and not is_pass:
            regressions.append(case_id)
        elif is_pass and not was_pass:
            fixes.append(case_id)
    return Comparison(
        split=base["split"],
        regressions=tuple(regressions),
        fixes=tuple(fixes),
        lost_coverage=tuple(lost),
        redefined=tuple(redefined),
        added=tuple(sorted(new.keys() - old.keys())),
        removed=tuple(sorted(old.keys() - new.keys())),
        common=common,
        base_passed=base_passed,
        head_passed=head_passed,
    )
