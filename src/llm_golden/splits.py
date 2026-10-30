"""Stable train/validation/test assignment, and the lock on the test split.

Assignment hashes the case id (or its group), so adding a case never moves an
existing one. The lock cannot stop someone determined to read the YAML. What it
does is make every look at the test split explicit, reasoned and counted, because
a test set stops being one once results from it feed back into tuning.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from llm_golden.cases import SPLITS, Case, Suite, SuiteError


class TestSetLocked(Exception):
    """The test split was requested without an unlock, or its look budget is spent."""

    __test__ = False  # Not a pytest class, whatever the name suggests.


def assign_split(case: Case, *, salt: str, ratios: dict[str, float]) -> str:
    if case.split:
        return case.split
    key = case.group or case.id
    digest = hashlib.sha256(f"{salt}\0{key}".encode()).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    edge = 0.0
    for name in SPLITS:
        edge += ratios[name]
        if position < edge:
            return name
    return SPLITS[-1]  # Rounding left position above the last edge.


def split_map(suite: Suite) -> dict[str, str]:
    """Case id to split. Fails if a group is pinned so that it would straddle splits."""
    assigned = {c.id: assign_split(c, salt=suite.salt, ratios=suite.ratios) for c in suite.cases}
    by_group: dict[str, set[str]] = {}
    for case in suite.cases:
        if case.group:
            by_group.setdefault(case.group, set()).add(assigned[case.id])
    for group, splits in by_group.items():
        if len(splits) > 1:
            raise SuiteError(f"group {group!r} straddles splits {sorted(splits)}")
    return assigned


def find_leaks(suite: Suite) -> list[str]:
    """Identical inputs in different splits: a test case the system has already seen."""
    assigned = split_map(suite)
    first_seen: dict[str, Case] = {}
    leaks = []
    for case in suite.cases:
        key = " ".join(unicodedata.normalize("NFKC", str(case.input)).casefold().split())
        other = first_seen.setdefault(key, case)
        if other is not case and assigned[other.id] != assigned[case.id]:
            leaks.append(
                f"{other.id} ({assigned[other.id]}) and {case.id} ({assigned[case.id]}) "
                "have the same input"
            )
    return leaks


def cases_in(suite: Suite, split: str) -> list[Case]:
    assigned = split_map(suite)
    return [c for c in suite.cases if assigned[c.id] == split]


@dataclass(frozen=True)
class TestUnlock:
    """Proof that this look at the test split was recorded. Get one from a TestLedger."""

    __test__ = False

    look_number: int
    reason: str


class TestLedger:
    """Append-only record of looks at a test split. Commit it with the suite.

    Looks are counted per suite name, not per fingerprint, so editing a case does
    not wipe the history of what has already been seen.
    """

    __test__ = False

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def beside(cls, suite: Suite) -> TestLedger:
        if suite.path is None:
            raise ValueError("suite has no path; give TestLedger an explicit one")
        return cls(suite.path.with_name(suite.path.stem + ".test-ledger.jsonl"))

    def entries(self, suite_name: str) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        rows = (json.loads(line) for line in self.path.read_text("utf-8").splitlines() if line)
        return [row for row in rows if row.get("suite") == suite_name]

    def unlock(
        self, suite: Suite, *, system: str, reason: str, max_looks: int = 1
    ) -> TestUnlock:
        if not reason.strip():
            raise TestSetLocked("a look at the test split needs a reason")
        previous = self.entries(suite.name)
        if len(previous) >= max_looks:
            history = "; ".join(f"{e['at']} {e['system']}: {e['reason']}" for e in previous)
            raise TestSetLocked(
                f"test split of {suite.name!r} already looked at {len(previous)} time(s), "
                f"limit is {max_looks}. Raising the limit means reporting it as a "
                f"validation-grade number. History: {history}"
            )
        entry = {
            "suite": suite.name,
            "suite_fingerprint": suite.fingerprint,
            "system": system,
            "reason": reason,
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return TestUnlock(look_number=len(previous) + 1, reason=reason)
