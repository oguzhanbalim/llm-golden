from __future__ import annotations

from pathlib import Path

import pytest

from llm_golden.cases import Suite, build_suite

EXAMPLE = Path(__file__).parent.parent / "examples" / "ticket_triage"


@pytest.fixture
def example_dir() -> Path:
    return EXAMPLE


def exact_case(case_id: str, expected: str = "ok", **extra) -> dict:
    return {
        "id": case_id,
        "input": f"input for {case_id}",
        "graders": [{"type": "exact", "expected": expected}],
        **extra,
    }


@pytest.fixture
def small_suite() -> Suite:
    return build_suite(
        "small",
        [
            exact_case("a", split="validation"),
            exact_case("b", split="validation"),
            exact_case("c", expected="other", split="validation"),
            exact_case("t", split="test"),
            exact_case("u", split="train"),
        ],
    )
