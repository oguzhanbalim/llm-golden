from __future__ import annotations

import json

import pytest
from conftest import exact_case

from llm_golden.cases import build_suite
from llm_golden.compare import CompareError, compare, load_result, sign_test_p
from llm_golden.graders import ERROR, FAIL, PASS, SKIPPED
from llm_golden.runner import run_suite, wilson_interval
from llm_golden.splits import TestLedger


def test_statuses_and_summary(small_suite):
    result = run_suite(lambda _: "ok", small_suite, split="validation")
    assert result.summary["passed"] == 2
    assert result.summary["failed"] == 1
    assert result.summary["pass_rate"] == pytest.approx(0.6667)


def test_system_exception_is_an_error_with_the_cause(small_suite):
    def boom(_):
        raise RuntimeError("no capacity")

    result = run_suite(boom, small_suite, split="validation")
    assert {c.status for c in result.cases} == {ERROR}
    assert "no capacity" in result.cases[0].error


def test_non_string_output_is_an_error(small_suite):
    result = run_suite(lambda _: 7, small_suite, split="validation")
    assert {c.status for c in result.cases} == {ERROR}


def test_skipped_is_excluded_from_the_rate_and_blocks_a_clean_pass():
    judged = {
        "id": "j",
        "input": "x",
        "split": "validation",
        "graders": [{"type": "judge", "criteria": "c"}],
    }
    suite = build_suite("s", [judged, exact_case("a", split="validation")])
    summary = run_suite(lambda _: "ok", suite, split="validation").summary
    assert summary["skipped"] == 1
    assert summary["pass_rate"] == 1.0
    assert summary["passed"] < summary["total"]


def test_deterministic_fail_outranks_skipped_judge():
    case = {
        "id": "j",
        "input": "x",
        "split": "validation",
        "graders": [{"type": "exact", "expected": "no"}, {"type": "judge", "criteria": "c"}],
    }
    suite = build_suite("s", [case])
    assert run_suite(lambda _: "yes", suite, split="validation").cases[0].status == FAIL


def test_empty_split_is_an_error():
    suite = build_suite("s", [exact_case("a", split="train")])
    with pytest.raises(ValueError, match="no cases"):
        run_suite(lambda _: "ok", suite, split="validation")


def test_test_results_are_redacted_on_disk(small_suite, tmp_path):
    unlock = TestLedger(tmp_path / "l.jsonl").unlock(small_suite, system="v", reason="r")
    result = run_suite(lambda _: "secret output", small_suite, split="test", test_unlock=unlock)
    out = tmp_path / "r.json"
    result.write(out)
    text = out.read_text()
    assert "secret output" not in text
    assert json.loads(text)["cases"][0]["status"] == FAIL


def test_wilson_known_values():
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-3)
    assert high == pytest.approx(0.9433, abs=1e-3)
    assert wilson_interval(0, 0) is None
    assert wilson_interval(0, 5)[0] == 0.0


def _result(statuses: dict[str, str], *, split="validation", fingerprints=None) -> dict:
    fingerprints = fingerprints or {}
    return {
        "format": "llm-golden/results/v1",
        "suite": "s",
        "split": split,
        "cases": [
            {"id": k, "status": v, "fingerprint": fingerprints.get(k, "f")}
            for k, v in statuses.items()
        ],
    }


def test_compare_finds_flips_adds_and_removes():
    base = _result({"a": PASS, "b": FAIL, "c": PASS, "gone": PASS})
    head = _result({"a": FAIL, "b": PASS, "c": PASS, "new": PASS})
    result = compare(base, head)
    assert result.regressions == ("a",)
    assert result.fixes == ("b",)
    assert result.removed == ("gone",)
    assert result.added == ("new",)
    assert (result.common, result.base_passed, result.head_passed) == (3, 2, 2)


def test_error_counts_as_a_regression():
    assert compare(_result({"a": PASS}), _result({"a": ERROR})).regressions == ("a",)


def test_relaxed_expectation_is_not_a_fix():
    base = _result({"a": FAIL, "b": PASS}, fingerprints={"a": "strict"})
    head = _result({"a": PASS, "b": PASS}, fingerprints={"a": "relaxed"})
    result = compare(base, head)
    assert result.fixes == ()
    assert result.redefined == ("a",)
    assert result.common == 1


def test_pass_to_skipped_is_lost_coverage():
    result = compare(_result({"a": PASS}), _result({"a": SKIPPED}))
    assert result.regressions == ()
    assert result.lost_coverage == ("a",)


def test_refuses_mismatched_splits():
    with pytest.raises(CompareError, match="splits differ"):
        compare(_result({"a": PASS}), _result({"a": PASS}, split="train"))


def test_load_result_checks_format(tmp_path):
    bad = tmp_path / "x.json"
    bad.write_text("{}")
    with pytest.raises(CompareError):
        load_result(bad)


@pytest.mark.parametrize(
    "regressions,fixes,expected",
    [(0, 0, 1.0), (1, 1, 1.0), (0, 5, 0.0625), (1, 9, 0.02148), (5, 0, 0.0625)],
)
def test_sign_test_exact_values(regressions, fixes, expected):
    assert sign_test_p(regressions, fixes) == pytest.approx(expected, abs=1e-4)
