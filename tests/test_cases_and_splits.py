from __future__ import annotations

import json

import pytest
from conftest import exact_case

from llm_golden.cases import SuiteError, build_suite, load_suite
from llm_golden.runner import run_suite
from llm_golden.splits import TestLedger, TestSetLocked, find_leaks, split_map


def test_load_yaml_and_jsonl_agree(tmp_path):
    case = exact_case("one", tags=["x"])
    jsonl = tmp_path / "s.jsonl"
    jsonl.write_text(json.dumps(case) + "\n\n")
    yml = tmp_path / "s.yaml"
    yml.write_text("suite: s\ncases:\n" + "  - " + json.dumps(case) + "\n")
    a, b = load_suite(jsonl), load_suite(yml)
    assert a.cases[0].fingerprint == b.cases[0].fingerprint
    assert a.cases[0].tags == ("x",)


@pytest.mark.parametrize(
    "bad,message",
    [
        ([exact_case("a"), exact_case("a")], "duplicate"),
        ([{"id": "a", "input": "x"}], "missing 'graders'"),
        ([{"id": "a", "input": "x", "graders": []}], "non-empty"),
        ([exact_case("a", split="holdout")], "split must be"),
        ([exact_case("a", typo=1)], "unknown keys"),
        ([], "no cases"),
    ],
)
def test_malformed_suites_are_rejected(bad, message):
    with pytest.raises(SuiteError, match=message):
        build_suite("s", bad)


def test_ratios_must_sum_to_one():
    with pytest.raises(SuiteError, match="sum to 1"):
        build_suite(
            "s", [exact_case("a")], ratios={"train": 0.5, "validation": 0.5, "test": 0.5}
        )


def test_fingerprint_changes_with_expectation_only():
    one = build_suite("s", [exact_case("a", expected="x", tags=["t"])]).cases[0]
    same = build_suite("s", [exact_case("a", expected="x", tags=["other"])]).cases[0]
    other = build_suite("s", [exact_case("a", expected="y")]).cases[0]
    assert one.fingerprint == same.fingerprint
    assert one.fingerprint != other.fingerprint


def _many(n, **extra):
    return [exact_case(f"case-{i}", **extra) for i in range(n)]


def test_split_ratio_is_roughly_honoured_and_deterministic():
    suite = build_suite("s", _many(2000))
    assigned = split_map(suite)
    assert assigned == split_map(build_suite("s", _many(2000)))
    share = sum(v == "test" for v in assigned.values()) / 2000
    assert 0.17 < share < 0.23


def test_adding_cases_never_moves_an_existing_case():
    before = split_map(build_suite("s", _many(300)))
    after = split_map(build_suite("s", _many(600)))
    assert all(after[case_id] == split for case_id, split in before.items())


def test_salt_reshuffles():
    a = split_map(build_suite("s", _many(100), salt="a"))
    b = split_map(build_suite("s", _many(100), salt="b"))
    assert a != b


def test_group_stays_together():
    cases = [exact_case(f"c{i}", group=f"g{i % 7}") for i in range(70)]
    suite = build_suite("s", cases)
    assigned = split_map(suite)
    for g in range(7):
        assert len({assigned[f"c{i}"] for i in range(70) if i % 7 == g}) == 1


def test_group_pinned_across_splits_is_an_error():
    suite = build_suite(
        "s",
        [exact_case("a", group="g", split="train"), exact_case("b", group="g", split="test")],
    )
    with pytest.raises(SuiteError, match="straddles"):
        split_map(suite)


def test_leak_detector_ignores_case_and_whitespace():
    a = {**exact_case("a", split="train"), "input": "Hello   World"}
    b = {**exact_case("b", split="test"), "input": "hello world"}
    c = {**exact_case("c", split="train"), "input": "HELLO WORLD"}
    leaks = find_leaks(build_suite("s", [a, b, c]))
    assert len(leaks) == 1
    assert "a (train) and b (test)" in leaks[0]


def test_test_split_is_locked_without_unlock(small_suite):
    with pytest.raises(TestSetLocked):
        run_suite(lambda _: "ok", small_suite, split="test")


def test_ledger_allows_one_look_then_refuses(small_suite, tmp_path):
    ledger = TestLedger(tmp_path / "ledger.jsonl")
    unlock = ledger.unlock(small_suite, system="v1", reason="release candidate")
    assert unlock.look_number == 1
    assert run_suite(lambda _: "ok", small_suite, split="test", test_unlock=unlock).cases
    with pytest.raises(TestSetLocked, match="already looked at 1"):
        ledger.unlock(small_suite, system="v2", reason="one more try")
    assert ledger.unlock(small_suite, system="v2", reason="ok", max_looks=2).look_number == 2


def test_empty_reason_is_refused(small_suite, tmp_path):
    with pytest.raises(TestSetLocked, match="reason"):
        TestLedger(tmp_path / "l.jsonl").unlock(small_suite, system="v1", reason="  ")


def test_editing_a_case_does_not_launder_the_look_count(small_suite, tmp_path):
    ledger = TestLedger(tmp_path / "ledger.jsonl")
    ledger.unlock(small_suite, system="v1", reason="first")
    edited = build_suite("small", [exact_case("t", expected="changed", split="test")])
    assert edited.fingerprint != small_suite.fingerprint
    with pytest.raises(TestSetLocked):
        ledger.unlock(edited, system="v1", reason="second")


def test_ledgers_are_separate_per_suite(small_suite, tmp_path):
    ledger = TestLedger(tmp_path / "ledger.jsonl")
    ledger.unlock(small_suite, system="v1", reason="first")
    other = build_suite("other", [exact_case("t", split="test")])
    assert ledger.unlock(other, system="v1", reason="first").look_number == 1
