from __future__ import annotations

import pytest

from llm_golden.calibrate import LabelPair, calibrate, cohen_kappa


def pairs(tp=0, fp=0, fn=0, tn=0):
    rows = (
        [(True, True)] * tp + [(False, True)] * fp + [(True, False)] * fn + [(False, False)] * tn
    )
    return [LabelPair(str(i), h, j) for i, (h, j) in enumerate(rows)]


def test_kappa_textbook_value():
    # po = 0.7, pe = 0.5, kappa = 0.4
    assert cohen_kappa(20, 5, 10, 15) == pytest.approx(0.4)


def test_kappa_perfect_and_chance_and_worse_than_chance():
    assert cohen_kappa(10, 0, 0, 10) == pytest.approx(1.0)
    assert cohen_kappa(5, 5, 5, 5) == pytest.approx(0.0)
    assert cohen_kappa(0, 10, 10, 0) == pytest.approx(-1.0)


def test_kappa_undefined_when_both_raters_never_vary():
    assert cohen_kappa(10, 0, 0, 0) is None
    assert cohen_kappa(0, 0, 0, 0) is None


def test_lazy_judge_on_skewed_set_has_high_agreement_and_zero_kappa():
    result = calibrate(pairs(tp=45, fp=5))
    assert result.agreement == pytest.approx(0.9)
    assert result.kappa == pytest.approx(0.0)
    assert any("same verdict" in w for w in result.warnings)
    assert any("flattered" in w for w in result.warnings)


def test_ci_is_reproducible_and_contains_the_point():
    labelled = pairs(tp=20, fp=5, fn=10, tn=15)
    one, two = calibrate(labelled), calibrate(labelled)
    assert one.kappa_ci == two.kappa_ci
    assert one.kappa_ci[0] < one.kappa < one.kappa_ci[1]
    assert calibrate(labelled, seed=1).kappa_ci != one.kappa_ci


def test_small_samples_are_flagged():
    assert any("only 6 labels" in w for w in calibrate(pairs(tp=3, tn=3)).warnings)


def test_no_pairs_is_an_error():
    with pytest.raises(ValueError):
        calibrate([])
