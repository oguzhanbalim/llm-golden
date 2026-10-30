"""How far can the judge be trusted? Measure it against human labels.

Raw agreement flatters a judge on a skewed set: if 90% of outputs are good, a judge
that passes everything agrees 90% of the time and has learned nothing. Cohen's kappa
corrects for that, so both are reported.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LabelPair:
    id: str
    human: bool
    judge: bool


def cohen_kappa(tp: int, fp: int, fn: int, tn: int) -> float | None:
    """Kappa for a 2x2 table, or None where it is undefined.

    When both raters give one identical label throughout, chance agreement is 1 and
    the ratio is 0/0. Returning 0 or 1 there would be inventing a result.
    """
    n = tp + fp + fn + tn
    if n == 0:
        return None
    observed = (tp + tn) / n
    judge_pass, human_pass = (tp + fp) / n, (tp + fn) / n
    chance = judge_pass * human_pass + (1 - judge_pass) * (1 - human_pass)
    if chance >= 1.0:
        return None
    return (observed - chance) / (1 - chance)


def _table(pairs: Sequence[LabelPair]) -> tuple[int, int, int, int]:
    tp = sum(1 for p in pairs if p.human and p.judge)
    fp = sum(1 for p in pairs if not p.human and p.judge)
    fn = sum(1 for p in pairs if p.human and not p.judge)
    tn = sum(1 for p in pairs if not p.human and not p.judge)
    return tp, fp, fn, tn


def _bootstrap_kappa(
    pairs: Sequence[LabelPair], rounds: int, seed: int
) -> tuple[float, float] | None:
    rng = random.Random(seed)
    values = []
    for _ in range(rounds):
        kappa = cohen_kappa(*_table(rng.choices(pairs, k=len(pairs))))
        if kappa is not None:
            values.append(kappa)
    if len(values) < rounds * 0.9:
        return None  # Too many resamples were degenerate for an interval to mean anything.
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


@dataclass(frozen=True)
class Calibration:
    n: int
    tp: int
    fp: int
    fn: int
    tn: int
    agreement: float
    kappa: float | None
    kappa_ci: tuple[float, float] | None
    warnings: tuple[str, ...]

    @property
    def judge_pass_rate(self) -> float:
        return (self.tp + self.fp) / self.n

    @property
    def human_pass_rate(self) -> float:
        return (self.tp + self.fn) / self.n

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "confusion": {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn},
            "agreement": round(self.agreement, 4),
            "kappa": None if self.kappa is None else round(self.kappa, 4),
            "kappa_ci_95": None if self.kappa_ci is None else [round(x, 4) for x in self.kappa_ci],
            "human_pass_rate": round(self.human_pass_rate, 4),
            "judge_pass_rate": round(self.judge_pass_rate, 4),
            "warnings": list(self.warnings),
        }


def calibrate(pairs: Sequence[LabelPair], *, rounds: int = 2000, seed: int = 0) -> Calibration:
    if not pairs:
        raise ValueError("no labelled pairs to calibrate on")
    tp, fp, fn, tn = _table(pairs)
    n = len(pairs)
    kappa = cohen_kappa(tp, fp, fn, tn)
    warnings = []
    if n < 30:
        warnings.append(f"only {n} labels: treat the kappa interval as the result, not the point")
    majority = max(tp + fn, fp + tn) / n
    if majority >= 0.9:
        warnings.append(
            f"human labels are {majority:.0%} one class: agreement is flattered, read kappa"
        )
    if tp + fp == n or fn + tn == n:
        warnings.append("judge gave the same verdict every time")
    return Calibration(
        n=n,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        agreement=(tp + tn) / n,
        kappa=kappa,
        kappa_ci=_bootstrap_kappa(pairs, rounds, seed),
        warnings=tuple(warnings),
    )
