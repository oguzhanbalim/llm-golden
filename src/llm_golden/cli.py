"""Command line: split, run, compare, calibrate."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from llm_golden.calibrate import LabelPair, calibrate
from llm_golden.cases import SPLITS, Suite, SuiteError, load_suite
from llm_golden.compare import CompareError, compare, load_result
from llm_golden.graders import PASS
from llm_golden.judge import JudgeError, LLMJudge
from llm_golden.runner import RunResult, run_suite
from llm_golden.splits import TestLedger, TestSetLocked, find_leaks, split_map


def load_callable(spec: str) -> Callable[..., Any]:
    """Resolve ``package.module:name`` or ``path/to/file.py:name``."""
    target, sep, attr = spec.rpartition(":")
    if not sep:
        raise ValueError(f"expected module:callable or file.py:callable, got {spec!r}")
    if target.endswith(".py"):
        path = Path(target).resolve()
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        if module_spec is None or module_spec.loader is None:
            raise ValueError(f"cannot load {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    return getattr(module, attr)


def _judge_from(spec: str | None) -> LLMJudge | None:
    return LLMJudge(load_callable(spec)()) if spec else None


def _print_run(result: RunResult) -> None:
    s = result.summary
    print(f"suite={result.suite} split={result.split} system={result.system}")
    rate = "n/a" if s["pass_rate"] is None else f"{s['pass_rate']:.1%}"
    low, high = s["wilson_95"] or (0, 0)
    print(
        f"passed {s['passed']}/{s['total']}  failed {s['failed']}  errors {s['errors']}  "
        f"skipped {s['skipped']}  pass rate {rate} (95% CI {low:.1%} to {high:.1%})"
    )
    if result.split == "test":
        print("test split: per-case detail withheld.")
        return
    for case in result.cases:
        if case.status == PASS:
            continue
        detail = case.error or "; ".join(
            f"{g.grader}: {g.detail}" for g in case.grades if g.status != PASS
        )
        print(f"  {case.status.upper():7} {case.id}  {detail}")


def cmd_split(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    assigned = split_map(suite)
    counts = Counter(assigned.values())
    print(f"suite={suite.name} cases={len(suite.cases)} fingerprint={suite.fingerprint}")
    for split in SPLITS:
        print(f"  {split:10} {counts[split]}")
    leaks = find_leaks(suite)
    for leak in leaks:
        print(f"  LEAK {leak}")
    return 1 if leaks else 0


def _unlock(args: argparse.Namespace, suite: Suite):
    if args.split != "test":
        return None
    if not args.unlock_test:
        raise TestSetLocked("--split test needs --unlock-test and --reason")
    ledger = TestLedger(args.ledger) if args.ledger else TestLedger.beside(suite)
    return ledger.unlock(
        suite,
        system=args.name or args.system,
        reason=args.reason or "",
        max_looks=args.max_test_looks,
    )


def cmd_run(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    leaks = find_leaks(suite)
    if leaks and not args.allow_leaks:
        print("refusing to run, inputs leak across splits:", *leaks, sep="\n  ", file=sys.stderr)
        return 2
    system = load_callable(args.system)
    judge = _judge_from(args.judge)
    # Last before running: a typo in --system should not spend the look budget.
    unlock = _unlock(args, suite)
    result = run_suite(
        system,
        suite,
        split=args.split,
        system_name=args.name or args.system,
        judge=judge,
        test_unlock=unlock,
    )
    _print_run(result)
    if args.out:
        result.write(args.out)
    return 0 if result.summary["passed"] == result.summary["total"] else 1


def cmd_compare(args: argparse.Namespace) -> int:
    result = compare(load_result(args.base), load_result(args.head))
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(
            f"split={result.split}  unchanged cases: {result.common}  "
            f"passed {result.base_passed} -> {result.head_passed}"
        )
        for label, ids in (
            ("regressions", result.regressions),
            ("fixes", result.fixes),
            ("lost coverage (now skipped)", result.lost_coverage),
            ("redefined, not counted", result.redefined),
            ("added", result.added),
            ("removed", result.removed),
        ):
            if ids:
                print(f"  {label}: {', '.join(ids)}")
        print(f"  sign test on flipped cases: p={result.p_value:.3f}")
    bad = len(result.regressions) + len(result.lost_coverage)
    return 1 if bad > args.max_regressions else 0


def _labels_from(path: str, suite: Suite | None, judge: LLMJudge | None, allow_test: bool):
    assigned = split_map(suite) if suite else {}
    pairs, unchecked = [], 0
    for number, line in enumerate(Path(path).read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        split = assigned.get(row["id"], row.get("split"))
        if split is None:
            unchecked += 1
        if split == "test" and not allow_test:
            raise TestSetLocked(
                f"{path}:{number}: {row['id']} is in the test split; "
                "calibrating on it lets test cases shape the judge"
            )
        human = row["human"] == "pass"
        if "judge" in row:
            judged = row["judge"] == "pass"
        elif judge is not None:
            verdict = judge.evaluate(
                criteria=row["criteria"], case_input=row.get("input", ""), output=row["output"]
            )
            judged = verdict.passed
        else:
            raise ValueError(f"{path}:{number}: no judge label and no --judge given")
        pairs.append(LabelPair(row["id"], human, judged))
    return pairs, unchecked


def cmd_calibrate(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite) if args.suite else None
    pairs, unchecked = _labels_from(
        args.labels, suite, _judge_from(args.judge), args.allow_test_labels
    )
    result = calibrate(pairs)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        kappa = "undefined" if result.kappa is None else f"{result.kappa:.3f}"
        ci = "n/a" if result.kappa_ci is None else (
            f"{result.kappa_ci[0]:.3f} to {result.kappa_ci[1]:.3f}"
        )
        print(f"labels {result.n}  agreement {result.agreement:.1%}  kappa {kappa} (95% CI {ci})")
        print(f"             human pass  human fail\n  judge pass {result.tp:10}  {result.fp:10}")
        print(f"  judge fail {result.fn:10}  {result.tn:10}")
        for warning in result.warnings:
            print(f"  warning: {warning}")
    if unchecked:
        print(
            f"warning: {unchecked} label(s) have no known split, so test leakage is unchecked",
            file=sys.stderr,
        )
    if args.min_kappa is None:
        return 0
    return 0 if result.kappa is not None and result.kappa >= args.min_kappa else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-golden", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    split = commands.add_parser("split", help="show split sizes and check for leaks")
    split.add_argument("suite")
    split.set_defaults(func=cmd_split)

    run = commands.add_parser("run", help="run a system over one split")
    run.add_argument("suite")
    run.add_argument("--system", required=True, help="module:callable or file.py:callable")
    run.add_argument("--split", choices=SPLITS, default="validation")
    run.add_argument("--name", help="system name recorded in the result file")
    run.add_argument("--judge", help="callable returning a JudgeClient")
    run.add_argument("--out", help="write a result file here")
    run.add_argument("--allow-leaks", action="store_true")
    run.add_argument("--unlock-test", action="store_true", help="permit --split test")
    run.add_argument("--reason", help="why this look at the test split is happening")
    run.add_argument("--max-test-looks", type=int, default=1)
    run.add_argument("--ledger", help="test ledger path, default beside the suite")
    run.set_defaults(func=cmd_run)

    cmp_ = commands.add_parser("compare", help="compare two result files")
    cmp_.add_argument("base")
    cmp_.add_argument("head")
    cmp_.add_argument("--max-regressions", type=int, default=0)
    cmp_.add_argument("--json", action="store_true")
    cmp_.set_defaults(func=cmd_compare)

    cal = commands.add_parser("calibrate", help="score a judge against human labels")
    cal.add_argument("labels", help="JSONL with id, human, and judge or output+criteria")
    cal.add_argument("--suite", help="suite file, used to refuse labels from the test split")
    cal.add_argument("--judge", help="callable returning a JudgeClient")
    cal.add_argument("--min-kappa", type=float, help="exit 1 below this")
    cal.add_argument("--allow-test-labels", action="store_true")
    cal.add_argument("--json", action="store_true")
    cal.set_defaults(func=cmd_calibrate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (SuiteError, CompareError, TestSetLocked, JudgeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
