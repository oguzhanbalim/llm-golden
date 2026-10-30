# llm-golden

Golden-set evals where looking at the test set costs something.

Most eval scripts grade outputs and print a number. That number stops meaning much once the test cases have been read and the prompt tuned against them, and once an expected value has been quietly relaxed to make a case pass. This harness grades deterministically wherever it can, keeps the test split behind a counted lock, and tells a real fix apart from a moved goalpost.

## Quickstart

```
pip install -e .
cd examples/ticket_triage
```

The example is 30 invented support tickets and two scripted fake systems. `scripted_system.py` looks replies up in a table and `scripted_judge.py` applies a regex. No model runs, so the pass rates below only show the harness working.

A new prompt looks better on the validation split:

```
$ llm-golden run cases.yaml --system scripted_system.py:v1 --name v1 --judge scripted_judge.py:make_judge --out v1.json
suite=ticket-triage split=validation system=v1
passed 4/8  failed 4  errors 0  skipped 0  pass rate 50.0% (95% CI 21.5% to 78.5%)
  FAIL    label-11  exact: expected 'account', got 'billing'
  FAIL    amount-05  numeric: ambiguous: found [5.0, 240.0, 12.0]; set pick: first or last
  FAIL    extract-01  json_schema: not JSON: Expecting value at char 0
  FAIL    tone-04  judge: scripted rule

$ llm-golden run cases.yaml --system scripted_system.py:v2 --name v2 --judge scripted_judge.py:make_judge --out v2.json
suite=ticket-triage split=validation system=v2
passed 5/8  failed 3  errors 0  skipped 0  pass rate 62.5% (95% CI 30.6% to 86.3%)
  FAIL    label-01  exact: expected 'billing', got 'bug'
  FAIL    label-11  exact: expected 'account', got 'billing'
  FAIL    ack-03  regex: no match for 'TKT-7781\\b'
```

Comparing them shows the +1 is three fixes and two new breakages, and that this many flips is noise:

```
$ llm-golden compare v1.json v2.json
split=validation  unchanged cases: 8  passed 4 -> 5
  regressions: ack-03, label-01
  fixes: amount-05, extract-01, tone-04
  sign test on flipped cases: p=1.000
```

It exits 1 on any regression, so it can gate CI. The test split needs an explicit unlock and a reason. It reports the aggregate only, and the look is written to a ledger you commit next to the suite:

```
$ llm-golden run cases.yaml --system scripted_system.py:v2 --split test
error: --split test needs --unlock-test and --reason

$ llm-golden run cases.yaml --system scripted_system.py:v2 --name v2 --split test --unlock-test --reason "final check of v2" --judge scripted_judge.py:make_judge
suite=ticket-triage split=test system=v2
passed 6/6  failed 0  errors 0  skipped 0  pass rate 100.0% (95% CI 61.0% to 100.0%)
test split: per-case detail withheld.

$ llm-golden run cases.yaml --system scripted_system.py:v2 --name v2 --split test --unlock-test --reason "just one more" --judge scripted_judge.py:make_judge
error: test split of 'ticket-triage' already looked at 1 time(s), limit is 1. Raising the limit means reporting it as a validation-grade number. History: 2026-10-30T09:19:58+00:00 v2: final check of v2
```

Judge calibration, on two hand-written label files. The second judge agrees more often and is worse, which is the point of kappa:

```
$ llm-golden calibrate labels_balanced.jsonl --min-kappa 0.6
labels 40  agreement 82.5%  kappa 0.650 (95% CI 0.392 to 0.850)
             human pass  human fail
  judge pass         17           4
  judge fail          3          16

$ llm-golden calibrate labels_skewed.jsonl --min-kappa 0.6
labels 40  agreement 90.0%  kappa -0.039 (95% CI -0.084 to 0.000)
             human pass  human fail
  judge pass         36           3
  judge fail          1           0
  warning: human labels are 92% one class: agreement is flattered, read kappa
```

The "human" column in those files is sample data I wrote to show the arithmetic. It is not from a labelling study.

## Case format

YAML, or JSONL with one case per line:

```yaml
suite: ticket-triage
split_salt: v1
split_ratios: {train: 0.5, validation: 0.3, test: 0.2}
cases:
  - id: amount-01
    input: "Refund due: plan 49.90 minus 10.00 credit. Reply with the amount."
    group: refunds          # optional: all of a group lands in one split
    split: validation       # optional: pin instead of hashing
    tags: [numeric]
    graders:                # all must pass
      - {type: numeric, expected: 39.90, abs_tol: 0.01}
```

| Grader | Options | Notes |
|---|---|---|
| `exact` | `expected`, `ignore_case`, `normalize_whitespace` | NFC-normalised, outer whitespace stripped |
| `regex` | `pattern`, `ignore_case`, `fullmatch`, `must_not_match` | Bad patterns fail when the suite loads |
| `numeric` | `expected`, `abs_tol`, `rel_tol`, `pick` | `pick: only` (default) fails an answer containing several numbers, so a hedge cannot pass because one number is right |
| `json_schema` | `schema`, `allow_fence` | Draft 2020-12. Prose around the JSON fails unless `allow_fence` is set |
| `judge` | `criteria` | Uses the injected client. With no client it is skipped, and skipped never counts as passed |

A system is any `Callable[[input], str]`, given as `module:callable` or `file.py:callable`.

## What it catches

- **Test-set tuning.** Asking for the test split without a recorded unlock raises `TestSetLocked`, in the library as well as the CLI. One look is allowed by default. Test result files keep per-case status for regression checks and drop outputs and grader detail.
- **Reshuffled splits.** Assignment hashes the case id or group with a salt, so adding cases never moves an existing one.
- **Duplicate inputs across splits.** `run` refuses if the same input, ignoring case and whitespace, appears in two splits.
- **Moved goalposts.** Each result stores a fingerprint of the case's input and graders. If you relax an expectation, `compare` lists the case as redefined and does not count it as a fix.
- **Coverage loss.** A pass that becomes skipped, for example because the judge was not configured, fails `compare` the same way a regression does.
- **Judge outages read as failures.** A client exception or an unreadable verdict is an `error`, kept apart from `fail`.
- **Judges that are only agreeable.** `calibrate` reports kappa with a seeded bootstrap interval, warns on skewed labels and on small samples, returns "undefined" rather than a number when kappa is 0/0, and refuses labels from the test split when it can tell.

## Optional model judge

`llm_golden.anthropic_client.AnthropicJudgeClient` implements the judge against the Messages API with the default model `claude-opus-5-5` (`pip install -e '.[anthropic]'`). **It has never been run live.** Its tests use a fake SDK client and only check the request shape and the handling of refusals. Any object with `complete(system, prompt) -> str` can take its place.

Not affiliated with Anthropic.

## Limits

- The lock is a speed bump. Anyone can read the YAML, raise `--max-test-looks`, rename the suite or delete the ledger. It makes tuning against the test set deliberate and visible, and it cannot make it impossible.
- Leak detection only finds identical inputs after normalisation. Paraphrases get through; use `group` for related cases.
- Hash splits only approximate the ratios, and small suites can land far from them (the 30-case example gets 16/8/6 against 15/9/6).
- The numeric grader reads dot-decimal numbers only. `1.234,50` will not parse as one number.
- The sign test ignores effect size and treats every case as equally important. Intervals assume independent cases, and generated near-duplicates break that.
- The bundled systems, judge and labels are scripted examples I wrote, so nothing here says how good any model is.
- Runs are sequential, with no retries, caching or parallelism.

## Development

```
pip install -e '.[dev]'
ruff check .
pytest
```

MIT licence.
