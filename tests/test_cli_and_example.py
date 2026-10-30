from __future__ import annotations

import json
import shutil

import pytest

from llm_golden.anthropic_client import AnthropicJudgeClient
from llm_golden.cli import load_callable, main
from llm_golden.judge import JUDGE_SYSTEM, JudgeError, LLMJudge


@pytest.fixture
def workdir(example_dir, tmp_path, monkeypatch):
    target = tmp_path / "triage"
    shutil.copytree(example_dir, target, ignore=shutil.ignore_patterns("out", "*.jsonl.lock"))
    monkeypatch.chdir(target)
    return target


def _run(system, out, split="validation", *extra):
    return main(
        [
            "run",
            "cases.yaml",
            "--system",
            f"scripted_system.py:{system}",
            "--name",
            system,
            "--split",
            split,
            "--judge",
            "scripted_judge.py:make_judge",
            "--out",
            out,
            *extra,
        ]
    )


def test_example_regression_flow(workdir, capsys):
    assert main(["split", "cases.yaml"]) == 0
    assert _run("v1", "v1.json") == 1
    assert _run("v2", "v2.json") == 1
    capsys.readouterr()
    assert main(["compare", "v1.json", "v2.json"]) == 1
    text = capsys.readouterr().out
    assert "regressions: ack-03, label-01" in text
    assert "fixes: amount-05, extract-01, tone-04" in text
    assert main(["compare", "v1.json", "v2.json", "--max-regressions", "2"]) == 0


def test_cli_refuses_test_split_without_unlock(workdir, capsys):
    assert _run("v2", "t.json", "test") == 2
    assert "needs --unlock-test" in capsys.readouterr().err


def test_cli_test_look_budget(workdir, capsys):
    args = ("--unlock-test", "--reason", "final check of v2")
    assert _run("v2", "t.json", "test", *args) in (0, 1)
    out = capsys.readouterr().out
    assert "per-case detail withheld" in out
    assert _run("v2", "t2.json", "test", *args) == 2
    assert "already looked at 1" in capsys.readouterr().err


def test_bad_system_does_not_spend_the_look_budget(workdir):
    args = ["run", "cases.yaml", "--system", "scripted_system.py:nope", "--split", "test"]
    args += ["--unlock-test", "--reason", "r"]
    with pytest.raises(AttributeError):
        main(args)
    assert not (workdir / "cases.test-ledger.jsonl").exists()


def test_calibrate_cli_gate(workdir, capsys):
    assert main(["calibrate", "labels_balanced.jsonl", "--min-kappa", "0.6"]) == 0
    assert main(["calibrate", "labels_skewed.jsonl", "--min-kappa", "0.6"]) == 1
    assert "flattered" in capsys.readouterr().out


def test_calibrate_refuses_test_split_labels(workdir, capsys):
    rows = [{"id": "label-05", "human": "pass", "judge": "pass"}]
    (workdir / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert main(["calibrate", "labels.jsonl", "--suite", "cases.yaml"]) == 2
    assert "test split" in capsys.readouterr().err
    assert main(["calibrate", "labels.jsonl", "--suite", "cases.yaml", "--allow-test-labels"]) == 0


def test_calibrate_warns_when_split_is_unknown(workdir, capsys):
    (workdir / "labels.jsonl").write_text(json.dumps({"id": "x", "human": "pass", "judge": "pass"}))
    assert main(["calibrate", "labels.jsonl"]) == 0
    assert "unchecked" in capsys.readouterr().err


def test_calibrate_can_run_the_judge(workdir, capsys):
    row = {"id": "x", "split": "train", "human": "fail", "output": "you should have known",
           "criteria": "be kind"}
    (workdir / "labels.jsonl").write_text(json.dumps(row))
    assert main(["calibrate", "labels.jsonl", "--judge", "scripted_judge.py:make_judge"]) == 0
    assert "agreement 100.0%" in capsys.readouterr().out


def test_load_callable_rejects_bad_spec():
    with pytest.raises(ValueError):
        load_callable("no_colon_here")


class _Block:
    def __init__(self, type_, text=""):
        self.type, self.text = type_, text


class _Response:
    def __init__(self, stop_reason, blocks):
        self.stop_reason, self.content = stop_reason, blocks


class _FakeSdk:
    """Stands in for the SDK client. Nothing here reaches the network."""

    def __init__(self, response):
        self.response, self.calls = response, []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_anthropic_client_request_shape_and_text_extraction():
    sdk = _FakeSdk(
        _Response(
            "end_turn",
            [_Block("thinking"), _Block("text", '{"verdict": "pass", "reason": "ok"}')],
        )
    )
    verdict = LLMJudge(AnthropicJudgeClient(sdk_client=sdk)).evaluate(
        criteria="c", case_input="i", output="o"
    )
    assert verdict.passed
    call = sdk.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["system"] == JUDGE_SYSTEM
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert not {"temperature", "top_p", "top_k", "tool_choice", "thinking"} & call.keys()
    assert call["messages"][-1]["role"] == "user"


def test_anthropic_client_turns_refusal_into_judge_error():
    client = AnthropicJudgeClient(sdk_client=_FakeSdk(_Response("refusal", [])))
    with pytest.raises(JudgeError, match="refusal"):
        client.complete("s", "p")
