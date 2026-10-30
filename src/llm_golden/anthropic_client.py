"""Optional Messages API judge client. Never run live by this repository's tests.

Install with ``pip install 'llm-golden[anthropic]'``. The SDK is imported only when
no client is injected, so the rest of the package has no dependency on it.
"""

from __future__ import annotations

from typing import Any

from llm_golden.judge import VERDICT_SCHEMA, JudgeError


class AnthropicJudgeClient:
    def __init__(
        self, model: str = "claude-opus-5-5", max_tokens: int = 4096, sdk_client: Any = None
    ) -> None:
        if sdk_client is None:
            import anthropic

            sdk_client = anthropic.Anthropic()
        self._sdk = sdk_client
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, system: str, prompt: str) -> str:
        response = self._sdk.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
        )
        if response.stop_reason != "end_turn":
            raise JudgeError(f"judge stopped with stop_reason={response.stop_reason!r}")
        return "".join(block.text for block in response.content if block.type == "text")
