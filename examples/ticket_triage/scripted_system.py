"""Two scripted fake systems. No model runs here: replies are looked up in a table.

v1 and v2 start from the same correct replies and then break different cases, so
the regression report has something real to find.
"""

import json
from pathlib import Path

_GOOD = json.loads((Path(__file__).parent / "replies_good.json").read_text("utf-8"))


def _lookup(prompt: str, overrides: dict[str, str]) -> str:
    for needle, reply in overrides.items():
        if needle in prompt:
            return reply
    return _GOOD[prompt]


# v1 confuses labels, returns chatty numbers, breaks JSON and blames a customer.
V1_MISTAKES = {
    "How do I add a second admin": "billing",
    "Search returns a blank page": "account",
    "Late fee is 5 percent": "5 percent of 240 is 12.",
    "Two refunds of 18.25": "36.00",
    "Acknowledge ticket 3350": "Thanks, logged as TKT-3350. We guarantee a fix today.",
    "Sample-A": "Customer: Sample-A, priority urgent",
    "Sample-D": '{"customer": "Sample-D", "priority": "critical"}',
    "a feature does not exist": "You should have read the roadmap before asking.",
}

# v2 fixes most of those and breaks two cases v1 had right.
V2_MISTAKES = {
    "How do I add a second admin": "billing",
    "Charged twice for the March plan": "bug",
    "Acknowledge ticket 7781": "Thanks, we are looking into it.",
}


def v1(prompt: str) -> str:
    return _lookup(prompt, V1_MISTAKES)


def v2(prompt: str) -> str:
    return _lookup(prompt, V2_MISTAKES)
