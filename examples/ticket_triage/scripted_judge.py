"""A scripted fake judge client, standing in for a model so the example runs offline.

It fails a reply that contains blaming words and passes the rest. It is a stub with
the JudgeClient shape and says nothing about how any real judge would behave.
"""

import json
import re


class ScriptedJudge:
    def complete(self, system: str, prompt: str) -> str:
        output = prompt.split("<output>")[-1]
        blames = re.search(r"you should have|your fault", output, re.IGNORECASE)
        verdict = "fail" if blames else "pass"
        return json.dumps({"verdict": verdict, "reason": "scripted rule"})


def make_judge() -> ScriptedJudge:
    return ScriptedJudge()
