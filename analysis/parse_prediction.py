"""Model-runner core — turn a VLM's raw text output into CORIVA prediction files.

Backend-agnostic: whatever produces the text (local Qwen3-VL on GPU, or an API),
this extracts the JSON and normalizes it into the l3_pre.json / l4_pre.json shape
that scripts/assemble_dataset.py consumes:

    l3_pre.json = {"scenario_id", "predicted_branches": [...]}
    l4_pre.json = {"scenario_id", "predicted_counterfactuals":
                       [{"intervention", "control_chain", "treated_chain"}]}
                  (`chain` is accepted by the scorer as the treated alias)

Lenient by design: a malformed or partial answer still yields (empty) files plus
an error list, so the scenario scores (CTMS gates penalize it) instead of
crashing the whole run.

An answer that is valid JSON except for missing `{}[]` delimiters is repaired
by `analysis.json_repair` under the parser policy: minimum
insertions, accepted only when the output contract admits exactly one such
structure. Anything else (a formula, a surplus closer, a stray quote, a missing
key) stays unparsed and is the model's error.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from analysis.json_repair import repair_missing_delimiters

PARSER_VERSION = "coriva-parser-0.8"

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_RESPONSE_KEYS = ("predicted_branches", "predicted_counterfactuals", "branches", "counterfactuals")


def _first_json_object(s: str) -> Optional[str]:
    """Return the first balanced {...} substring, respecting string literals."""
    start = s.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            c = s[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
        start = s.find("{", start + 1)
    return None


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull a JSON object out of noisy model text. Tries fenced ```json blocks
    first, then the first balanced brace span, then a missing-delimiter repair
    of the answer (see module docstring). Returns None if nothing parses."""
    candidates: List[str] = []
    for m in _FENCE.finditer(text or ""):
        candidates.append(m.group(1))
    candidates.append(text or "")
    fallback: Optional[Dict[str, Any]] = None
    for cand in candidates:
        obj = _first_json_object(cand)
        if obj is None:
            continue
        try:
            parsed = json.loads(obj)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            if any(key in parsed for key in _RESPONSE_KEYS):
                return parsed
            # A balanced inner object (e.g. one `intervention`) is not the
            # answer: the outer object is broken. Try the delimiter repair
            # first; keep the old behaviour when it declines.
            if fallback is None:
                fallback = parsed
    repaired = repair_missing_delimiters(text or "")
    return repaired if repaired is not None else fallback


def to_prediction_files(obj: Dict[str, Any], scenario_id: str
                        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Normalize a parsed prediction dict into (l3_pre, l4_pre). Accepts the
    canonical `predicted_branches`/`predicted_counterfactuals` keys or the plain
    `branches`/`counterfactuals` aliases a model may emit."""
    obj = obj or {}
    branches = obj.get("predicted_branches", obj.get("branches", [])) or []
    cfs = obj.get("predicted_counterfactuals", obj.get("counterfactuals", [])) or []
    l3 = {"scenario_id": scenario_id, "predicted_branches": branches}
    l4 = {"scenario_id": scenario_id, "predicted_counterfactuals": cfs}
    return l3, l4


def parse_model_output(text: str, scenario_id: str
                       ) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """text -> (l3_pre, l4_pre, errors). Empty predictions + a non-empty error
    list when the output cannot be parsed (so the scenario still scores)."""
    errors: List[str] = []
    obj = extract_json(text)
    if obj is None:
        errors.append("no parseable JSON object in model output")
        obj = {}
    l3, l4 = to_prediction_files(obj, scenario_id)
    if not l3["predicted_branches"]:
        errors.append("no predicted_branches found")
    return l3, l4, errors
