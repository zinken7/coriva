"""Audit alias of the parser's delimiter-repair solver (`analysis/json_repair.py`).

One live implementation: the audit probes and the production parser share it,
so an audit candidate is exactly what the parser would accept.
"""
from analysis.json_repair import (  # noqa: F401
    CONTRACT_MARKERS, PUBLIC_SCHEMA, REQUIRED, SCHEMA_PATH, SCHEMA_VERSION, SHAPES,
    apply_insertions, candidate_text, json_tokens, missing_fields, repair_missing_delimiters,
    repair_structure, structure_provenance)
