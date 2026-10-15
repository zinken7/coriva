"""Insertion-only recovery of a model answer that lacks `{}[]` delimiters.

Parser policy: the parser may repair an answer
whose only defect is missing delimiters. The search inserts the minimum number
of `{}[]` at token boundaries and accepts a structure only when the public
output contract's container shapes (property ownership from
`ctms/schema_branch.json`) admit exactly ONE minimal structure. It never adds
or changes a key, value, comma, quote or digit; a formula, a surplus closer, a
stray quote, a missing key or colon, an unknown key, several minimal
structures or an exhausted search leave the answer unrepaired (the model's
error). Missing required fields are reported, never filled: the gates judge
them. `scripts/audit_json_outputs.py` reuses the same solver read-only.
"""
from collections import deque
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / 'ctms/schema_branch.json'
PUBLIC_SCHEMA = json.loads(SCHEMA_PATH.read_text())
SCHEMA_VERSION = 'coriva-audit-structure-1.0'


def _shape(spec):
    if '$ref' in spec:
        name = spec['$ref'].rsplit('/', 1)[-1]
        return '@scalar' if name == 'bbox' else name
    if spec.get('type') == 'array':
        return '@' + _shape(spec['items'])
    return 'scalar'


SHAPES = {name: {key: _shape(spec) for key, spec in node['properties'].items()}
          for name, node in PUBLIC_SCHEMA['$defs'].items() if node.get('type') == 'object'}
SHAPES['branch'] = {key: _shape(spec) for key, spec in PUBLIC_SCHEMA['properties'].items()}
SHAPES['pair'] = {'intervention': 'intervention', 'control_chain': '@node',
                  'treated_chain': '@node', 'chain': '@node'}
SHAPES['response'] = {'predicted_branches': '@branch', 'predicted_counterfactuals': '@pair',
                      'branches': '@branch', 'counterfactuals': '@pair'}
# Matched-pair structure and field lists are stated in the frozen system
# prompt's OUTPUT CONTRACT. The aliases above are documented by parse_prediction.
REQUIRED = {
    'response': ('predicted_branches', 'predicted_counterfactuals'),
    'branch': ('branch_id', 'is_factual', 'intervention', 'chain'),
    'pair': ('intervention', 'control_chain', 'treated_chain'),
    'intervention': ('intervention_type', 'target_actor_id', 'trigger_depth'),
    'node': ('depth', 'timestamp', 'event_type', 'actors_update'),
    'actor_state': ('id', 'entity_type', 'role_type', 'x', 'y', 'vx', 'vy', 'speed'),
}
CONTRACT_MARKERS = (
    '- Top-level keys: `predicted_branches`, `predicted_counterfactuals`.',
    '- Each ego-action branch: `branch_id`, `is_factual`, `intervention`, `chain`.',
    '- Each target-intervention pair: `intervention`, `control_chain`, `treated_chain`.',
    '- Each intervention: `intervention_type`, `target_actor_id`, `trigger_depth`.',
    '- Each future node: `depth`, `timestamp`, `event_type`, `actors_update`, plus',
    '- Each `actors_update` is an ARRAY of full actor objects',
    '- For each actor output `id`, `entity_type`, `role_type`, `x`, `y`, `vx`,',
    '`vy`, `speed`.',
)


def structure_provenance():
    return {'version': SCHEMA_VERSION, 'schema_file': str(SCHEMA_PATH),
            'schema_sha256': hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest(),
            'shape_sha256': hashlib.sha256(json.dumps(SHAPES, sort_keys=True).encode()).hexdigest(),
            'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'policy': 'minimum delimiter insertions; property ownership and container shapes only'}


def missing_fields(obj, kind='response', path='$'):
    """Diagnostics only; this does not filter or rank candidate answers."""
    issues = []
    if kind.startswith('@'):
        for i, item in enumerate(obj):
            issues.extend(missing_fields(item, kind[1:], f'{path}[{i}]'))
    elif kind in SHAPES:
        missing = [key for key in REQUIRED.get(kind, ()) if key not in obj]
        if missing:
            issues.append({'path': path, 'missing': missing})
        for key, value in obj.items():
            issues.extend(missing_fields(value, SHAPES[kind][key], f'{path}.{key}'))
    return issues


def _step(stack, token, inserted=False):
    # Frame: kind, phase, seen object keys, expected child kind, fresh opener.
    # "fresh" means an inserted opener with no original token consumed yet;
    # closing it with another insertion would invent an empty value.
    kind, phase, seen, child, fresh = stack[-1]
    prefix = stack[:-1]

    def frame(new_phase, expected=child, keys=seen):
        return prefix + ((kind, new_phase, keys, expected, fresh),)

    def close():
        return None if inserted and fresh else prefix

    def value(expected):
        parent = frame('after')
        if expected == 'scalar':
            return parent if token not in ('{', '}', '[', ']', ',', ':') else None
        opener = '[' if expected.startswith('@') else '{'
        if token != opener:
            return None
        return parent + ((expected, 'first', frozenset(), '', inserted),)

    if kind == 'document':
        result = value('response') if phase == 'value' else None
    elif kind.startswith('@'):
        if phase == 'after':
            result = frame('value') if token == ',' else close() if token == ']' else None
        elif phase == 'first' and token == ']':
            result = close()
        else:
            result = value(kind[1:])
    elif phase in ('first', 'key'):
        if phase == 'first' and token == '}':
            result = close()
        elif token.startswith('"'):
            key = json.loads(token)
            result = (frame('colon', SHAPES[kind][key], seen | {key})
                      if key in SHAPES[kind] and key not in seen else None)
        else:
            result = None
    elif phase == 'colon':
        result = frame('value') if token == ':' else None
    elif phase == 'value':
        result = value(child)
    else:
        result = frame('key') if token == ',' else close() if token == '}' else None
    if result is not None and not inserted:
        result = tuple((k, p, s, c, False) for k, p, s, c, f in result)
    return result


def _strict_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('duplicate key')
        obj[key] = value
    return obj


def _candidate(text, edits):
    repaired = text
    for offset, char in reversed(edits):
        repaired = repaired[:offset] + char + repaired[offset:]
    obj = json.loads(repaired, object_pairs_hook=_strict_object)
    # Normalize equivalent positions of adjacent identical closers.
    normalized, offset = [], 0
    for char in repaired:
        if offset < len(text) and char == text[offset]:
            offset += 1
        elif normalized and normalized[-1]['offset'] == offset:
            normalized[-1]['text'] += char
        else:
            normalized.append({'offset': offset, 'text': char})
    internal = sum(len(e['text']) for e in normalized if text[e['offset']:].strip())
    return json.dumps(obj, sort_keys=True), {
        'insertions': normalized, 'internal_insertions': internal,
        'eof_insertions': len(edits) - internal,
        'sha256': hashlib.sha256(repaired.encode()).hexdigest(),
        'missing_contract_fields': missing_fields(obj),
    }


def _repair_at_budget(text, tokens, *, max_edits, max_states, max_paths):
    """Streaming dynamic programming over the entire token sequence.

    Retain minimum-cost prefixes per grammar state and distinct token traces.
    Unlike local error-window repair, insertions may occur at ANY boundary.
    All minimum-insertion results are considered unless a stated bound is hit.
    A trace limit is explicit, never presented as proof of a unique structure.
    """
    result = {'status': 'unresolved', 'candidate_count': 0, 'candidates': [],
              'search_complete': True, 'states_checked': 0, 'reason': None,
              'schema_version': SCHEMA_VERSION, 'method': 'schema_minimum_delimiter_insertion',
              'limits': {'max_edits': max_edits, 'max_states': max_states, 'max_paths': max_paths},
              'minimal_insertions': None, 'eof_scalar_preserved': False}
    if tokens is None or not tokens:
        result['reason'] = 'non_json_token_or_unterminated_string' if text.strip() else 'empty_response'
        return result
    result['eof_scalar_preserved'] = tokens[-1][0] not in ('{', '}', '[', ']', ',', ':')
    initial = (('document', 'value', frozenset(), 'response', False),)
    # Each state stores its cheapest cost and {token-trace digest: insertions}.
    states = {initial: (0, {b'': ()})}
    edit_limit = False
    for index in range(len(tokens) + 1):
        token = tokens[index][0] if index < len(tokens) else None
        offset = tokens[index][1] if index < len(tokens) else len(text)
        agenda = deque((stack, cost, digest, edits)
                       for stack, (cost, paths) in states.items() for digest, edits in paths.items())
        following = {}

        def retain(table, stack, cost, digest, edits):
            previous = table.get(stack)
            if previous is None or cost < previous[0]:
                table[stack] = (cost, {digest: edits})
                return True
            if cost > previous[0] or digest in previous[1]:
                return False
            if len(previous[1]) >= max_paths:
                result.update(search_complete=False, reason='max_paths')
                return False
            previous[1][digest] = edits
            return True

        while agenda:
            stack, cost, digest, edits = agenda.popleft()
            if cost != states[stack][0]:
                continue
            if result['states_checked'] >= max_states:
                result.update(status='search_limit', search_complete=False, reason='max_states')
                return result
            result['states_checked'] += 1
            if token is not None:
                advanced = _step(stack, token)
                if advanced is not None:
                    trace = hashlib.sha256(digest + token.encode()).digest()
                    retain(following, advanced, cost, trace, edits)
            for char in '{}[]':
                advanced = _step(stack, char, inserted=True)
                if advanced is None:
                    continue
                if cost >= max_edits:
                    edit_limit = True
                    continue
                trace = hashlib.sha256(digest + char.encode()).digest()
                new_edits = edits + ((offset, char),)
                if retain(states, advanced, cost + 1, trace, new_edits):
                    agenda.append((advanced, cost + 1, trace, new_edits))
        if token is None:
            break
        states = following
        if not states:
            result['reason'] = 'no_structure_preserving_parse'
            result['furthest_token_offset'] = offset
            break
    complete = [(cost, paths) for stack, (cost, paths) in states.items()
                if len(stack) == 1 and stack[0][1] == 'after'] if index == len(tokens) else []
    if complete:
        best = min(cost for cost, _ in complete)
        candidates = {}
        for cost, paths in complete:
            if cost != best:
                continue
            for edits in paths.values():
                canonical, candidate = _candidate(text, edits)
                if (canonical not in candidates or candidate['internal_insertions']
                        < candidates[canonical]['internal_insertions']):
                    candidates[canonical] = candidate
        result.update(minimal_insertions=best, candidate_count=len(candidates),
                      candidates=list(candidates.values()))
        result['status'] = 'ambiguous' if len(candidates) > 1 else 'candidate_found'
    elif edit_limit:
        result.update(search_complete=False, reason='max_edits')
    if not result['search_complete'] and result['status'] != 'ambiguous':
        result['status'] = 'search_limit'
    return result


def repair_structure(text, tokens, *, max_edits=64, max_states=2_000_000, max_paths=32):
    """Increase the edit budget only when needed, sharing a total state budget.

    This avoids spending most of the search on expensive alternative parses
    when one or two missing brackets suffice. Each attempted budget explores
    the full token sequence, not a fixed window around the first syntax error.
    """
    if min(max_edits, max_states, max_paths) < 1:
        raise ValueError('Search limits must be positive')
    if tokens:
        keys = {json.loads(t[0]) for i, t in enumerate(tokens[:-1])
                if t[0].startswith('"') and tokens[i + 1][0] == ':'}
        reachable = {'response'}
        pending = ['response']
        while pending:
            kind = pending.pop()
            for key, child in SHAPES[kind].items():
                child = child.lstrip('@')
                if key in keys and child in SHAPES and child not in reachable:
                    reachable.add(child)
                    pending.append(child)
        supported = set().union(*(set(SHAPES[kind]) for kind in reachable))
        unsupported = sorted(keys - supported)
        if unsupported:
            result = _repair_at_budget('', [], max_edits=max_edits,
                                       max_states=max_states, max_paths=max_paths)
            result.update(reason='unreachable_contract_fields', unsupported_fields=unsupported,
                          attempted_edit_budgets=[])
            return result
    total, attempted, budget = 0, [], 1
    while True:
        budget = min(budget, max_edits)
        attempted.append(budget)
        result = _repair_at_budget(text, tokens, max_edits=budget,
                                  max_states=max_states - total, max_paths=max_paths)
        total += result['states_checked']
        if (result['candidate_count'] or result['search_complete']
                or result['reason'] == 'max_states' or budget == max_edits):
            break
        budget *= 2
    result['states_checked'] = total
    result['attempted_edit_budgets'] = attempted
    result['limits'] = dict(max_edits=max_edits, max_states=max_states, max_paths=max_paths)
    return result


# ── the parser's entry point ─────────────────────────────────────────────

JSON_TOKEN = re.compile(
    r'"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"'
    r'|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?'
    r'|true|false|null|[{}\[\],:]')
_FENCE = re.compile(r'```(?:json)?\s*(.*?)```', re.DOTALL)


def json_tokens(text: str) -> Optional[List[Tuple[str, int, int]]]:
    """Lex `text` as JSON tokens (token, start, end); None at the first
    non-JSON character (an expression, an unfinished string, prose)."""
    tokens, pos = [], 0
    while pos < len(text):
        if text[pos] in ' \t\r\n':
            pos += 1
            continue
        match = JSON_TOKEN.match(text, pos)
        if not match:
            return None
        tokens.append((match.group(), pos, match.end()))
        pos = match.end()
    return tokens


def candidate_text(text: str) -> str:
    """The part of a model answer that should be the JSON object: a complete
    fenced block if there is one, else everything from the first `{`/`[`
    (or a leading `"` when the object's own opener is missing), with a
    dangling closing fence removed. Extraction only; nothing is inserted."""
    match = _FENCE.search(text or '')
    cand = (match.group(1) if match else (text or '')).strip()
    while cand.endswith('```'):
        cand = cand[:-3].rstrip()
    if cand.startswith('"'):
        return cand
    starts = [p for p in (cand.find('{'), cand.find('[')) if p >= 0]
    return cand[min(starts):] if starts else ''


def apply_insertions(text: str, insertions) -> str:
    repaired = text
    for edit in sorted(insertions, key=lambda e: -e['offset']):
        repaired = repaired[:edit['offset']] + edit['text'] + repaired[edit['offset']:]
    return repaired


def repair_missing_delimiters(text: str, **limits) -> Optional[Dict[str, Any]]:
    """Return the parsed object when `text` becomes valid JSON by inserting
    delimiters and the contract admits exactly one minimal structure; else
    None. Duplicate keys are rejected by the solver."""
    cand = candidate_text(text)
    if not cand:
        return None
    tokens = json_tokens(cand)
    if not tokens:
        return None
    result = repair_structure(cand, tokens, **limits)
    if result['status'] != 'candidate_found' or not result['search_complete']:
        return None
    if result['candidate_count'] != 1:
        return None
    try:
        obj = json.loads(apply_insertions(cand, result['candidates'][0]['insertions']),
                         object_pairs_hook=_strict_object)
    except (ValueError, RecursionError):
        return None
    return obj if isinstance(obj, dict) else None
