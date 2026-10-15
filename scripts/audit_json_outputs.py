"""Read-only audit of API-panel JSON syntax and recorded decoding settings.

Usage (from Coding):
  .venv/bin/python scripts/audit_json_outputs.py
  .venv/bin/python scripts/audit_json_outputs.py --model qwen3.5-9b-thinking
  .venv/bin/python scripts/audit_json_outputs.py --schema-only
  .venv/bin/python scripts/audit_json_outputs.py --out /tmp/qwen-json-audit.json

Defaults to all qwen* configurations in results/api_panel. Reads final_response
(the final-answer portion of raw generation), never the private reference or
scorer. It neither repairs saved outputs nor scores predictions. The optional
delimiter-insertion probe runs ONLY in memory and does not change parser results.
Its success means syntactically parseable, NOT complete or semantically correct.
The schema probe is the parser's own solver (analysis/json_repair.py): a
single complete candidate is exactly what the current parser accepts.
The bounded, error-guided search inserts only {}[] at token boundaries, including
inside JSON. It reports alternative structures and search limits; finding one
candidate does not prove that it is the intended structure. It never evaluates
expressions, adds values, or consults references. Offsets are zero-based Python
character offsets in the extracted JSON candidate (not UTF-8 byte offsets).
The additional schema probe uses all token boundaries and the public output
contract, verified against each run's saved system prompt. It retains complete
scalar tokens at EOF, reports missing fields without filling them, and searches
minimum-insertion structures rather than choosing from scores or references.
--schema-only skips the slower legacy syntax search (EOF comparison remains).

Flags overlap. Delimiters counted here are characters outside quoted strings,
not tokenizer IDs. Associations with recorded sampling settings cannot isolate
the causal effect of presence_penalty. Missing metadata remains unknown.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from bisect import bisect_left
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.parse_prediction import PARSER_VERSION, parse_model_output
from scripts.json_structure_probe import repair_structure, structure_provenance, CONTRACT_MARKERS

AUDIT_VERSION = 'coriva-json-audit-1.2'
CONDITIONS = ('video', 'history_json', 'video_history')
FENCE = re.compile(r'```(?:json)?\s*(.*?)```', re.DOTALL)
ARITHMETIC = re.compile(r'(?<![\w.])[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*[+*/-]\s*\d')
JSON_TOKEN = re.compile(
    r'"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"'
    r'|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?'
    r'|true|false|null|[{}\[\],:]')


def _json_tokens(text):
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


def schema_repair_probe(text, **limits):
    return repair_structure(text, _json_tokens(text), **limits)


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate object key')
        result[key] = value
    return result


def _insertion_sites(tokens, error_offset, window):
    """Nearby boundaries plus the start of the preceding complete container.

    This is intentionally a bounded heuristic, not an exhaustive repair solver.
    Looking across a complete container permits an absent '[' before a long
    object to be found when the first error occurs after that object.
    """
    index = bisect_left([t[1] for t in tokens], error_offset)
    sites = set(range(max(0, index - window), min(len(tokens), index) + 1))
    cursor = min(index, len(tokens)) - 1
    if cursor >= 0 and tokens[cursor][0] == ',':
        cursor -= 1
    if cursor >= 0 and tokens[cursor][0] in ('}', ']'):
        depth = 1
        cursor -= 1
        while cursor >= 0 and depth:
            char = tokens[cursor][0]
            if char in ('}', ']'):
                depth += 1
            elif char in ('{', '['):
                depth -= 1
            cursor -= 1
        if depth == 0:
            sites.add(cursor + 1)
    return sorted(sites)


def delimiter_repair_probe(text, *, max_edits=6, max_states=512, window=6):
    """Try insertion-only repairs without returning or saving repaired answers.

    Breadth-first search reports distinct structures at the lowest edit depth
    found. Search is bounded and error-guided: even a completed search does NOT
    establish uniqueness over all possible repairs. Original text is a literal
    subsequence of every candidate. Duplicate keys and fabricated empty values
    are rejected; numbers/strings/keys/commas are never edited or supplied.
    """
    result = {'status': 'unresolved', 'candidate_count': 0, 'candidates': [],
              'search_complete': True, 'states_checked': 0,
              'method': 'bounded_error_guided_delimiter_insertion',
              'limits': {'max_edits': max_edits, 'max_states': max_states,
                         'lookback_tokens': window}, 'reason': None}
    tokens = _json_tokens(text)
    if tokens is None or not tokens:
        result['reason'] = 'non_json_token_or_unterminated_string'
        return result
    # Preserve complete scalar tokens at EOF exactly as emitted. The lexer
    # rejects unfinished strings, exponents and expressions; no digits are added.
    result['eof_scalar_preserved'] = tokens[-1][0] not in ('{', '}', '[', ']', ',', ':')
    # Keep only insertion offsets, not a full character map per search state.
    frontier = [(text, ())]
    seen = {hashlib.sha256(text.encode()).digest()}
    structures = {}
    for depth in range(max_edits + 1):
        next_frontier = []
        frontier_overflow = False
        for current, insertions in frontier:
            if result['states_checked'] >= max_states:
                result.update(search_complete=False, reason='max_states')
                break
            result['states_checked'] += 1
            inserted_positions = [offset + i for i, (offset, _) in enumerate(insertions)]
            try:
                obj = json.loads(current, object_pairs_hook=_no_duplicate_keys)
            except json.JSONDecodeError as exc:
                error_offset = exc.pos
            except (ValueError, RecursionError):
                continue
            else:
                if not isinstance(obj, dict):
                    continue
                current_tokens = _json_tokens(current)
                stack, invented_empty = [], False
                for token, start, end in current_tokens:
                    if token in ('{', '['):
                        stack.append((start, end))
                    elif token in ('}', ']'):
                        opening, after_opening = stack.pop()
                        if opening in inserted_positions and start in inserted_positions:
                            if not any(p not in inserted_positions and not current[p].isspace()
                                       for p in range(after_opening, start)):
                                invented_empty = True
                                break
                if invented_empty:
                    continue
                canonical = json.dumps(obj, sort_keys=True, ensure_ascii=False)
                edits, original_offset = [], 0
                # Greedy subsequence alignment puts identical adjacent closers
                # at the latest position; insertion order must not inflate the
                # internal-repair count for an EOF-only correction.
                for char in current:
                    if original_offset < len(text) and char == text[original_offset]:
                        original_offset += 1
                        continue
                    offset = original_offset
                    if edits and edits[-1]['offset'] == offset:
                        edits[-1]['text'] += char
                    else:
                        edits.append({'offset': offset, 'text': char})
                internal = sum(len(e['text']) for e in edits if text[e['offset']:].strip())
                # Moving a closer across an existing closer can yield the same
                # structure. Prefer EOF for equivalent repairs, rather than
                # counting that response as requiring an internal insertion.
                if (canonical not in structures
                        or internal < structures[canonical]['internal_insertions']):
                    structures[canonical] = {
                        'insertions': edits, 'internal_insertions': internal,
                        'eof_insertions': depth - internal,
                        'sha256': hashlib.sha256(current.encode()).hexdigest()}
                continue
            if depth == max_edits:
                continue
            current_tokens = _json_tokens(current)
            starts = [t[1] for t in current_tokens]
            for index in _insertion_sites(current_tokens, error_offset, window):
                offset = starts[index] if index < len(starts) else len(current)
                following = current_tokens[index][0] if index < len(starts) else ''
                # Closers only before separators/closers/EOF. Open objects
                # only before a key-colon pair; open arrays before a value.
                chars = '}]' if following in (',', '}', ']', '') else ''
                if (following.startswith('"') and index + 1 < len(starts)
                        and current_tokens[index + 1][0] == ':'):
                    chars += '{'
                if following and following not in (',', ':', '}', ']'):
                    chars += '['
                for char in chars:
                    proposed = current[:offset] + char + current[offset:]
                    digest = hashlib.sha256(proposed.encode()).digest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    # Queue size is also bounded to avoid allocating many
                    # full-size copies for a long malformed model response.
                    if len(next_frontier) >= max_states:
                        frontier_overflow = True
                        continue
                    insertion_index = bisect_left(inserted_positions, offset)
                    original_offset = offset - insertion_index
                    edits = (insertions[:insertion_index] + ((original_offset, char),)
                             + insertions[insertion_index:])
                    next_frontier.append((proposed, edits))
        if structures or not next_frontier or result['states_checked'] >= max_states:
            if not structures and depth == max_edits:
                result.update(search_complete=False, reason='max_edits')
            elif not structures and next_frontier and result['states_checked'] >= max_states:
                result.update(search_complete=False, reason='max_states')
            break
        if frontier_overflow:
            result.update(search_complete=False, reason='frontier_limit')
        frontier = next_frontier
    result['candidate_count'] = len(structures)
    result['candidates'] = list(structures.values())
    if len(structures) > 1:
        result['status'] = 'ambiguous'
    elif not result['search_complete']:
        result['status'] = 'search_limit'
    elif structures:
        result['status'] = 'candidate_found'
    return result


def delimiter_scan(text):
    stack, mismatches, counts = [], [], Counter()
    quoted = escaped = False
    masked = []
    for pos, char in enumerate(text):
        if quoted:
            masked.append(' ')
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
            masked.append(' ')
            continue
        masked.append(char)
        if char in '{}[]':
            counts[char] += 1
        if char in '{[':
            stack.append((char, pos))
        elif char in '}]':
            if stack and stack[-1][0] == {'}': '{', ']': '['}[char]:
                stack.pop()
            else:
                mismatches.append(pos)
    return ({'counts_outside_strings': dict(counts),
             'unclosed_count': len(stack),
             'unclosed_delimiters': ''.join(x[0] for x in stack),
             'mismatched_closer_count': len(mismatches),
             'first_mismatched_closer_offset': mismatches[0] if mismatches else None,
             'unterminated_string': quoted}, ''.join(masked))


def inspect_text(text, *, schema_contract_verified=False, schema_only=False, schema_limits=None):
    """Describe original text; no recovered text is returned or persisted."""
    _, _, parser_errors = parse_model_output(text, 'audit')
    match = FENCE.search(text)
    candidate = match.group(1).strip() if match else text.strip()
    whole_candidate = candidate
    starts = [p for p in (candidate.find('{'), candidate.find('[')) if p >= 0]
    row = {'parser_errors': parser_errors, 'fenced': bool(match),
           'syntax_status': 'empty_response' if not candidate else 'no_json_start',
           'json_error': None, 'append_closers_probe_parses': False,
           'arithmetic_expression_outside_strings': False,
           'trailing_comma_outside_strings': False,
           'trailing_text_after_json': False}
    if starts:
        candidate = candidate[min(starts):]
        whole_candidate = whole_candidate if whole_candidate.startswith('"') else candidate
        try:
            obj, end = json.JSONDecoder().raw_decode(candidate)
            row['syntax_status'] = 'valid_object' if isinstance(obj, dict) else 'valid_non_object'
            row['trailing_text_after_json'] = bool(candidate[end:].strip())
            candidate = candidate[:end]
        except json.JSONDecodeError as exc:
            row['syntax_status'] = 'invalid_json'
            row['json_error'] = {'message': exc.msg, 'line': exc.lineno,
                                 'column': exc.colno, 'offset': exc.pos,
                                 'excerpt': candidate[max(0, exc.pos - 80):exc.pos + 80]}
    scan, masked = delimiter_scan(candidate)
    row['delimiter_scan'] = scan
    row['arithmetic_expression_outside_strings'] = bool(ARITHMETIC.search(masked))
    row['trailing_comma_outside_strings'] = bool(re.search(r',\s*[}\]]', masked))
    # Only probe a missing outer closure after an already closed container.
    # Never complete a string, number, key, expression, value or missing branch.
    if (row['syntax_status'] == 'invalid_json' and scan['unclosed_count']
            and not scan['unterminated_string'] and not scan['mismatched_closer_count']
            and candidate.rstrip().endswith(('}', ']'))):
        suffix = ''.join({'{': '}', '[': ']'}[c] for c in reversed(scan['unclosed_delimiters']))
        try:
            row['append_closers_probe_parses'] = isinstance(json.loads(candidate + suffix), dict)
        except (ValueError, RecursionError):
            pass
    needs_probe = (row['syntax_status'] in ('invalid_json', 'no_json_start')
                   or row['trailing_text_after_json'])
    row['delimiter_repair_probe'] = (
        delimiter_repair_probe(whole_candidate)
        if needs_probe and not schema_only
        else {'status': 'not_run' if needs_probe else 'not_needed',
              'candidate_count': 0, 'candidates': []})
    row['schema_repair_probe'] = (
        schema_repair_probe(whole_candidate, **(schema_limits or {}))
        if needs_probe and schema_contract_verified
        else {'status': 'contract_unverified' if needs_probe else 'not_needed',
              'candidate_count': 0, 'candidates': []})
    return row


def read_object(path, issues):
    try:
        obj = json.loads(path.read_text())
        if not isinstance(obj, dict):
            raise ValueError('expected JSON object')
        return obj
    except (OSError, ValueError) as exc:
        issues.append(f'{path.name}: {exc}')
        return {}


def audit_run(path, model, split, condition, frozen_model, *, schema_only=False, schema_limits=None):
    issues = []
    manifest = read_object(path / 'run_manifest.json', issues)
    meta = manifest.get('generation_metadata') or {}
    sampling = dict(frozen_model.get('sampling') or {})
    sampling.update(meta.get('sampling') or {})
    sampling.setdefault('temperature', manifest.get('temperature'))
    cap = manifest.get('max_output_tokens', frozen_model.get('max_output_tokens'))
    tokens = meta.get('output_tokens')
    finish = meta.get('finish_reason')
    numeric = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
    cap_hit = (str(finish).lower() in ('length', 'max_tokens') or
               (numeric(cap) and numeric(tokens) and tokens >= cap))
    row = {'model': model, 'split': split, 'condition': condition,
           'scenario': path.name, 'path': str(path), 'metadata_issues': issues,
           'recorded_parser_version': manifest.get('parser_version'),
           'model_revision': manifest.get('model_revision'),
           'sampling': sampling, 'temperature': sampling.get('temperature'),
           'presence_penalty': sampling.get('presence_penalty'),
           'seed': manifest.get('generation_seed', manifest.get('seed')),
           'finish_reason': finish, 'output_tokens': tokens,
           'reasoning_tokens': meta.get('reasoning_tokens', meta.get('thinking_tokens')),
           'output_cap': cap, 'output_cap_hit': bool(cap_hit)}
    contract_verified = False
    try:
        prompt = (path / 'system_prompt.txt').read_bytes()
        prompt_text = prompt.decode('utf-8')
        contract_verified = all(marker in prompt_text for marker in CONTRACT_MARKERS)
        row['schema_contract'] = {
            'status': 'verified' if contract_verified else 'unsupported_contract',
            'system_prompt_sha256': hashlib.sha256(prompt).hexdigest()}
    except (OSError, UnicodeError) as exc:
        row['schema_contract'] = {'status': 'unavailable', 'detail': str(exc)}
    try:
        raw = (path / 'final_response.txt').read_bytes()
        row['final_response_sha256'] = hashlib.sha256(raw).hexdigest()
        row.update(inspect_text(raw.decode('utf-8'), schema_contract_verified=contract_verified,
                                schema_only=schema_only, schema_limits=schema_limits))
    except (OSError, UnicodeError) as exc:
        row.update(syntax_status='missing_final_response', parser_errors=None)
        issues.append(f'final_response.txt: {exc}')
    return row


def summarize(rows):
    present = [r for r in rows if r['syntax_status'] != 'missing_final_response']
    failed = [r for r in present if r['parser_errors']]
    invalid = [r for r in present if r['syntax_status'] == 'invalid_json']
    probes = [r['delimiter_repair_probe'] for r in present]
    found = [p for p in probes if p['candidate_count']]
    schema_probes = [r['schema_repair_probe'] for r in invalid]
    return {'observed_requests': len(rows), 'responses_read': len(present),
            'missing_final_response': len(rows) - len(present),
            'parser_failures': len(failed),
            'parser_failure_rate': len(failed) / len(present) if present else None,
            'syntax_invalid': len(invalid),
            'syntax_status_counts': dict(Counter(r['syntax_status'] for r in rows)),
            'first_json_error_counts': dict(Counter(r['json_error']['message'] for r in invalid)),
            'unclosed_delimiters_no_mismatch': sum(bool(r['delimiter_scan']['unclosed_count'])
                and not r['delimiter_scan']['mismatched_closer_count'] for r in invalid),
            'mismatched_closers': sum(bool(r['delimiter_scan']['mismatched_closer_count']) for r in invalid),
            'unterminated_strings': sum(r['delimiter_scan']['unterminated_string'] for r in invalid),
            'append_closers_probe_parses': sum(r['append_closers_probe_parses'] for r in invalid),
            'delimiter_probe_status_counts': dict(Counter(p['status'] for p in probes)),
            'delimiter_probe_any_candidate': len(found),
            'delimiter_probe_candidates_in_invalid_json': sum(
                bool(r['delimiter_repair_probe']['candidate_count']) for r in invalid),
            'delimiter_probe_candidates_in_parser_failures': sum(
                bool(r['delimiter_repair_probe']['candidate_count']) for r in failed),
            'delimiter_probe_additional_to_eof_probe': sum(
                bool(r['delimiter_repair_probe']['candidate_count'])
                and not r['append_closers_probe_parses'] for r in invalid),
            'delimiter_probe_internal_candidate': sum(
                any(c['internal_insertions'] for c in p['candidates']) for p in found),
            'delimiter_probe_eof_only_candidate': sum(
                any(c['eof_insertions'] and not c['internal_insertions'] for c in p['candidates']) for p in found),
            'schema_probe_invalid_status_counts': dict(Counter(p['status'] for p in schema_probes)),
            'schema_probe_candidates_in_invalid_json': sum(bool(p['candidate_count']) for p in schema_probes),
            'schema_probe_single_complete': sum(p['status'] == 'candidate_found' for p in schema_probes),
            'schema_probe_ambiguous': sum(p['status'] == 'ambiguous' for p in schema_probes),
            'schema_probe_limited_with_candidates': sum(
                p['status'] == 'search_limit' and bool(p['candidate_count']) for p in schema_probes),
            'schema_probe_candidates_with_missing_fields': sum(
                any(c['missing_contract_fields'] for c in p['candidates']) for p in schema_probes),
            'schema_probe_candidates_at_scalar_eof': sum(
                bool(p['candidate_count']) and p.get('eof_scalar_preserved', False) for p in schema_probes),
            'schema_probe_candidates_in_parser_failures': sum(
                bool(r['schema_repair_probe']['candidate_count']) for r in failed),
            'schema_probe_total_candidate_responses': sum(
                bool(r['schema_repair_probe']['candidate_count']) for r in present),
            'combined_probe_candidates_in_invalid_json': sum(
                bool(r['schema_repair_probe']['candidate_count'] or
                     r['delimiter_repair_probe']['candidate_count']) for r in invalid),
            'schema_additional_to_syntax_probe': sum(
                bool(r['schema_repair_probe']['candidate_count']) and
                not r['delimiter_repair_probe']['candidate_count'] for r in invalid),
            'syntax_only_candidates': sum(
                bool(r['delimiter_repair_probe']['candidate_count']) and
                not r['schema_repair_probe']['candidate_count'] for r in invalid),
            'arithmetic_in_invalid_json': sum(r['arithmetic_expression_outside_strings'] for r in invalid),
            'trailing_comma_in_invalid_json': sum(r['trailing_comma_outside_strings'] for r in invalid),
            'output_cap_hits': sum(r['output_cap_hit'] for r in rows),
            'metadata_issue_requests': sum(bool(r['metadata_issues']) for r in rows),
            'sampling_variants': [json.loads(s) for s in sorted({json.dumps(r['sampling'], sort_keys=True) for r in rows})]}


def build_report(root, models, *, progress=None, schema_only=False, schema_limits=None):
    runs, cells, issues = [], [], []
    for model in models:
        base = root / model
        frozen_issues = []
        frozen = read_object(base / 'frozen.json', frozen_issues)
        issues.extend(f'{model}: {x}' for x in frozen_issues)
        inventory = frozen.get('datasets') or {}
        splits = sorted(set(inventory) | {p.name for p in (base / 'cells').glob('*') if p.is_dir()})
        for split in splits:
            expected = {r['name'] for r in inventory.get(split, [])}
            for condition in CONDITIONS:
                cell = base / 'cells' / split / f'{model}_{condition}'
                paths = sorted(p for p in cell.glob('scenario_*') if p.is_dir())
                rows = [audit_run(p, model, split, condition, frozen.get('model') or {},
                                  schema_only=schema_only, schema_limits=schema_limits) for p in paths]
                names = {p.name for p in paths}
                cells.append(dict(model=model, split=split, condition=condition,
                    expected_requests=len(expected) if split in inventory else None,
                    missing_scenarios=sorted(expected - names),
                    unexpected_scenarios=sorted(names - expected) if split in inventory else [],
                    **summarize(rows)))
                runs.extend(rows)
                if progress:
                    progress(f'{model} / {split} / {condition}: {len(rows)} responses audited')
    return {'audit_version': AUDIT_VERSION, 'current_parser_version': PARSER_VERSION,
            'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'structural_schema': structure_provenance(), 'schema_only': schema_only,
            'created_at': datetime.now(timezone.utc).isoformat(), 'root': str(root),
            'notes': ['Flags overlap; first-error counts are not a complete inventory of defects.',
                      'The append-closers probe does not establish semantic completeness or repairability.',
                      'Delimiter repair is bounded, error-guided, insertion-only and uses no reference answers.',
                      'candidate_found means one structure found at the lowest successful search depth, not proven uniqueness or intended meaning.',
                      'ambiguous means multiple structures found at that depth; search_limit may still contain candidates. Inspect search_complete.',
                      'Only braces/brackets are inserted: no keys, values, commas, quotes or arithmetic are repaired. Complete EOF scalars retain exactly their emitted digits/content.',
                      'The schema probe additionally preserves complete scalar tokens at EOF and verifies property ownership and container shapes against the public contract.',
                      'Schema candidates minimize delimiter insertions across all token boundaries. Search bounds and incomplete searches are reported explicitly.',
                      'Schema recovery never uses required-field completeness, enums, numeric bounds, future references or scores to select candidates. Missing fields remain missing.',
                      'Candidate insertion offsets refer to the extracted JSON text, not the raw response; repaired answers are not saved or scored.',
                      'Unknown sampling parameters remain null/missing; no default is inferred.',
                      'Sampling association is not evidence that presence_penalty caused the errors.',
                      'Parser errors are recomputed in memory with the current parser; stored outputs and scores are untouched.'],
            'issues': issues, 'models': {m: summarize([r for r in runs if r['model'] == m]) for m in models},
            'cells': cells, 'runs': runs}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=ROOT / 'results/api_panel')
    ap.add_argument('--model', action='append', help='Exact configuration directory; repeatable. Default: qwen*')
    ap.add_argument('--out', type=Path, help='Optional NEW JSON report outside the campaign directory')
    ap.add_argument('--schema-only', action='store_true', help='Skip legacy syntax search; retain EOF comparison')
    ap.add_argument('--schema-max-edits', type=int, default=64)
    ap.add_argument('--schema-max-states', type=int, default=2_000_000)
    ap.add_argument('--schema-max-paths', type=int, default=32)
    args = ap.parse_args(argv)
    schema_limits = dict(max_edits=args.schema_max_edits, max_states=args.schema_max_states,
                         max_paths=args.schema_max_paths)
    if any(value < 1 for value in schema_limits.values()):
        ap.error('Schema search limits must be positive integers')
    root = args.root.resolve()
    if not root.is_dir():
        ap.error(f'No campaign directory: {root}')
    models = args.model or sorted(p.name for p in root.glob('qwen*') if p.is_dir())
    if not models:
        ap.error('No qwen* configurations found; use --model NAME')
    for model in models:
        if Path(model).name != model or not (root / model).is_dir():
            ap.error(f'Invalid configuration directory: {model}')
    if args.out:
        out = args.out.resolve()
        if out == root or root in out.parents or out.exists():
            ap.error('--out must be a new file outside the campaign directory')
        if not out.parent.is_dir():
            ap.error('--out parent directory does not exist')
    report = build_report(root, list(dict.fromkeys(models)),
                          progress=lambda message: print(message, file=sys.stderr, flush=True),
                          schema_only=args.schema_only, schema_limits=schema_limits)
    for model, s in report['models'].items():
        print(f"\n{model}: {s['responses_read']}/{s['observed_requests']} responses read")
        print(f"  Parser failures={s['parser_failures']}; invalid JSON={s['syntax_invalid']}; cap hits={s['output_cap_hits']}")
        print(f"  Unclosed/no mismatch={s['unclosed_delimiters_no_mismatch']}; mismatched closers={s['mismatched_closers']}; unterminated strings={s['unterminated_strings']}")
        print(f"  Appending only outer closers parses={s['append_closers_probe_parses']}; arithmetic={s['arithmetic_in_invalid_json']}; trailing commas={s['trailing_comma_in_invalid_json']}")
        print('  Delimiter insertion probe:', json.dumps(s['delimiter_probe_status_counts'], sort_keys=True))
        print(f"  Candidate found (including ambiguous/limited)={s['delimiter_probe_any_candidate']}; internal={s['delimiter_probe_internal_candidate']}; EOF-only={s['delimiter_probe_eof_only_candidate']}")
        print(f"  Candidates among invalid JSON={s['delimiter_probe_candidates_in_invalid_json']}; additional to EOF probe={s['delimiter_probe_additional_to_eof_probe']}")
        print('  Schema probe (invalid JSON only):', json.dumps(s['schema_probe_invalid_status_counts'], sort_keys=True))
        print(f"  Schema candidates={s['schema_probe_candidates_in_invalid_json']}; single/completed={s['schema_probe_single_complete']}; ambiguous={s['schema_probe_ambiguous']}; limited/with candidate={s['schema_probe_limited_with_candidates']}")
        print(f"  Either probe finds candidate={s['combined_probe_candidates_in_invalid_json']}; schema adds={s['schema_additional_to_syntax_probe']}; syntax-only={s['syntax_only_candidates']}")
        print('  Recorded sampling:', json.dumps(s['sampling_variants'], sort_keys=True))
    print('\nCells: model / split / condition | present/expected | parser errors | invalid JSON')
    for c in report['cells']:
        print(f"  {c['model']} / {c['split']} / {c['condition']} | {c['observed_requests']}/{c['expected_requests']} | {c['parser_failures']} | {c['syntax_invalid']}")
    for issue in report['issues']:
        print('WARNING:', issue, file=sys.stderr)
    print('\nFlags overlap. These counts do not isolate the effect of presence_penalty. No input files were changed.')
    if args.out:
        with out.open('x', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write('\n')
        print(f'Detailed report: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
