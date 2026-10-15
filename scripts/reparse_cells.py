"""Re-parse every cell's saved answers with the current parser, then re-score.

Parser policy: one parser for every configuration and cell. After
a parser change, run this over the whole results root so no cell keeps
predictions from an older parser:

  .venv/bin/python scripts/reparse_cells.py --root results/api_panel --dry-run
  .venv/bin/python scripts/reparse_cells.py --root results/api_panel

For each scenario directory that holds `final_response.txt` the parser is
re-run and `l3_pre.json` / `l4_pre.json` are rewritten when they differ; the
model's text, prompts, manifests and metadata are never touched. A cell whose
predictions changed is re-scored with `score_cell` (report, coherence, L4 pair,
provenance) and collected again, as is a cell whose scores are stale (its
provenance no longer matches its prediction files, or an older scorer, L4
protocol, coherence or L4-pair version); a baseline cell has no saved answer
and is only re-scored when stale. Each root's tables are re-exported. A root that came back from
another machine without `inputs/` gets it relinked from the panel's dataset
directories and checked against the frozen fingerprints first. A cell with a
scenario that has no saved answer is reported and skipped, never partially
rewritten. `--score-all` re-scores unchanged cells too.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

_here = Path(__file__).resolve()
_root = next((p for p in _here.parents if (p / "analysis").is_dir()), _here.parent)
sys.path.insert(0, str(_root))

from analysis.parse_prediction import parse_model_output  # noqa: E402


def reparse_cell(cell_dir: Path, rows: List[Dict[str, Any]], *, write: bool) -> Dict[str, Any]:
    """Re-parse one cell. Returns counts; writes only when `write` and complete."""
    out = dict(cell=str(cell_dir), scenarios=len(rows), missing=[], changed=[],
               unchanged=0, parse_failures=0, written=False)
    plans = []
    for row in rows:
        d = cell_dir / row["name"]
        if not (d / "final_response.txt").is_file() or not (d / "run_manifest.json").is_file():
            out["missing"].append(row["name"])
            continue
        text = (d / "final_response.txt").read_text(encoding="utf-8")
        l3, l4, errors = parse_model_output(text, row["scenario_id"])
        out["parse_failures"] += int(bool(errors))
        old = []
        for name in ("l3_pre.json", "l4_pre.json"):
            try:
                old.append(json.loads((d / name).read_text(encoding="utf-8")))
            except (OSError, ValueError):
                old.append(None)
        if old == [l3, l4]:
            out["unchanged"] += 1
        else:
            out["changed"].append(row["name"])
            plans.append((d, l3, l4))
    if out["missing"] or not write:
        return out
    for d, l3, l4 in plans:
        with open(d / "l3_pre.json", "w") as f:
            json.dump(l3, f, indent=2)
        with open(d / "l4_pre.json", "w") as f:
            json.dump(l4, f, indent=2)
    out["written"] = True
    return out


def cell_spec(root: Path, cell_dir: Path, split: str, frozen: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild the cell dict `run_cell` / the Gemini fetch used, from a manifest."""
    first = next(p for p in sorted(cell_dir.glob("scenario_*")) if (p / "run_manifest.json").is_file())
    m = json.loads((first / "run_manifest.json").read_text(encoding="utf-8"))
    cell = dict(split=split, backend=m.get("backend"), condition=m.get("condition"),
                configuration=m.get("configuration_name"), out_dir=str(cell_dir))
    sampling = (frozen.get("model") or {}).get("sampling")
    if sampling:
        cell["sampling"] = sampling
    return cell


def split_inputs(root: Path, split: str, frozen: Dict[str, Any]) -> Path:
    """The split's scenario root for scoring: `inputs/<split>` of the root,
    recreated from the panel's dataset directories when a root came back
    from another machine without it, and checked row by row against the
    frozen inventory fingerprints before any score is computed."""
    from analysis.evaluation_campaign import dataset_inventory
    from scripts.run_api_panel import choose, link
    subset = root / "inputs" / split
    rows = frozen["datasets"][split]
    if not subset.is_dir():
        chosen = [d for d in choose(split, None) if d.name in {r["name"] for r in rows}]
        link(subset, chosen)
    names = sorted(p.name for p in subset.glob("scenario_*") if p.is_dir())
    if names != sorted(r["name"] for r in rows):
        raise ValueError(f"{subset}: scenario set differs from frozen.json")
    current = {r["name"]: r["fingerprint"] for r in dataset_inventory(subset, len(rows), split)}
    frozen_fp = {r["name"]: r["fingerprint"] for r in rows}
    if current != frozen_fp:
        bad = sorted(n for n in frozen_fp if current.get(n) != frozen_fp[n])
        raise ValueError(f"{subset}: {len(bad)} scenario(s) differ from the frozen inputs: {bad[:5]}")
    return subset


def stale_score(cell_dir: Path, rows: List[Dict[str, Any]]) -> Optional[str]:
    """Why the cell's scores no longer describe its prediction files, or None."""
    from analysis.causal_coherence import COHERENCE_VERSION
    from analysis.evaluation_campaign import sha
    from analysis.intervention_pair import L4_PAIR_VERSION
    from ctms.ctms import L4_PAIR_PROTOCOL, SCORER_VERSION
    try:
        proof = json.loads((cell_dir / "score_provenance.json").read_text(encoding="utf-8"))
        report = json.loads((cell_dir / "report.json").read_text(encoding="utf-8"))
        coherence = json.loads((cell_dir / "coherence.json").read_text(encoding="utf-8"))
        l4_pair = json.loads((cell_dir / "l4_pair.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"unreadable score artifacts: {exc}"
    if report.get("scorer_version") != SCORER_VERSION or report.get("l4_pair_protocol") != L4_PAIR_PROTOCOL:
        return "report scorer/L4 protocol version differs"
    if coherence.get("coherence_version") != COHERENCE_VERSION:
        return f"coherence {coherence.get('coherence_version')} differs from {COHERENCE_VERSION}"
    if l4_pair.get("l4_pair_version") != L4_PAIR_VERSION:
        return f"l4 pair {l4_pair.get('l4_pair_version')} differs from {L4_PAIR_VERSION}"
    files = proof.get("prediction_files") or {}
    for row in rows:
        for name in ("l3_pre.json", "l4_pre.json", "run_manifest.json"):
            rel = f"{row['name']}/{name}"
            if files.get(rel) != sha(cell_dir / row["name"] / name):
                return f"{rel} differs from the scored file"
    return None


def rescore(root: Path, cell: Dict[str, Any], frozen: Dict[str, Any]) -> Dict[str, Any]:
    from analysis.evaluation_campaign import score_cell, sha, write_json
    from analysis.evaluation_report import collect_cell
    campaign = {"campaign_dir": str(root),
                "splits": {s: {"path": str(split_inputs(root, s, frozen))} for s in frozen["datasets"]}}
    score_cell(campaign, cell, frozen)
    split = cell["split"]
    data = collect_cell(cell, frozen["datasets"][split], split_path=campaign["splits"][split]["path"],
                        frozen_sha256=sha(root / "frozen.json"))
    write_json(Path(cell["out_dir"]) / "collected.json", data)
    return data


def export_root_tables(root: Path, frozen: Dict[str, Any]) -> None:
    from analysis.evaluation_report import export_tables, mean_stat
    records = []
    for collected in sorted(root.glob("cells/*/*/collected.json")):
        data = json.loads(collected.read_text(encoding="utf-8"))
        parts = collected.parent.name.rsplit("_", 1)
        name = json.loads((collected.parent / "campaign_identity.json").read_text()).get("configuration") \
            if (collected.parent / "campaign_identity.json").exists() else None
        first = next((p for p in sorted(collected.parent.glob("scenario_*")) if (p / "run_manifest.json").is_file()), None)
        m = json.loads((first / "run_manifest.json").read_text()) if first else {}
        records += [dict(split=collected.parent.parent.name, configuration=m.get("configuration_name", name or parts[0]),
                         condition=m.get("condition", parts[-1]), comparison="scores", metric=k, **mean_stat(list(v.values())))
                    for k, v in data["metrics"].items()]
    if records:
        export_tables(root, records)


def find_roots(base: Path) -> List[Path]:
    return sorted(p.parent for p in base.rglob("frozen.json") if (p.parent / "cells").is_dir())


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=_root / "results" / "api_panel")
    ap.add_argument("--dry-run", action="store_true", help="Report what would change; write nothing")
    ap.add_argument("--no-score", action="store_true", help="Rewrite predictions but do not re-score")
    ap.add_argument("--score-all", action="store_true", help="Re-score every complete cell, changed or not")
    args = ap.parse_args(argv)
    base = args.root.resolve()
    if not base.is_dir():
        ap.error(f"no such directory: {base}")
    total_changed = total_failures = 0
    for root in find_roots(base):
        frozen = json.loads((root / "frozen.json").read_text(encoding="utf-8"))
        touched = False
        for split, rows in (frozen.get("datasets") or {}).items():
            for cell_dir in sorted((root / "cells" / split).glob("*")) if (root / "cells" / split).is_dir() else []:
                if not cell_dir.is_dir() or not any(cell_dir.glob("scenario_*")):
                    continue
                rel = cell_dir.relative_to(base)
                if not any((cell_dir / r["name"] / "final_response.txt").is_file() for r in rows):
                    # A baseline cell: predictions come from a rule, not a
                    # saved answer. Nothing to re-parse; it is still re-scored
                    # when its scores are stale.
                    res = dict(changed=[], unchanged=len(rows), parse_failures=0, missing=[])
                    print(f"{rel}: baseline cell, nothing to re-parse")
                else:
                    res = reparse_cell(cell_dir, rows, write=not args.dry_run)
                    if res["missing"]:
                        print(f"{rel}: {len(res['missing'])} scenario(s) without a saved answer; skipped")
                        continue
                    print(f"{rel}: {len(res['changed'])} changed, {res['unchanged']} unchanged, "
                          f"{res['parse_failures']} still unparsed" + (" (dry run)" if args.dry_run else ""))
                total_changed += len(res["changed"]); total_failures += res["parse_failures"]
                if args.dry_run or args.no_score:
                    continue
                stale = stale_score(cell_dir, rows)
                if not (res["changed"] or stale or args.score_all):
                    continue
                rescore(root, cell_spec(root, cell_dir, split, frozen), frozen)
                touched = True
                print(f"{rel}: re-scored" + (f" ({stale})" if stale and not res["changed"] else ""))
        if touched:
            export_root_tables(root, frozen)
    print(f"total: {total_changed} prediction file pair(s) changed, {total_failures} answer(s) still unparsed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
