"""Rebuild CARLA reference chains offline from the retained raw telemetry.

Every accepted CARLA scenario directory keeps `raw_telemetry.json` (all
0.1 s ticks of every rollout) and `scenario_config.json` (the exact config
and attempt seed). When a reference rule changes (terminal semantics, role
labelling) the `ground_truth.json` can therefore be re-derived without
re-running the simulator. This script rebuilds each scenario with the current
`simulation.build_ground_truth`, reports what would change, and rewrites the
file only with `--write`. Run the dataset verifier afterwards.

    .venv/bin/python scripts/rebuild_carla_gt_from_raw.py --root <dataset_dir>
    .venv/bin/python scripts/rebuild_carla_gt_from_raw.py --root <dataset_dir> --write
"""
import argparse
import copy
import json
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from simulation.build_ground_truth import build_scenario_gt  # noqa: E402

# Keys of the protocol block that record the build, not the reference.
VOLATILE_PROTOCOL_KEYS = {"build_code_commit", "build_worktree_clean", "built_at"}


def _streams(gt: Dict[str, Any]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    out = [(str(b.get("branch_id")), b.get("chain") or []) for b in gt.get("branches", [])]
    for i, q in enumerate(gt.get("l4_queries", []) or []):
        out.append((f"L4[{i}]:{q.get('intervention_type')}",
                    (q.get("counterfactual_chain") or {}).get("chain") or []))
    return out


def _summary(chain: List[Dict[str, Any]]) -> Tuple[int, str, bool]:
    if not chain:
        return 0, "empty", False
    last = chain[-1]
    return len(chain), str(last.get("event_type")), bool(last.get("terminal"))


def _strip_volatile(gt: Dict[str, Any]) -> Dict[str, Any]:
    g = copy.deepcopy(gt)
    g.pop("_observed_history", None)
    proto = g.get("protocol")
    if isinstance(proto, dict):
        for k in VOLATILE_PROTOCOL_KEYS:
            proto.pop(k, None)
    return g


def rebuild_one(scen_dir: str) -> Dict[str, Any]:
    raw = json.load(open(os.path.join(scen_dir, "raw_telemetry.json")))
    cfg = json.load(open(os.path.join(scen_dir, "scenario_config.json")))
    old = json.load(open(os.path.join(scen_dir, "ground_truth.json")))
    track = ((old.get("protocol") or {}).get("track")) or "primary"
    new = build_scenario_gt(raw, cfg, track=track, protocol=True)
    new.pop("_observed_history", None)
    # Carry the build provenance the batch runner stamped on the old file.
    if isinstance(old.get("protocol"), dict) and isinstance(new.get("protocol"), dict):
        for k in VOLATILE_PROTOCOL_KEYS:
            if k in old["protocol"] and k not in new["protocol"]:
                new["protocol"][k] = old["protocol"][k]
    changes = []
    old_streams = dict(_streams(old))
    for name, chain in _streams(new):
        before = _summary(old_streams.get(name, []))
        after = _summary(chain)
        if before != after or old_streams.get(name) != chain:
            changes.append({"stream": name, "before": before, "after": after,
                            "chain_equal": old_streams.get(name) == chain})
    other_diff = [k for k in set(old) | set(new)
                  if k not in ("branches", "counterfactuals", "l4_queries")
                  and _strip_volatile({k: old.get(k)}) != _strip_volatile({k: new.get(k)})]
    return {"dir": scen_dir, "new": new, "changes": changes, "other_diff": sorted(other_diff),
            "identical": _strip_volatile(old) == _strip_volatile(new)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dataset directory with scenario subdirectories")
    ap.add_argument("--write", action="store_true", help="overwrite ground_truth.json with the rebuilt reference")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", help="write a JSON report of every change here")
    ap.add_argument("--qc", action="store_true",
                    help="run the current dataset QC (all scorer gates, root-seeded kinematic "
                         "check, protocol grid) on every rebuilt reference; any failure blocks --write")
    args = ap.parse_args()
    qc_env = None
    if args.qc:
        from ctms.ctms import load_ctms_environment
        from simulation.qc import qc_scenario_gt
        here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ctms")
        qc_env = load_ctms_environment(*(os.path.join(here, f) for f in (
            "ontology.yaml", "ctms_config.yaml", "schema_branch.json", "schema_counterfactual.json")))
    # Every scenario directory of the release is in scope; one that lacks any
    # of the three inputs is an error, not a skip (hidden dirs such as
    # `.rejected` diagnostics and `.carla_supervisor` are not scenarios).
    dirs = sorted(os.path.join(args.root, n) for n in os.listdir(args.root)
                  if n.startswith("scenario_") and os.path.isdir(os.path.join(args.root, n)))
    if args.limit:
        dirs = dirs[:args.limit]
    if not dirs:
        print(f"INCOMPLETE: no scenario_* directories under {args.root}", file=sys.stderr)
        return 1
    # Rebuild everything first; write nothing until every scenario rebuilt,
    # so a failure cannot leave a dataset that mixes old and new references.
    results = []
    errors = []
    report = []
    kinds: Dict[str, int] = {}
    for d in dirs:
        missing = [f for f in ("raw_telemetry.json", "scenario_config.json", "ground_truth.json")
                   if not os.path.isfile(os.path.join(d, f))]
        if missing:
            print(f"ERROR {d}: missing {missing}")
            errors.append(d)
            report.append({"dir": d, "error": f"missing {missing}"})
            continue
        try:
            r = rebuild_one(d)
        except Exception as exc:  # report every failure, then refuse to write
            print(f"ERROR {d}: {type(exc).__name__}: {exc}")
            errors.append(d)
            report.append({"dir": d, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if qc_env is not None:
            ok, reasons = qc_scenario_gt(r["new"], qc_env["ontology"], qc_env["config"], qc_env["branch_schema"])
            if not ok:
                print(f"ERROR {d}: QC rejects the rebuilt reference: {reasons[:1]}")
                errors.append(d)
                report.append({"dir": d, "error": "qc_rejected", "reasons": reasons})
                continue
        results.append(r)
        if not r["identical"]:
            for c in r["changes"]:
                key = f"{c['before'][1]}@{c['before'][0]} -> {c['after'][1]}@{c['after'][0]}"
                kinds[key] = kinds.get(key, 0) + 1
            report.append({k: v for k, v in r.items() if k != "new"})
    changed = [r for r in results if not r["identical"]]
    written = 0
    if args.write and not errors:
        for r in changed:
            target = os.path.join(r["dir"], "ground_truth.json")
            tmp = target + ".rebuild.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(r["new"], f, indent=2)
            os.replace(tmp, target)  # atomic per file
            written += 1
    print(f"scenarios={len(dirs)} rebuilt={len(results)} errors={len(errors)} "
          f"changed={len(changed)} written={written}")
    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"  {v:4d}  {k}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    if errors:
        print(f"INCOMPLETE: {len(errors)} scenario(s) failed to rebuild; nothing was written",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
