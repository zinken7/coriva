"""Print the diagnostic fields that decide whether a prompt change worked.

Usage: python scripts/inspect_cell.py <cell_dir> [<cell_dir> ...]
A cell dir holds report.json and pred.json (a campaign matrix cell or a
single-cell diagnostic cell).
"""
import json
import math
import sys
from pathlib import Path


def inspect(cell: Path) -> None:
    if not (cell / "report.json").is_file():
        print(f"== {cell}\n   (no report.json — cell incomplete or interrupted)")
        return
    rep = json.loads((cell / "report.json").read_text())
    pred = json.loads((cell / "pred.json").read_text())
    s = rep["summary"]
    l3 = rep["per_sample"][0]["layer3"]
    br_c = s["compliance"]["branches"]
    cf_c = s["compliance"]["counterfactuals"]
    sample = pred["scenarios"][0]["samples"][0]["prediction"]
    branches = sample.get("branches") or sample.get("predicted_branches") or []
    cf = (sample.get("predicted_counterfactuals") or [{}])[0]

    ego = []
    chains = [b.get("chain") for b in branches]
    chains += [cf.get("control_chain"), cf.get("treated_chain")]
    for chain in chains:
        if chain:
            for a in chain[0].get("actors_update") or []:
                if a.get("id") == "ego":
                    ego.append((a.get("x"), a.get("vx"), a.get("speed")))

    print(f"== {cell}")
    print(f"   branches {len(branches)}/{l3['diagnostics']['gt_count']}  "
          f"chain lengths {[len(b.get('chain') or []) for b in branches]}  "
          f"CF pair ({len(cf.get('control_chain') or [])},{len(cf.get('treated_chain') or [])})")
    print(f"   actors in node[0] {len(branches[0].get('chain', [{}])[0].get('actors_update', []))
                              if branches and branches[0].get('chain') else 0}"
          f"   ego node[0] {sorted(set(ego))}")
    old_ex = [h for h in ego if h == (4.0, 8.0, 8.0) or h[0] == 4.0]
    print(f"   COPIED old example (4.0/8.0/8.0): {'YES — ' + str(sorted(set(old_ex))) if old_ex else 'no'}")

    # Frozen-actor check: node[0] must have MOVED from the observed T=0 state.
    # Comparing against x == 0.0 is meaningless: CARLA's ego starts at x = 0.0.
    gt = json.loads((cell / "gt.json").read_text())
    obs = (gt.get("scenarios", [{}])[0].get("samples", [{}])[0]
             .get("ground_truth", {}).get("observed_state") or {})
    root = {str(a.get("id")): a for a in obs.get("actors", [])}
    frozen, dt = [], 0.5
    for chain in chains:
        if not chain:
            continue
        for a in chain[0].get("actors_update") or []:
            r = root.get(str(a.get("id")))
            if not r:
                continue
            moved = math.dist((a.get("x", 0), a.get("y", 0)), (r.get("x", 0), r.get("y", 0)))
            declared = a.get("speed")
            if declared is not None and abs(declared - moved / dt) > 4.0:
                frozen.append(f"{a['id']}(declared {declared:.1f} vs moved {moved / dt:.2f} m/s)")
    print(f"   ACTORS NOT PROPAGATED at node[0]: "
          f"{', '.join(sorted(set(frozen))[:4]) if frozen else 'none'}")
    print(f"   L3 valid {br_c['valid']}/{br_c['total_unique']}   "
          f"L4 pair_valid {cf_c['pair']['pair_valid']}   "
          f"failed gate: { {k: v for k, v in br_c['first_failure_counts'].items() if v} } "
          f"/ L4 { {k: v for k, v in cf_c['first_failure_counts'].items() if v} }")
    print(f"   F1={s['Causal_Foresight_F1_mean']:.4f}  "
          f"CAS={s['Counterfactual_Alignment_Score_mean']:.4f}  "
          f"CTMS={s['CTMS_mean']:.4f}")
    cfs = sample.get("predicted_counterfactuals") or []
    print(f"   CF objects: {len(cfs)}")
    for n, c in enumerate(cfs):
        shape = {k: (f"array[{len(v)}]" if isinstance(v, list)
                     else type(v).__name__) for k, v in c.items()}
        print(f"     CF[{n}] keys={shape}")
        interv = c.get("intervention") or {}
        print(f"            intervention={ {k: interv.get(k) for k in ('intervention_type', 'target_actor_id', 'trigger_depth')} }")
    for g in l3["diagnostics"].get("pred_gate_failures", [])[:1]:
        for cat in ("schema_msgs", "semantic_msgs", "kinematic_msgs", "metadata_msgs"):
            for m in (g.get(cat) or [])[:2]:
                print(f"     L3 {m[:118]}")
    for r in rep["per_sample"][0].get("layer4", {}).get("results", []):
        diag = r.get("diagnostics", {})
        print(f"   L4 reason={diag.get('reason')}")
        for chain in ("treated", "control"):
            for cat, msgs in (diag.get(chain) or {}).items():
                for m in (msgs or [])[:2] if isinstance(msgs, list) else []:
                    print(f"     L4 {chain}/{m[:110]}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        inspect(Path(arg))
