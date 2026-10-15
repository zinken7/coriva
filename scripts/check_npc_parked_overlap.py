"""Orientation-free pass-through check for a CARLA release: does any moving
non-ego actor come closer to a parked vehicle than the sum of their
half-widths at the same reference node? Two car bodies that close overlap
whatever their headings, so such a node means the NPC drove through the
parked car (NPC physics are off and the detector only checks the ego).

    "$CORIVA_PYTHON" scripts/check_npc_parked_overlap.py --root "$CARLA_BUILD"
"""
import argparse, glob, json, math, os, sys

STATIC = {"parked_vehicle", "guardrail", "curb", "barrier", "traffic_cone", "road_edge", "traffic_sign", "pole"}


def scan(root):
    """Prefer the raw telemetry with the release gate itself
    (`static_pass_through`: oriented footprints, every tick and mid-tick, all
    L3/L4 rollouts); fall back to the released GT nodes with a centre-distance
    test when raw is absent."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from simulation.build_ground_truth import static_pass_through
    files = sorted(glob.glob(os.path.join(root, "scenario_*", "ground_truth.json")))
    scenes, pairs, rows = set(), {}, []
    for p in files:
        raw_path = os.path.join(os.path.dirname(p), "raw_telemetry.json")
        cfg_path = os.path.join(os.path.dirname(p), "scenario_config.json")
        if os.path.isfile(raw_path) and os.path.isfile(cfg_path):
            raw = json.load(open(raw_path)); cfg = json.load(open(cfg_path))
            for h in static_pass_through(raw, cfg.get("actor_defs", {})):
                scene = os.path.basename(os.path.dirname(p))
                scenes.add(scene)
                pairs[(h["mover"], h["static"])] = pairs.get((h["mover"], h["static"]), 0) + 1
                rows.append(dict(scene=scene, stream=h["branch"], time_sec=h["time_sec"],
                                 mover=h["mover"], parked=h["static"], distance_m=h["distance_m"]))
            continue
        gt = json.load(open(p))
        base = {a["id"]: a for a in gt["observed_state"]["actors"]}
        streams = [(b["branch_id"], b["chain"]) for b in gt["branches"]]
        streams += [(f"L4_{q.get('intervention_type')}", q["counterfactual_chain"]["chain"]) for q in gt.get("l4_queries", [])]
        for name, chain in streams:
            w = dict(base)
            for node in chain:
                for a in node["actors_update"]:
                    w[a["id"]] = a
                for mid, m in w.items():
                    if mid == "ego" or base.get(mid, {}).get("entity_type") in STATIC:
                        continue
                    for sid, s in w.items():
                        if base.get(sid, {}).get("entity_type") != "parked_vehicle":
                            continue
                        d = math.hypot(m["x"] - s["x"], m["y"] - s["y"])
                        if d < (m.get("width", 2.0) + s.get("width", 2.0)) / 2:
                            scene = os.path.basename(os.path.dirname(p))
                            scenes.add(scene)
                            pairs[(mid, sid)] = pairs.get((mid, sid), 0) + 1
                            rows.append(dict(scene=scene, stream=name, depth=node.get("depth"), mover=mid, parked=sid, distance_m=round(d, 2)))
    return files, scenes, pairs, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--report", help="write every hit as JSON here")
    args = ap.parse_args()
    files, scenes, pairs, rows = scan(args.root)
    print(f"scenarios={len(files)} scenes_with_npc_through_parked_vehicle={len(scenes)} hits={len(rows)}")
    for (m, s), n in sorted(pairs.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {m} -> {s}")
    if args.report:
        json.dump(rows, open(args.report, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
