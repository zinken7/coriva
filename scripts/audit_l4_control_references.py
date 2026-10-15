"""Audit the L4 control reference of every accepted scene (Option B, step 1).

Option B turns the L4 query into a matched control/treated pair from the same
observed root under the same fixed ego policy. The treated chain is the
released ``l4_queries[i].counterfactual_chain``. The control chain (named
target on its baseline, nothing else changed) was run by both builders for
admission but never exported. This script decides, per scene, whether the
control reference is

* ``stored_exact``        — CARLA: the paired control rollout is retained in
                            ``raw_telemetry.json`` (``_l4_control_steps``) and
                            the current builder reproduces the released
                            treated chain from the retained raw bit-for-bit;
* ``reconstructed_exact`` — nuScenes: re-rolling the canonical twin from the
                            released root, ``twin_model``, the recorded
                            ``base_steer`` and the recorded resolved profile
                            reproduces the released treated chain bit-for-bit
                            AND the recorded per-step paired effect, so the
                            control produced by the same call is exact;
* anything else           — reported with reasons; the scene needs a decision.

Nothing is written unless ``--write`` is given; then every scene must be
exact, and the control chain is added under ``l4_queries[i].control_chain``
next to the (unchanged) treated ``counterfactual_chain``.

    .venv/bin/python scripts/audit_l4_control_references.py --split carla --root <dir> --report <json>
    .venv/bin/python scripts/audit_l4_control_references.py --split nuscenes --root <dir> --report <json>
"""
import argparse
import copy
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from simulation.build_ground_truth import (  # noqa: E402
    build_chain, build_observed_inputs, build_scenario_gt, scored_collision)
from simulation.kinematic_twin import (  # noqa: E402
    ENV_PROFILES, TWIN_VERSION, build_twin_branch, roll_branch)
from simulation.l4_control import (  # noqa: E402
    CONTROL_INTERVENTION_TYPE, CONTROL_SEMANTICS as _SHARED_SEMANTICS, control_query_entry,
    mark_pair_protocol)
from simulation.nuscenes_admission import observed_velocity_baseline  # noqa: E402
from simulation.protocol import track_steps  # noqa: E402

AUDIT_VERSION = "coriva-l4-control-audit-1.0"

# The control reference contract (`simulation/l4_control.py`) added to every
# ``l4_queries[i]`` entry by ``--write``; the treated chain keeps its key and
# content. The builders now write the same entry for new builds.
CONTROL_SEMANTICS = {split: {k: v for k, v in sem.items()} for split, sem in _SHARED_SEMANTICS.items()}
CONTROL_SEMANTICS["nuscenes"]["twin_version"] = TWIN_VERSION


def _round_trip(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _speed(state: Dict[str, Any]) -> float:
    if state.get("vx") is not None or state.get("vy") is not None:
        return math.hypot(float(state.get("vx") or 0.0), float(state.get("vy") or 0.0))
    return float(state.get("speed") or 0.0)


def _chain_actor(chain: List[Dict[str, Any]], depth: int, actor_id: str) -> Optional[Dict[str, Any]]:
    for node in chain:
        if int(node.get("depth", -1)) != depth:
            continue
        for actor in node.get("actors_update") or []:
            if str(actor.get("id")) == str(actor_id):
                return actor
    return None


def _summary(chain: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not chain:
        return {"nodes": 0, "last_event": None, "terminal": False}
    last = chain[-1]
    return {"nodes": len(chain), "last_event": last.get("event_type"),
            "terminal": bool(last.get("terminal"))}


def _pair_effect(control: List[Dict[str, Any]], treated: List[Dict[str, Any]],
                 target_id: str) -> Dict[str, Any]:
    """Max target position/speed difference over the depths both chains carry."""
    common = 0
    max_pos = 0.0
    max_speed = 0.0
    for depth in range(1, 7):
        c = _chain_actor(control, depth, target_id)
        t = _chain_actor(treated, depth, target_id)
        if c is None or t is None:
            break
        common += 1
        max_pos = max(max_pos, math.hypot(float(t["x"]) - float(c["x"]),
                                          float(t["y"]) - float(c["y"])))
        max_speed = max(max_speed, abs(_speed(t) - _speed(c)))
    return {"common_depths": common, "max_target_position_effect_m": round(max_pos, 3),
            "max_target_speed_effect_mps": round(max_speed, 3)}


def _control_target_drift(chain: List[Dict[str, Any]], root: Dict[str, Any],
                          target_id: str) -> Dict[str, Any]:
    """How far the control target departs from its T0 speed and lateral
    position (the baseline is constant velocity; on a curve y still moves)."""
    t0 = next((a for a in root.get("actors", []) if str(a.get("id")) == str(target_id)), None)
    if t0 is None:
        return {"target_in_root": False}
    speeds = []
    ys = []
    for depth in range(1, 7):
        a = _chain_actor(chain, depth, target_id)
        if a is None:
            break
        speeds.append(_speed(a))
        ys.append(float(a.get("y") or 0.0))
    if not speeds:
        return {"target_in_root": True, "depths": 0}
    return {"target_in_root": True, "depths": len(speeds),
            "max_abs_speed_change_mps": round(max(abs(s - _speed(t0)) for s in speeds), 3),
            "max_abs_lateral_change_m": round(max(abs(y - float(t0.get("y") or 0.0)) for y in ys), 3)}


def _compare_exported_control(query: Dict[str, Any], control_chain: List[Dict[str, Any]],
                              rec: Dict[str, Any]) -> None:
    """A reference that already carries `control_chain` (a build after the
    pair protocol) must hold exactly what this audit derives."""
    exported = query.get("control_chain") if isinstance(query, dict) else None
    rec["control_already_exported"] = isinstance(exported, dict)
    if isinstance(exported, dict):
        rec["control_reproduced"] = _round_trip(exported.get("chain") or []) == control_chain
        if not rec["control_reproduced"]:
            rec["reasons"].append("exported control_chain differs from the re-derived control")


def _control_query_entry(query: Dict[str, Any], chain: List[Dict[str, Any]], split: str) -> Dict[str, Any]:
    intervention = (query.get("counterfactual_chain") or {}).get("intervention") or {}
    extra = {"twin_version": TWIN_VERSION} if split == "nuscenes" else {}
    return control_query_entry(intervention, chain, split, **extra)


# ---------------------------------------------------------------- CARLA ----

def audit_carla_scene(scen_dir: str) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"dir": scen_dir, "split": "carla", "reasons": []}
    missing = [f for f in ("raw_telemetry.json", "scenario_config.json", "ground_truth.json")
               if not os.path.isfile(os.path.join(scen_dir, f))]
    if missing:
        rec.update(status="missing_inputs", reasons=[f"missing {missing}"])
        return rec
    raw = json.load(open(os.path.join(scen_dir, "raw_telemetry.json")))
    cfg = json.load(open(os.path.join(scen_dir, "scenario_config.json")))
    gt = json.load(open(os.path.join(scen_dir, "ground_truth.json")))
    rec["scenario_id"] = gt.get("scenario_id")
    proto = gt.get("protocol") or {}
    track = proto.get("track") or "primary"
    policy = proto.get("l4_ego_policy") or (cfg.get("l4_query") or {}).get("ego_policy") or "do_ego_no_action"
    rec["ego_policy"] = policy
    queries = gt.get("l4_queries") or []
    if len(queries) != 1:
        rec["reasons"].append(f"expected exactly one l4 query, found {len(queries)}")
    control_steps = raw.get("_l4_control_steps") or []
    treated_steps = raw.get("_l4_steps") or []
    rec["control_rollout_ticks"] = len(control_steps)
    rec["treated_rollout_ticks"] = len(treated_steps)
    if not control_steps:
        rec["reasons"].append("raw_telemetry has no _l4_control_steps")
    if not treated_steps:
        rec["reasons"].append("raw_telemetry has no _l4_steps")
    if rec["reasons"]:
        rec["status"] = "not_recoverable"
        return rec

    # Determinism: the current builder must reproduce the released reference
    # from the retained raw before its control export can be trusted.
    rebuilt = build_scenario_gt(raw, cfg, track=track, protocol=True)
    rebuilt.pop("_observed_history", None)
    stored_treated = _round_trip((queries[0].get("counterfactual_chain") or {}).get("chain") or [])
    rebuilt_treated = _round_trip((rebuilt.get("l4_queries") or [{}])[0].get("counterfactual_chain", {}).get("chain") or [])
    rec["treated_reproduced"] = stored_treated == rebuilt_treated
    stored_branches = {b["branch_id"]: _round_trip(b["chain"]) for b in gt.get("branches", [])}
    rebuilt_branches = {b["branch_id"]: _round_trip(b["chain"]) for b in rebuilt.get("branches", [])}
    rec["l3_branches_reproduced"] = stored_branches == rebuilt_branches
    if not rec["treated_reproduced"]:
        rec["reasons"].append("current builder does not reproduce the released treated chain from raw")
    if not rec["l3_branches_reproduced"]:
        rec["reasons"].append("current builder does not reproduce the released L3 branches from raw")

    observed_state, _, reference = build_observed_inputs(raw, cfg)
    root_ids = {a["id"] for a in observed_state["actors"]}
    control_chain = _round_trip(build_chain(
        control_steps, cfg, {"intervention_type": policy}, track=track,
        reference=reference, root_actor_ids=root_ids, protocol=True))
    rec["control"] = _summary(control_chain)
    rec["treated"] = _summary(stored_treated)
    required = track_steps(track)
    if not (len(control_chain) == required or (control_chain and control_chain[-1].get("terminal"))):
        rec["reasons"].append(f"control chain has {len(control_chain)} nodes without a terminal")

    # Raw/reference collision agreement on the control stream (what the
    # release verifier already checks on the treated stream).
    horizon_end = round(float(cfg["intervention_time_sec"]) + required * 0.5, 2)
    hit = scored_collision(control_steps, cfg["actor_defs"], horizon_end, root_ids)
    collision_nodes = [n for n in control_chain if n.get("event_type") == "collision"]
    rec["control_raw_collision_time"] = None if hit is None else hit[0]
    if (hit is None) != (not collision_nodes):
        rec["reasons"].append("control raw collision and control chain collision node disagree")
    if hit is None:
        last_t = float(control_steps[-1].get("time_sec", float("nan")))
        if not last_t >= horizon_end - 1e-3:
            rec["reasons"].append(f"control rollout ends at {last_t:.2f}s before the horizon {horizon_end:.2f}s")

    intervention = (queries[0].get("counterfactual_chain") or {}).get("intervention") or {}
    target_id = str(intervention.get("target_actor_id"))
    rec["target_actor_id"] = target_id
    rec["intervention_type"] = intervention.get("intervention_type")
    target_key = next((k for k, s in cfg.get("actor_defs", {}).items()
                       if str(s.get("canonical_id", k)) == target_id), target_id)
    rec["treated_target_env_behavior"] = (cfg.get("env_behavior") or {}).get(target_key)
    rec["control_target_drift"] = _control_target_drift(control_chain, observed_state, target_id)
    rec["pair_effect"] = _pair_effect(control_chain, stored_treated, target_id)
    admission = {}
    manifest_path = os.path.join(scen_dir, "scenario_manifest.json")
    if os.path.isfile(manifest_path):
        admission = (json.load(open(manifest_path)).get("admission") or {})
    rec["admission_effect"] = {k: admission.get(k) for k in
                               ("max_l4_target_position_effect_m", "max_l4_target_speed_effect_mps")}
    if rec["pair_effect"]["common_depths"] == 0:
        rec["reasons"].append("control and treated chains share no depth")

    # Design fact for the dissertation: the treated L4 rollout runs the same
    # ego policy and the same scripted hazard as the L3 no-action branch.
    noaction = stored_branches.get("CF_NoAction") or stored_branches.get("Factual_NoAction")
    rec["treated_equals_l3_no_action_states"] = (
        noaction is not None and
        [n.get("actors_update") for n in noaction] == [n.get("actors_update") for n in stored_treated])
    _compare_exported_control(queries[0], control_chain, rec)

    rec["status"] = "stored_exact" if not rec["reasons"] else "needs_review"
    rec["control_chain"] = control_chain
    return rec


# ------------------------------------------------------------- nuScenes ----

def audit_nuscenes_scene(scen_dir: str) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"dir": scen_dir, "split": "nuscenes", "reasons": []}
    path = os.path.join(scen_dir, "ground_truth.json")
    if not os.path.isfile(path):
        rec.update(status="missing_inputs", reasons=["missing ground_truth.json"])
        return rec
    gt = json.load(open(path))
    rec["scenario_id"] = gt.get("scenario_id")
    proto = gt.get("protocol") or {}
    track = proto.get("track") or "primary"
    n_steps = track_steps(track)
    root = gt.get("observed_state") or {}
    twin_model = gt.get("twin_model")
    probe = ((gt.get("scenario_template") or {}).get("interaction_probe") or {})
    base_steer = probe.get("base_steer")
    if proto.get("twin_version") != TWIN_VERSION:
        rec["reasons"].append(f"twin version {proto.get('twin_version')!r} != current {TWIN_VERSION}")
    if twin_model is None:
        rec["reasons"].append("ground truth has no twin_model")
    if base_steer is None:
        rec["reasons"].append("scenario_template.interaction_probe.base_steer missing")
    elif probe.get("twin_model") not in (None, twin_model):
        rec["reasons"].append("interaction_probe twin_model differs from gt twin_model")
    queries = gt.get("l4_queries") or []
    if len(queries) != 1:
        rec["reasons"].append(f"expected exactly one l4 query, found {len(queries)}")
    admission = gt.get("nuscenes_admission") or {}
    applicability = admission.get("role_state_applicability") or {}
    profile = applicability.get("resolved_motion_profile")
    if profile is None:
        rec["reasons"].append("nuscenes_admission.role_state_applicability.resolved_motion_profile missing")
    if rec["reasons"]:
        rec["status"] = "not_recoverable"
        return rec

    query = queries[0]
    intervention = (query.get("counterfactual_chain") or {}).get("intervention") or {}
    itype = intervention.get("intervention_type")
    target_id = str(intervention.get("target_actor_id"))
    rec["intervention_type"] = itype
    rec["target_actor_id"] = target_id
    rec["ego_policy"] = proto.get("l4_ego_policy") or "do_ego_no_action"
    if itype not in ENV_PROFILES:
        rec["reasons"].append(f"intervention {itype!r} is not a twin environment profile")
        rec["status"] = "not_recoverable"
        return rec
    if applicability.get("selected_intervention") not in (None, itype):
        rec["reasons"].append("applicability selected_intervention differs from the released query")
    env_query = {
        "intervention_type": itype, "target_actor_id": target_id,
        "trigger_depth": int(intervention.get("trigger_depth", 1)),
        "resolved_accel_mps2": float(profile.get("accel", 0.0)),
        "resolved_lateral_velocity_mps": float(profile.get("lat", 0.0)),
    }
    baseline = observed_velocity_baseline(root)
    kwargs = dict(model=twin_model, env_behavior=baseline, base_steer=float(base_steer))
    treated = build_twin_branch("L4_Reviewed", False, itype, root, n_steps, 0.5, 0.5,
                                intervention_meta=dict(intervention),
                                environment_intervention=env_query, **kwargs)
    stored_treated = _round_trip((query.get("counterfactual_chain") or {}).get("chain") or [])
    rec["treated_reproduced"] = _round_trip(treated["chain"]) == stored_treated
    if not rec["treated_reproduced"]:
        rec["reasons"].append("re-rolled twin does not reproduce the released treated chain")

    control = build_twin_branch(
        "L4_Control", False, rec["ego_policy"], root, n_steps, 0.5, 0.5,
        intervention_meta={"intervention_type": CONTROL_INTERVENTION_TYPE,
                           "target_actor_id": target_id, "trigger_depth": 1},
        **kwargs)
    control_chain = _round_trip(control["chain"])

    # The admission gate recorded the per-step paired effect from the same
    # two rollouts; recompute them from the raw rolls and require equality.
    per_step = ((admission.get("paired_effect_gate") or {}).get("per_step")) or []
    c_states = roll_branch(root["actors"], rec["ego_policy"], n_steps, 0.5, **kwargs)
    t_states = roll_branch(root["actors"], rec["ego_policy"], n_steps, 0.5,
                           environment_intervention=env_query, **kwargs)
    recomputed = []
    for c_world, t_world in zip(c_states, t_states):
        c, t = c_world.get(target_id), t_world.get(target_id)
        if c is None or t is None:
            break
        recomputed.append((round(math.hypot(float(t["x"]) - float(c["x"]),
                                            float(t["y"]) - float(c["y"])), 3),
                           round(abs(_speed(t) - _speed(c)), 3)))
    stored = [(s.get("target_position_effect_m"), s.get("target_speed_effect_mps")) for s in per_step]
    rec["per_step_effect_reproduced"] = bool(per_step) and stored == recomputed
    if not per_step:
        rec["reasons"].append("paired_effect_gate.per_step missing; cannot cross-check the control roll")
    elif stored != recomputed:
        rec["reasons"].append("re-rolled control/treated effect differs from the recorded admission effect")

    rec["control"] = _summary(control_chain)
    rec["treated"] = _summary(stored_treated)
    if not (len(control_chain) == n_steps or (control_chain and control_chain[-1].get("terminal"))):
        rec["reasons"].append(f"control chain has {len(control_chain)} nodes without a terminal")
    _compare_exported_control(query, control_chain, rec)
    rec["control_target_drift"] = _control_target_drift(control_chain, root, target_id)
    rec["pair_effect"] = _pair_effect(control_chain, stored_treated, target_id)
    if rec["pair_effect"]["common_depths"] == 0:
        rec["reasons"].append("control and treated chains share no depth")
    l3_noaction = next((b for b in gt.get("branches", [])
                        if (b.get("intervention") or {}).get("intervention_type") == "do_ego_no_action"), None)
    rec["control_equals_l3_no_action_states"] = (
        l3_noaction is not None and
        [n.get("actors_update") for n in _round_trip(l3_noaction.get("chain") or [])]
        == [n.get("actors_update") for n in control_chain])
    rec["status"] = "reconstructed_exact" if not rec["reasons"] else "needs_review"
    rec["control_chain"] = control_chain
    return rec


# ----------------------------------------------------------------- main ----

def write_control(scen_dir: str, rec: Dict[str, Any], split: str) -> None:
    path = os.path.join(scen_dir, "ground_truth.json")
    gt = json.load(open(path))
    query = gt["l4_queries"][0]
    query["control_chain"] = _control_query_entry(query, rec["control_chain"], split)
    # Only what a native pair-protocol build writes (so a rebuild from raw
    # reproduces the migrated file byte-for-byte); the audit report is the
    # migration record.
    mark_pair_protocol(gt.setdefault("protocol", {}), split)
    gt["protocol"].pop("l4_control_audit", None)  # stamp of an earlier audit version
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2)
    input_path = os.path.join(scen_dir, "input_manifest.json")
    if os.path.isfile(input_path):
        manifest = json.load(open(input_path))
        mark_pair_protocol(manifest, split)
        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    # The CARLA scenario manifest carries a copy of the reference protocol
    # block (run_batch._scenario_manifest); the release verifier requires the
    # two to be equal, so the copy follows the migration.
    manifest_path = os.path.join(scen_dir, "scenario_manifest.json")
    if split == "carla" and os.path.isfile(manifest_path):
        manifest = json.load(open(manifest_path))
        if isinstance(manifest.get("protocol"), dict):
            manifest["protocol"] = copy.deepcopy(gt["protocol"])
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("carla", "nuscenes"), required=True)
    ap.add_argument("--root", required=True, help="dataset directory with scenario_* subdirectories")
    ap.add_argument("--report", help="write the per-scene JSON report here")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--write", action="store_true",
                    help="add control_chain to every ground_truth.json; refused unless every scene is exact")
    args = ap.parse_args()
    if args.write and args.limit:
        ap.error("--write cannot be combined with --limit: migration requires a whole-split audit")
    dirs = sorted((os.path.join(args.root, n) for n in os.listdir(args.root)
                   if n.startswith("scenario_") and os.path.isdir(os.path.join(args.root, n))),
                  key=lambda p: int("".join(ch for ch in os.path.basename(p) if ch.isdigit()) or 0))
    if args.limit:
        dirs = dirs[:args.limit]
    if not dirs:
        print(f"INCOMPLETE: no scenario_* directories under {args.root}", file=sys.stderr)
        return 1
    audit = audit_carla_scene if args.split == "carla" else audit_nuscenes_scene
    records = []
    for d in dirs:
        try:
            records.append(audit(d))
        except Exception as exc:  # every scene is reported; nothing is written
            records.append({"dir": d, "split": args.split, "status": "error",
                            "reasons": [f"{type(exc).__name__}: {exc}"]})
    exact_status = "stored_exact" if args.split == "carla" else "reconstructed_exact"
    counts: Dict[str, int] = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    already = sum(bool(r.get("control_already_exported")) for r in records)
    summary = {
        "audit_version": AUDIT_VERSION, "split": args.split, "root": os.path.abspath(args.root),
        "scenes": len(records), "status_counts": counts,
        "control_already_exported": already,
        "control_terminal_counts": _count(records, lambda r: (r.get("control") or {}).get("last_event")),
        "treated_terminal_counts": _count(records, lambda r: (r.get("treated") or {}).get("last_event")),
        "intervention_counts": _count(records, lambda r: r.get("intervention_type")),
        "treated_equals_l3_no_action_states": sum(1 for r in records if r.get("treated_equals_l3_no_action_states")),
        "control_equals_l3_no_action_states": sum(1 for r in records if r.get("control_equals_l3_no_action_states")),
        "min_pair_effect_position_m": _min(records, "max_target_position_effect_m"),
        "min_pair_effect_speed_mps": _min(records, "max_target_speed_effect_mps"),
        "max_control_target_speed_change_mps": _max_drift(records, "max_abs_speed_change_mps"),
        "max_control_target_lateral_change_m": _max_drift(records, "max_abs_lateral_change_m"),
    }
    for line in ("audit_version", "split", "root", "scenes", "status_counts", "control_already_exported",
                 "control_terminal_counts", "treated_terminal_counts", "intervention_counts",
                 "treated_equals_l3_no_action_states", "control_equals_l3_no_action_states",
                 "min_pair_effect_position_m", "min_pair_effect_speed_mps",
                 "max_control_target_speed_change_mps", "max_control_target_lateral_change_m"):
        print(f"{line}: {summary[line]}")
    for r in records:
        if r["status"] != exact_status:
            print(f"  {r['status']}: {r['dir']}: {r['reasons'][:2]}")
    if args.report:
        slim = [{k: v for k, v in r.items() if k != "control_chain"} for r in records]
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "scenes": slim}, f, indent=2)
        print(f"report: {args.report}")
    if args.write:
        if counts.get(exact_status, 0) != len(records):
            print("REFUSED: --write requires every scene to be exact", file=sys.stderr)
            return 2
        for r in records:
            write_control(r["dir"], r, args.split)
        print(f"wrote control_chain into {len(records)} ground_truth.json files")
    return 0 if counts.get(exact_status, 0) == len(records) else 1


def _count(records, key):
    out: Dict[str, int] = {}
    for r in records:
        k = str(key(r))
        out[k] = out.get(k, 0) + 1
    return out


def _min(records, field):
    vals = [r["pair_effect"][field] for r in records if r.get("pair_effect")]
    return min(vals) if vals else None


def _max_drift(records, field):
    vals = [r["control_target_drift"][field] for r in records
            if (r.get("control_target_drift") or {}).get(field) is not None]
    return max(vals) if vals else None


if __name__ == "__main__":
    sys.exit(main())
