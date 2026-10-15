"""
Build CORIVA ground truths from nuScenes via the Kinematic Digital Twin.

For each selected nuScenes keyframe: extract the observed_state root (adapter),
impose the CORIVA ego-intervention branch set (twin), and accept only QC-passing
scenarios (reject-and-skip with a logged reason — no silent truncation). Writes
one output/<scenario_id>/ground_truth.json per accepted scenario.

CLI:
    python3 simulation/build_nuscenes_dataset.py --dataroot data/nuscenes \\
        --version v1.0-trainval --limit 200 --out-dir output_nuscenes

Reviewed T0 selections (CSV columns: scene_name,t0_sec):
    python3 -m simulation.build_nuscenes_dataset \\
        --dataroot data/nuscenes --version v1.0-trainval \\
        --selections-csv review_selections.csv --review-dir review_videos_trainval \\
        --with-frames 8 --out-dir output_nuscenes
"""

import argparse
import csv
import functools
import json
import math
import os
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional

# Ensure the workspace root is importable whether run as a script or a module.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.nuscenes_adapter import (
    load_tables, extract_observed_trajectory, infer_scenario_spec,
    SCENARIO_TEMPLATE_POLICY, t0_visible_actor_ids,
    estimate_root_steer, factual_fidelity_error)
from simulation.kinematic_twin import build_twin_gt, TWIN_VERSION, TwinReferenceError
from simulation.qc import qc_scenario_gt
from ctms.ctms import load_ctms_environment
from simulation.protocol import (CHALLENGE_TRACK, PRIMARY_TRACK, PROTOCOL_VERSION,
                                  OBSERVED_FRAME_COUNT, SCENARIO_ACTIONS,
                                  critical_lateral_offset, scenario_actions,
                                  track_steps, protocol_manifest)
from simulation.nuscenes_admission import (
    DEFAULT_NUSCENES_ADMISSION_THRESHOLDS,
    NuScenesAdmissionThresholds,
    canonical_environment_query,
    evaluate_nuscenes_effect,
    observed_velocity_baseline,
    resolve_nuscenes_intervention,
)
from simulation.nuscenes_visibility import (
    DEFAULT_VISIBILITY_POLICY, VISIBILITY_POLICIES,
    policy_actor_visibility, visibility_policy_manifest,
)


REVIEW_FRAME_COUNT = OBSERVED_FRAME_COUNT
REVIEW_FUTURE_STEPS = track_steps(PRIMARY_TRACK)


@functools.lru_cache(maxsize=None)
def _build_code_state(repo_dir: str) -> tuple:
    """(commit, worktree_clean) of the checkout running the builder, or
    (None, None) outside a git checkout. Never invented."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo_dir, check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
    try:
        commit = git("rev-parse", "HEAD")
        dirty = [line for line in git("status", "--porcelain", "--untracked-files=no").splitlines()
                 if line.strip()]
        return commit, not dirty
    except Exception:
        return None, None


def generator_provenance(repo_dir: Optional[str] = None) -> Dict[str, Any]:
    """Identify the reference generator beyond the protocol version, so that
    campaign admission (`analysis.evaluation_campaign.dataset_inventory`) can
    refuse a build made under a different template policy or twin semantics
    even when both carry `coriva-protocol-1.0`. Written to
    the GT protocol block and the released input manifest of every scenario."""
    commit, clean = _build_code_state(
        repo_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return {"scenario_template_policy": SCENARIO_TEMPLATE_POLICY,
            "twin_version": TWIN_VERSION,
            "build_code_commit": commit, "build_worktree_clean": clean}


def _default_spec(scenario_id: str, root: Dict[str, Any], n_steps: int, dt: float,
                  factual_states=None, factual_action: Optional[str] = None,
                  t0_visible_ids=None, critical_actor_id: Optional[str] = None,
                  thresholds=None, model: str = "point_mass",
                  base_steer: float = 0.0) -> Dict[str, Any]:
    """Choose the scenario template that matches the observed scene (rear-end /
    pedestrian / cyclist), then anchor the factual branch to the real observed
    future: its states come from `factual_states` and its
    intervention label from the observed `factual_action`. Counterfactual
    branches stay twin-generated (Pearl rung-3 outcomes are unobservable)."""
    spec = infer_scenario_spec(scenario_id, root, n_steps, dt, t0_visible_ids=t0_visible_ids,
                               critical_actor_id=critical_actor_id, thresholds=thresholds,
                               model=model, base_steer=base_steer)
    if factual_states is not None:
        spec["factual_states"] = factual_states
        for b in spec["branches"]:
            if b["is_factual"] and factual_action:
                b["intervention_type"] = factual_action
    return spec


def _usable_samples(tables, samples_per_scene: int, history_frames: int = 1,
                    future_steps: int = 1, channel: str = "CAM_FRONT",
                    cam_by_sample: Optional[Dict[str, str]] = None) -> List[str]:
    """Pick complete automatic candidates, capped per scene.

    The defaults require one successor sample.  When the builder requests
    observed frames, candidates are advanced until they have
    the required consecutive camera history and full future rollout.
    """
    ann_counts: Dict[str, int] = {}
    for a in tables["sample_annotation"].values():
        ann_counts[a["sample_token"]] = ann_counts.get(a["sample_token"], 0) + 1
    if cam_by_sample is None:
        from simulation.observed_clip import _cam_keyframe_by_sample
        cam_by_sample = _cam_keyframe_by_sample(tables, channel)

    chosen: List[str] = []
    for scene in tables["scene"].values():
        tok = scene["first_sample_token"]
        taken = 0
        while tok and taken < samples_per_scene:
            s = tables["sample"][tok]
            window_error = _validate_review_window(
                tables, tok, history_frames, future_steps, channel,
                cam_by_sample=cam_by_sample)
            if window_error is None and ann_counts.get(tok, 0) > 0:
                chosen.append(tok)
                taken += 1
            tok = s["next"]
    return chosen


def _read_review_selections(csv_path: str) -> List[Dict[str, Any]]:
    """Read the minimal manual-review CSV: ``scene_name,t0_sec``."""
    selections: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"scene_name", "t0_sec"}
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise ValueError(
                f"selections CSV is missing required column(s): {', '.join(missing)}")
        for line_number, row in enumerate(reader, start=2):
            scene_name = (row.get("scene_name") or "").strip()
            raw_t0 = (row.get("t0_sec") or "").strip()
            if not scene_name:
                raise ValueError(f"selections CSV line {line_number}: scene_name is empty")
            try:
                t0_sec = float(raw_t0)
            except ValueError as exc:
                raise ValueError(
                    f"selections CSV line {line_number}: invalid t0_sec={raw_t0!r}") from exc
            if not math.isfinite(t0_sec) or t0_sec < 0:
                raise ValueError(
                    f"selections CSV line {line_number}: t0_sec must be a finite number >= 0")
            selections.append({
                "scene_name": scene_name,
                "requested_t0_sec": t0_sec,
                "csv_line": line_number,
            })
    if not selections:
        raise ValueError("selections CSV contains no data rows")
    return selections


def _sidecar_keyframes(sidecar: Dict[str, Any], sidecar_path: str) -> List[Dict[str, Any]]:
    """Return annotated CAM frames with their exact sample token and video time."""
    fps = float((sidecar.get("video") or {}).get("fps", 0.0))
    if fps <= 0:
        raise ValueError(f"invalid or missing video.fps in {sidecar_path}")

    keyframes = []
    seen_tokens = set()
    for frame in sidecar.get("frames", []):
        if not frame.get("is_key_frame"):
            continue
        sample_token = frame.get("source_sample_token")
        if not sample_token or sample_token in seen_tokens:
            continue
        raw_video_time = frame.get("video_time_sec")
        if raw_video_time is None:
            raw_video_time = int(frame["video_frame_index"]) / fps
        keyframes.append({
            "sample_token": str(sample_token),
            "video_time_sec": float(raw_video_time),
            "video_frame_index": int(frame.get("video_frame_index", 0)),
        })
        seen_tokens.add(sample_token)
    keyframes.sort(key=lambda frame: (frame["video_time_sec"], frame["video_frame_index"]))
    if not keyframes:
        raise ValueError(f"no annotated keyframes in {sidecar_path}")
    return keyframes


def _validate_review_window(tables: Dict[str, Dict[str, Any]], sample_token: str,
                            history_frames: int, future_steps: int,
                            channel: str = "CAM_FRONT",
                            cam_by_sample: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Return an actionable rejection reason, or None for a complete window."""
    from simulation.observed_clip import collect_consecutive_cam_frames

    try:
        collect_consecutive_cam_frames(
            tables, sample_token, history_frames, channel,
            cam_by_sample=cam_by_sample)
    except ValueError as exc:
        return f"history_window: {exc}"

    samples = tables["sample"]
    if sample_token not in samples:
        return f"unknown T0 sample_token: {sample_token}"
    scene_token = samples[sample_token]["scene_token"]
    tok = sample_token
    for step in range(1, future_steps + 1):
        tok = samples[tok].get("next")
        if not tok:
            return f"future_window: only {step - 1}/{future_steps} successor samples after T0"
        if samples[tok]["scene_token"] != scene_token:
            return f"future_window: successor {step} crosses a scene boundary"
    return None


def _reviewed_candidates(tables: Dict[str, Dict[str, Any]], selections_csv: str,
                         review_dir: str, history_frames: int = REVIEW_FRAME_COUNT,
                         future_steps: int = REVIEW_FUTURE_STEPS,
                         channel: str = "CAM_FRONT",
                         cam_by_sample: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """Resolve reviewer video times to T0 sample tokens and validate both windows."""
    scenes_by_name = {scene["name"]: scene for scene in tables["scene"].values()}
    if cam_by_sample is None:
        from simulation.observed_clip import _cam_keyframe_by_sample
        cam_by_sample = _cam_keyframe_by_sample(tables, channel)
    resolved: List[Dict[str, Any]] = []
    seen_sample_tokens = set()
    for selection in _read_review_selections(selections_csv):
        scene_name = selection["scene_name"]
        candidate = dict(selection)
        scene = scenes_by_name.get(scene_name)
        if scene is None:
            candidate["error"] = f"unknown scene_name={scene_name!r} for loaded nuScenes version"
            resolved.append(candidate)
            continue

        sidecar_path = os.path.join(review_dir, f"{scene_name}_{channel}.json")
        try:
            with open(sidecar_path, "r", encoding="utf-8") as f:
                sidecar = json.load(f)
            sidecar_scene = (sidecar.get("scene") or {}).get("name")
            if sidecar_scene != scene_name:
                raise ValueError(
                    f"sidecar scene mismatch: expected {scene_name!r}, got {sidecar_scene!r}")
            keyframes = _sidecar_keyframes(sidecar, sidecar_path)
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            candidate["error"] = f"sidecar: {exc}"
            resolved.append(candidate)
            continue

        nearest = min(
            keyframes,
            key=lambda frame: (
                abs(frame["video_time_sec"] - selection["requested_t0_sec"]),
                frame["video_time_sec"],
            ),
        )
        sample_token = nearest["sample_token"]
        candidate.update({
            "sample_token": sample_token,
            "resolved_t0_sec": nearest["video_time_sec"],
            "t0_snap_delta_sec": nearest["video_time_sec"] - selection["requested_t0_sec"],
            "sidecar_path": sidecar_path,
        })
        sample = tables["sample"].get(sample_token)
        if sample is None:
            candidate["error"] = f"sidecar resolved unknown sample_token={sample_token}"
        elif sample["scene_token"] != scene["token"]:
            candidate["error"] = f"resolved sample_token={sample_token} belongs to another scene"
        elif sample_token in seen_sample_tokens:
            candidate["error"] = f"duplicate T0 selection resolved to sample_token={sample_token}"
        else:
            candidate["error"] = _validate_review_window(
                tables, sample_token, history_frames, future_steps, channel,
                cam_by_sample=cam_by_sample)
            if candidate["error"] is None:
                seen_sample_tokens.add(sample_token)
        resolved.append(candidate)
    return resolved


def _automatic_candidate_records(sample_tokens: Iterable[str]) -> List[Dict[str, Any]]:
    return [{"sample_token": token} for token in sample_tokens]


def _review_row_for_candidate(review_records: List[Dict[str, Any]],
                              candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve a review-manifest row without silently mixing scenes."""
    rows = [r for r in review_records
            if str(r.get("scene_name", "")) == str(candidate.get("scene_name", ""))]
    if not rows:
        return {}
    requested = candidate.get("requested_t0_sec")
    matches = []
    for row in rows:
        raw = row.get("t0_sec")
        if raw in (None, ""):
            matches.append(row)
            continue
        try:
            if requested is not None and abs(float(raw) - float(requested)) <= 0.26:
                matches.append(row)
        except (TypeError, ValueError):
            return {"_error": "review_manifest t0_sec is not numeric"}
    if not matches:
        return {"_error": "review_manifest t0_sec does not match selections CSV"}
    if len(matches) > 1:
        return {"_error": "review_manifest has ambiguous rows for selected scene/T0"}
    return matches[0]


def _validate_reviewed_actor(tables: Dict[str, Dict[str, Any]], sample_token: str,
                             root: Dict[str, Any], review_row: Dict[str, Any],
                             history_frames: int = OBSERVED_FRAME_COUNT,
                             channel: str = "CAM_FRONT",
                             horizon_seconds: float = 3.0,
                             admission_thresholds: NuScenesAdmissionThresholds =
                             DEFAULT_NUSCENES_ADMISSION_THRESHOLDS,
                             visibility_policy: str = DEFAULT_VISIBILITY_POLICY) -> Dict[str, Any]:
    """Fail-closed validation for the manual critical-actor selection.

    The instance token is the stable nuScenes identity.  The generated CORIVA
    id is derived from that token after priority insertion; a supplied id is
    checked against the derived mapping rather than trusted as free text.
    Visibility is required at every one of the eight CAM_FRONT keyframes.
    """
    required = ("critical_instance_token", "intervention_type", "audit_status",
                "interaction_within_3s")
    missing = [key for key in required if key not in review_row]
    if missing:
        return {"ok": False, "error": "review_manifest_missing:" + ",".join(missing)}
    if str(review_row.get("audit_status")).lower() != "approved":
        return {"ok": False, "error": "review_manifest_not_human_approved"}
    if review_row.get("interaction_within_3s") is not True:
        return {"ok": False, "error": "review_manifest_interaction_within_3s_not_approved"}
    result = _check_actor_geometry(
        tables, sample_token, root, review_row, history_frames, channel,
        horizon_seconds, admission_thresholds, visibility_policy)
    if result.get("ok"):
        result["interaction_within_3s"] = True
    return result


def _check_actor_geometry(tables, sample_token, root, review_row,
                          history_frames=OBSERVED_FRAME_COUNT, channel="CAM_FRONT",
                          horizon_seconds=3.0,
                          admission_thresholds=DEFAULT_NUSCENES_ADMISSION_THRESHOLDS,
                          visibility_policy=DEFAULT_VISIBILITY_POLICY):
    """Mechanical actor checks only; never grant human approval."""
    instance_token = str(review_row["critical_instance_token"])
    actor = next((a for a in root.get("actors", [])
                  if a.get("source_instance_token") == instance_token), None)
    if actor is None:
        return {"ok": False, "error": "critical_actor_not_in_root"}
    supplied_id = review_row.get("critical_actor_id")
    if supplied_id is not None and str(supplied_id) != str(actor["id"]):
        return {"ok": False, "error":
                f"critical_actor_id_mismatch: expected {actor['id']}, got {supplied_id}"}
    intervention_type = str(review_row["intervention_type"])
    applicability = resolve_nuscenes_intervention(
        actor, intervention_type, horizon_seconds=horizon_seconds,
        thresholds=admission_thresholds)
    if not applicability["passed"]:
        return {"ok": False, "error": applicability["reasons"][0],
                "applicability": applicability}

    from simulation.observed_clip import collect_consecutive_cam_frames
    try:
        collect_consecutive_cam_frames(tables, sample_token, history_frames, channel)
        tokens = [sample_token]
        tok = sample_token
        for _ in range(history_frames - 1):
            tok = tables["sample"][tok].get("prev")
            tokens.append(tok)
        tokens.reverse()
        visibility = [policy_actor_visibility(tables, tok, instance_token, channel,
                                              policy=visibility_policy)
                      for tok in tokens]
    except (KeyError, ValueError, TypeError) as exc:
        return {"ok": False, "error": f"critical_visibility_check:{exc}",
                "applicability": applicability}
    if not all(item.get("visible") for item in visibility):
        return {"ok": False, "error": "critical_actor_not_visible_all_history_frames",
                "visibility": visibility, "applicability": applicability}
    return {"ok": True, "critical_actor_id": actor["id"],
            "critical_instance_token": instance_token, "entity_type": actor.get("entity_type"),
            "role_type": actor.get("role_type"),
            "visibility": visibility,
            "applicability": applicability}


def _prepare_candidate_spec(scenario_id, root, factual_states, factual_action,
                            n_steps, dt, twin_model, base_steer, *,
                            t0_visible_ids=None, critical_actor_id=None, thresholds=None):
    """Shared branch construction for builds and metadata-only recovery checks.
    `t0_visible_ids`, `critical_actor_id` and `thresholds` feed the L3
    template selection (policy `interaction-v3`); `twin_model` and
    `base_steer` make its interaction probe the same rollout as the L4 gate."""
    spec = _default_spec(scenario_id, root, n_steps, dt, factual_states, factual_action,
                         t0_visible_ids=t0_visible_ids, critical_actor_id=critical_actor_id,
                         thresholds=thresholds, model=twin_model, base_steer=base_steer)
    scenario_type = spec.get("scenario_type")
    # The lateral escape follows the hazard the L3 template selected, which the
    # adapter already recorded. The actor reviewed for the L4 query can be a
    # different one, so using it here would overwrite the adapter's choice.
    lateral_actor = ((spec.get("scenario_template") or {}).get("selected_actor_id")
                     or spec.get("critical_actor_id") or critical_actor_id)
    declared_actions = list(scenario_actions(
        scenario_type, critical_lateral_offset(root, lateral_actor)))
    if declared_actions:
        # Deterministic declared action universe: preserve declaration
        # order, remove duplicates, and retain the observed factual label
        # as metadata even when it is outside the coarse template.
        unique_actions = list(dict.fromkeys(declared_actions))
        spec["branches"] = [
            {"branch_id": ("Factual" if action == factual_action else "CF") + "_" + action,
             "is_factual": action == factual_action,
             "intervention_type": action,
             "target_actor_id": "ego", "trigger_depth": 1}
            for action in unique_actions
        ]
        # The real factual branch is observed data, even when its coarse
        # action label is outside the declared three-action set. Keep one
        # factual branch and label it with the first admissible action.
        if not any(b["is_factual"] for b in spec["branches"]):
            # The observed real action can be outside the three declared
            # counterfactual action templates.  Keep all templates as L3
            # branches and add one factual branch instead of overwriting a
            # declared action and silently shrinking the action universe.
            spec["branches"] = [{
                "branch_id": "Factual_Observed",
                "is_factual": True,
                "intervention_type": factual_action or "do_ego_no_action",
                "target_actor_id": "ego", "trigger_depth": 1,
            }] + spec["branches"]
        spec["action_audit"] = {
            "declared_actions": unique_actions,
            "factual_action": factual_action,
            "enumerated_count": len(spec["branches"]),
            "deduplicated": len(unique_actions) != len(declared_actions),
            "gate": "qc_scenario_gt",
        }
    spec["model"] = twin_model
    spec["base_steer"] = base_steer
    return spec


def _append_reviewed_l4(spec, root, query):
    spec["branches"].append({
        "branch_id": "L4_Reviewed",
        "is_factual": False,
        "intervention_type": query["intervention_type"],
        "target_actor_id": query["target_actor_id"],
        "trigger_depth": 1,
        "environment_intervention": query,
        # L4 control and treated rollouts continue the T0-observed
        # velocity.  L3's scenario templates remain separate.
        "env_behavior": observed_velocity_baseline(root),
    })


def preview_reviewed_candidate(tables, sample_token, instance_token, intervention_type,
                               env, *, epsilon_max=2.0, twin_model="bicycle",
                               admission_thresholds=DEFAULT_NUSCENES_ADMISSION_THRESHOLDS,
                               visibility_policy=DEFAULT_VISIBILITY_POLICY):
    """Run primary mechanical gates without reading images or claiming review.

    Uses the same actor validation, root/history construction, branch spec,
    fidelity, paired effect and QC functions as the reviewed dataset builder.
    A passing preview still requires a fresh human decision and a final build.
    """
    from simulation.review_overlay import observable_root_universe
    from simulation.nuscenes_adapter import extract_observed_history
    visibility_policy_manifest(visibility_policy)
    if not math.isfinite(epsilon_max) or epsilon_max <= 0:
        raise ValueError("recovery requires a finite positive fidelity threshold")
    stage = "window"
    details = {}
    try:
        error = _validate_review_window(tables, sample_token, 8, 6)
        if error:
            return {"passed": False, "stage": stage, "reasons": [error]}
        stage = "root_universe"
        universe = observable_root_universe(tables, sample_token, 8,
                                            visibility_policy=visibility_policy)
        allowed = set(universe["allowed_instance_tokens"])
        if instance_token not in allowed:
            return {"passed": False, "stage": stage, "reasons": ["critical_actor_not_camera_observable"]}
        traj = extract_observed_trajectory(tables, sample_token, 6, max_actors=5, radius=60.0,
                                           priority_instance_token=instance_token,
                                           allowed_instance_tokens=allowed)
        root, factual, action = traj["root"], traj["factual_states"], traj["factual_action"]
        if len(factual) != 6:
            return {"passed": False, "stage": "window", "reasons": ["incomplete factual future"]}
        stage = "history"
        extract_observed_history(tables, sample_token, root, universe["visibility_by_instance"])
        stage = "actor"
        check = _check_actor_geometry(tables, sample_token, root,
            {"critical_instance_token": instance_token, "intervention_type": intervention_type},
            admission_thresholds=admission_thresholds, visibility_policy=visibility_policy)
        if not check["ok"]:
            return {"passed": False, "stage": stage, "reasons": [check["error"]]}
        speed = float(next(a for a in root["actors"] if a["id"] == "ego").get("speed", 0))
        steer = estimate_root_steer(tables, sample_token, speed) if twin_model == "bicycle" else 0.0
        stage = "fidelity"
        epsilon = factual_fidelity_error(root, factual, action, 0.5, model=twin_model, base_steer=steer)
        details.update(epsilon=epsilon, base_steer=steer, factual_action=action)
        if not math.isfinite(epsilon) or epsilon > epsilon_max:
            return {**details, "passed": False, "stage": stage, "reasons": ["fidelity_above_threshold"]}
        stage = "effect"
        query = canonical_environment_query(check["critical_actor_id"], check["applicability"])
        effect = evaluate_nuscenes_effect(root, query, n_steps=6, dt=0.5, model=twin_model,
                                         base_steer=steer, thresholds=admission_thresholds)
        if not effect["passed"]:
            return {**details, "passed": False, "stage": stage, "reasons": effect["reasons"]}
        stage = "qc"
        spec = _prepare_candidate_spec(
            "nuscenes_0001", root, factual, action, 6, 0.5, twin_model, steer,
            t0_visible_ids=t0_visible_actor_ids(root, universe["visibility_by_instance"], sample_token),
            critical_actor_id=check["critical_actor_id"], thresholds=admission_thresholds)
        _append_reviewed_l4(spec, root, query)
        gt = build_twin_gt(root, spec)
        ok, reasons = qc_scenario_gt(gt, env["ontology"], env["config"], env["branch_schema"])
        if not ok:
            return {**details, "passed": False, "stage": stage, "reasons": reasons}
        return {**details, "passed": True, "stage": "mechanical_pass", "reasons": [],
                "root": root, "sample_tokens": universe["sample_tokens"],
                "critical_actor_id": check["critical_actor_id"], "visibility": check["visibility"],
                "effect": effect, "scenario_template": spec["scenario_template"],
                "scenario_type": spec["scenario_type"], "human_review_status": "pending"}
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return {**details, "passed": False, "stage": stage, "reasons": [str(exc)], "metadata_error": True}


def build_dataset(dataroot: str, version: str, limit: int, out_dir: str,
                  env: Optional[Dict[str, Any]] = None, n_steps: int = 6, dt: float = 0.5,
                  samples_per_scene: int = 3, max_actors: int = 5, radius: float = 60.0,
                  with_frames: int = 0, twin_model: str = "bicycle",
                  epsilon_max: Optional[float] = 2.0,
                  selections_csv: Optional[str] = None,
                  review_dir: Optional[str] = None,
                  track: str = PRIMARY_TRACK,
                  review_manifest: Optional[str] = None,
                  admission_thresholds: NuScenesAdmissionThresholds =
                  DEFAULT_NUSCENES_ADMISSION_THRESHOLDS,
                  visibility_policy: str = DEFAULT_VISIBILITY_POLICY) -> Dict[str, Any]:
    """Generate up to `limit` QC-accepted CORIVA GTs from nuScenes. Returns
    {accepted, rejected, paths, rejected_reasons, fidelity, effect_gate}.

    `twin_model` defaults to 'bicycle' with per-scenario `base_steer`
    estimated from the ego's PAST yaw rate. Every GT
    records `factual_fidelity_error` + `twin_model`, the run writes a
    fidelity_report.json distribution, and `epsilon_max` (protocol default
    2.0 m; None disables, e.g. for pilot/instrument builds) enforces the
    admission gate of dissertation §3.5.2."""
    visibility_config = visibility_policy_manifest(visibility_policy)
    if os.path.lexists(out_dir) and (not os.path.isdir(out_dir) or os.listdir(out_dir)):
        raise ValueError("output directory must be new or empty; do not mix build runs or visibility policies")
    if env is None:
        env = load_ctms_environment("ctms/ontology.yaml", "ctms/ctms_config.yaml",
                                    "ctms/schema_branch.json", "ctms/schema_counterfactual.json")
    tables = load_tables(dataroot, version)
    review_records: List[Dict[str, Any]] = []
    if review_manifest:
        with open(review_manifest, encoding="utf-8") as f:
            raw_manifest = json.load(f)
        manifest_rows = raw_manifest.get("selections", raw_manifest if isinstance(raw_manifest, list) else [])
        for row in manifest_rows:
            if isinstance(row, dict) and row.get("scene_name"):
                review_records.append(dict(row))
    os.makedirs(out_dir, exist_ok=True)
    cam_by_sample: Optional[Dict[str, str]] = None
    if selections_csv or with_frames:
        from simulation.observed_clip import _cam_keyframe_by_sample
        cam_by_sample = _cam_keyframe_by_sample(tables, "CAM_FRONT")

    if selections_csv:
        if not review_dir:
            raise ValueError("review_dir is required when selections_csv is provided")
        if not review_manifest:
            raise ValueError(
                "review_manifest is required for reviewed protocol builds; "
                "provide critical_instance_token, intervention_type, "
                "interaction_within_3s=true and audit_status=approved")
        # Reviewed mode has a fixed benchmark contract: 8 consecutive observed
        # keyframes and the selected primary/challenge future horizon.
        if not math.isclose(dt, 0.5, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("reviewed mode requires dt=0.5 seconds")
        n_steps = track_steps(track)
        if with_frames == 0:
            with_frames = REVIEW_FRAME_COUNT
        elif with_frames != REVIEW_FRAME_COUNT:
            raise ValueError(
                f"reviewed mode requires --with-frames {REVIEW_FRAME_COUNT}")
        candidates = _reviewed_candidates(
            tables, selections_csv, review_dir,
            history_frames=REVIEW_FRAME_COUNT, future_steps=n_steps,
            cam_by_sample=cam_by_sample)
    else:
        history_frames = with_frames if with_frames > 0 else 1
        # Without requested frames one successor sample is enough.  Once a
        # frame window is part
        # of the contract, validate the complete requested future as well.
        required_future_steps = n_steps if with_frames else 1
        candidates = _automatic_candidate_records(_usable_samples(
            tables, samples_per_scene, history_frames=history_frames,
            future_steps=required_future_steps, cam_by_sample=cam_by_sample))

    paths: List[str] = []
    rejected_reasons: List[Dict[str, Any]] = []
    fidelity_rows: List[Dict[str, Any]] = []
    effect_rows: List[Dict[str, Any]] = []
    idx = 0
    selection_rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        if len(paths) >= limit:
            break
        idx += 1
        scenario_id = f"nuscenes_{idx:04d}"
        sample_token = candidate.get("sample_token")
        review_row = _review_row_for_candidate(review_records, candidate) if review_manifest else {}
        if selections_csv and review_manifest and not review_row:
            rejected_reasons.append({"scenario_id": scenario_id,
                                     "scene_name": candidate.get("scene_name"),
                                     "reasons": ["review_manifest_missing_scene_t0_row"]})
            selection_rows.append({**candidate, "scenario_id": scenario_id, "status": "rejected"})
            continue
        if review_row.get("_error"):
            rejected_reasons.append({"scenario_id": scenario_id,
                                     "scene_name": candidate.get("scene_name"),
                                     "reasons": [review_row["_error"]]})
            selection_rows.append({**candidate, "scenario_id": scenario_id, "status": "rejected"})
            continue
        if review_row.get("intervention_type"):
            candidate["review_intervention_type"] = review_row["intervention_type"]
        if candidate.get("error"):
            rejection = {
                "scenario_id": scenario_id,
                "scene_name": candidate.get("scene_name"),
                "reasons": [f"selection: {candidate['error']}"],
            }
            rejected_reasons.append(rejection)
            selection_rows.append({**candidate, "scenario_id": scenario_id, "status": "rejected"})
            continue
        if with_frames:
            from simulation.observed_clip import collect_consecutive_cam_frames
            try:
                source_frames = collect_consecutive_cam_frames(
                    tables, sample_token, with_frames, "CAM_FRONT",
                    cam_by_sample=cam_by_sample)
                missing_source = next(
                    (fn for fn in source_frames
                     if not os.path.isfile(os.path.join(dataroot, fn))), None)
                if missing_source:
                    raise FileNotFoundError(os.path.join(dataroot, missing_source))
            except (FileNotFoundError, ValueError) as exc:
                rejected_reasons.append({
                    "scenario_id": scenario_id,
                    "scene_name": candidate.get("scene_name"),
                    "reasons": [f"frame_window: {exc}"],
                })
                if selections_csv:
                    selection_rows.append({
                        **candidate, "scenario_id": scenario_id, "status": "rejected"})
                continue
        universe = None
        allowed_instances = None
        if with_frames == OBSERVED_FRAME_COUNT:
            from simulation.review_overlay import observable_root_universe
            try:
                universe = observable_root_universe(
                    tables, sample_token, OBSERVED_FRAME_COUNT, radius=radius,
                    visibility_policy=visibility_policy)
                allowed_instances = set(universe["allowed_instance_tokens"])
            except (KeyError, ValueError, TypeError) as exc:
                rejected_reasons.append({"scenario_id": scenario_id,
                                         "reasons": [f"root_universe: {exc}"]})
                continue
            critical_instance = review_row.get("critical_instance_token")
            if critical_instance and critical_instance not in allowed_instances:
                measurements = [policy_actor_visibility(
                    tables, token, critical_instance, policy=visibility_policy)
                    for token in universe["sample_tokens"]]
                rejected_reasons.append({"scenario_id": scenario_id,
                                         "scene_name": candidate.get("scene_name"),
                                         "reasons": ["critical_actor_not_camera_observable"],
                                         "visibility": measurements})
                selection_rows.append({**candidate, "scenario_id": scenario_id, "status": "rejected",
                                       "reasons": ["critical_actor_not_camera_observable"],
                                       "critical_actor_visibility_measurements": measurements})
                continue
        traj = extract_observed_trajectory(
            tables, sample_token, n_steps, max_actors=max_actors, radius=radius,
            priority_instance_token=review_row.get("critical_instance_token"),
            allowed_instance_tokens=allowed_instances)
        root, factual_states, factual_action = (
            traj["root"], traj["factual_states"], traj["factual_action"])
        observed_history = None
        if universe is not None:
            from simulation.nuscenes_adapter import extract_observed_history
            try:
                observed_history = extract_observed_history(
                    tables, sample_token, root, universe["visibility_by_instance"])
            except (KeyError, ValueError, TypeError) as exc:
                rejected_reasons.append({"scenario_id": scenario_id,
                                         "reasons": [f"observed_history: {exc}"]})
                continue
        if selections_csv and review_manifest:
            actor_check = _validate_reviewed_actor(
                tables, sample_token, root, review_row, REVIEW_FRAME_COUNT,
                horizon_seconds=REVIEW_FUTURE_STEPS * dt,
                admission_thresholds=admission_thresholds,
                visibility_policy=visibility_policy)
            if not actor_check.get("ok"):
                if (actor_check.get("applicability") is not None
                        and not actor_check["applicability"].get("passed")):
                    effect_rows.append({
                        "scenario_id": scenario_id,
                        "scene_name": candidate.get("scene_name"),
                        "source_sample_token": sample_token,
                        "stage": "role_state_applicability",
                        "gate_passed": False,
                        "role_state_applicability": actor_check["applicability"],
                        "effect": None,
                    })
                rejected_reasons.append({"scenario_id": scenario_id,
                                         "scene_name": candidate.get("scene_name"),
                                         "reasons": [actor_check["error"]],
                                         "visibility": actor_check.get("visibility")})
                selection_rows.append({**candidate, "scenario_id": scenario_id,
                                       "status": "rejected", "actor_check": actor_check})
                continue
            candidate["critical_actor_id"] = actor_check["critical_actor_id"]
            candidate["critical_instance_token"] = actor_check["critical_instance_token"]
            review_row["critical_actor_id"] = candidate["critical_actor_id"]
            review_row["critical_instance_token"] = candidate["critical_instance_token"]
            candidate["audit_status"] = review_row.get("audit_status")
            candidate["interaction_within_3s"] = actor_check["interaction_within_3s"]
            candidate["critical_actor_role"] = actor_check["role_type"]
            candidate["role_state_applicability"] = actor_check["applicability"]
            candidate["critical_actor_visibility_measurements"] = actor_check["visibility"]
        if candidate.get("critical_actor_id"):
            ids = {a["id"] for a in root["actors"]}
            if candidate["critical_actor_id"] not in ids:
                rejected_reasons.append({"scenario_id": scenario_id,
                                         "reasons": ["critical_actor_not_in_root"]})
                continue
        if selections_csv and len(factual_states) != n_steps:
            rejection = {
                "scenario_id": scenario_id,
                "scene_name": candidate.get("scene_name"),
                "reasons": [
                    f"future_window: adapter extracted {len(factual_states)}/{n_steps} states"],
            }
            rejected_reasons.append(rejection)
            selection_rows.append({**candidate, "scenario_id": scenario_id, "status": "rejected"})
            continue
        ego_speed = float(next(a for a in root["actors"] if a["id"] == "ego").get("speed", 0.0))
        base_steer = (estimate_root_steer(tables, sample_token, ego_speed)
                      if twin_model == "bicycle" else 0.0)
        t0_visible = (t0_visible_actor_ids(root, universe["visibility_by_instance"], sample_token)
                      if universe is not None else None)
        spec = _prepare_candidate_spec(scenario_id, root, factual_states, factual_action,
                                       n_steps, dt, twin_model, base_steer,
                                       t0_visible_ids=t0_visible,
                                       critical_actor_id=candidate.get("critical_actor_id"),
                                       thresholds=admission_thresholds)
        epsilon = factual_fidelity_error(root, factual_states, factual_action, dt,
                                         model=twin_model, base_steer=base_steer)
        fidelity_admitted = epsilon_max is None or epsilon <= epsilon_max
        fidelity_rows.append({"scenario_id": scenario_id, "epsilon": epsilon,
                              "base_steer": round(base_steer, 4), "admitted": fidelity_admitted,
                              "scene_name": candidate.get("scene_name"),
                              "source_sample_token": sample_token})
        effect_gate = None
        if candidate.get("review_intervention_type") and candidate.get("critical_actor_id"):
            if not review_manifest:
                raise ValueError("review_intervention_type requires --review-manifest with stable actor metadata")
            query = canonical_environment_query(
                candidate["critical_actor_id"], candidate["role_state_applicability"])
            effect_gate = evaluate_nuscenes_effect(
                root, query, n_steps=REVIEW_FUTURE_STEPS, dt=dt, model=twin_model,
                base_steer=base_steer, thresholds=admission_thresholds)
            effect_rows.append({
                "scenario_id": scenario_id,
                "scene_name": candidate.get("scene_name"),
                "source_sample_token": sample_token,
                "stage": "paired_canonical_twin_effect",
                "gate_passed": bool(effect_gate["passed"]),
                "role_state_applicability": candidate["role_state_applicability"],
                "effect": effect_gate,
            })
            _append_reviewed_l4(spec, root, query)
        admission_reasons = []
        if not fidelity_admitted:
            admission_reasons.append(
                f"fidelity_gate: eps={epsilon:.3f} > epsilon_max={epsilon_max}")
        if effect_gate is not None and not effect_gate["passed"]:
            admission_reasons.extend(
                f"effect_gate:{reason}" for reason in effect_gate["reasons"])
        if admission_reasons:
            rejected_reasons.append({
                "scenario_id": scenario_id,
                "scene_name": candidate.get("scene_name"),
                "reasons": admission_reasons,
                "effect_gate": effect_gate,
            })
            if selections_csv:
                selection_rows.append({
                    **candidate, "scenario_id": scenario_id, "status": "rejected",
                    "reasons": admission_reasons, "effect_gate": effect_gate})
            continue
        try:
            gt = build_twin_gt(root, spec)
        except TwinReferenceError as exc:
            # Contact inside the first sample interval has no representable
            # reference; reject the scene, do not stop the build.
            rejected_reasons.append({
                "scenario_id": scenario_id,
                "scene_name": candidate.get("scene_name"),
                "reasons": [f"twin_reference_error: {exc}"],
            })
            if selections_csv:
                selection_rows.append({
                    **candidate, "scenario_id": scenario_id, "status": "rejected",
                    "reasons": [f"twin_reference_error: {exc}"]})
            continue
        gt["source_sample_token"] = sample_token  # provenance (frame prep, audits)
        gt["scenario_template"] = spec["scenario_template"]
        if selections_csv:
            gt["source_scene_name"] = candidate["scene_name"]
            gt["review_requested_t0_sec"] = candidate["requested_t0_sec"]
            gt["review_resolved_t0_sec"] = candidate["resolved_t0_sec"]
        gt["twin_model"] = twin_model
        gt["factual_fidelity_error"] = epsilon
        if candidate.get("role_state_applicability") is not None:
            gt["nuscenes_admission"] = {
                "semantics": "canonical_twin_applicability_not_physical_truth",
                "visibility_policy": visibility_config,
                "critical_actor_visibility_measurements": candidate.get("critical_actor_visibility_measurements"),
                "role_state_applicability": candidate["role_state_applicability"],
                "paired_effect_gate": effect_gate,
            }
        protocol_build = bool(selections_csv) or with_frames == OBSERVED_FRAME_COUNT
        if protocol_build:
            gt["protocol"] = protocol_manifest(track=track, split=version)
            gt["protocol"]["visibility_policy"] = visibility_config
            gt["protocol"].update(generator_provenance())
            if candidate.get("critical_actor_id"):
                gt["protocol"]["critical_actor_id"] = candidate["critical_actor_id"]
                gt["protocol"]["critical_instance_token"] = candidate.get("critical_instance_token")
                gt["protocol"]["critical_actor_visibility"] = "all_8_cam_front_frames"
                gt["protocol"]["l4_ego_policy"] = "do_ego_no_action"
                gt["protocol"]["l4_reference_source"] = "deterministic_canonical_twin"
                if gt.get("l4_queries"):
                    from simulation.l4_control import mark_pair_protocol
                    mark_pair_protocol(gt["protocol"], "nuscenes")
                gt["protocol"]["human_audit_status"] = candidate.get("audit_status")
                gt["protocol"]["human_interaction_within_3s"] = True
                gt["protocol"]["critical_actor_role"] = candidate.get("critical_actor_role")
                gt["protocol"]["role_state_applicability_gate"] = "passed"
                gt["protocol"]["canonical_twin_effect_gate"] = "passed"
        ok, reasons = qc_scenario_gt(gt, env["ontology"], env["config"], env["branch_schema"])
        if not ok:
            rejected_reasons.append({"scenario_id": scenario_id, "reasons": reasons})
            if selections_csv:
                selection_rows.append({
                    **candidate, "scenario_id": scenario_id, "status": "rejected",
                    "reasons": reasons, "effect_gate": effect_gate})
            continue
        # Dir name matches the eval pipeline glob (scenario_*); scenario_id stays informative.
        scen_dir = os.path.join(out_dir, f"scenario_{idx}")
        os.makedirs(scen_dir, exist_ok=True)
        path = os.path.join(scen_dir, "ground_truth.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(gt, f, indent=2)
        if with_frames:
            if protocol_build:
                from simulation.observed_clip import write_model_frames
                write_model_frames(
                    tables, sample_token, dataroot, scen_dir, root,
                    num_frames=with_frames, cam_by_sample=cam_by_sample,
                    visibility_policy=visibility_policy)
            else:
                from simulation.observed_clip import write_scenario_frames
                write_scenario_frames(
                    tables, sample_token, dataroot, scen_dir, num_frames=with_frames,
                    cam_by_sample=cam_by_sample)
        if protocol_build:
            from simulation.observed_history import (write_observed_artifacts,
                                                       write_protocol_sidecars)
            if observed_history is None:
                raise ValueError("protocol build did not produce observed_history")
            write_observed_artifacts(scen_dir, root, observed_history)
            write_protocol_sidecars(scen_dir, gt, env["ontology"])
            with open(os.path.join(scen_dir, "input_manifest.json"), "w", encoding="utf-8") as f:
                json.dump(protocol_manifest(
                    track=track, split=version, condition="video",
                    l4_pair_protocol=gt["protocol"].get("l4_pair_protocol"),
                    l4_control_reference_source=gt["protocol"].get("l4_control_reference_source"),
                    l4_ego_policy=gt["protocol"].get("l4_ego_policy"),
                    visibility_policy=visibility_config,
                    **generator_provenance(),
                    scenario_template_actor_id=spec["scenario_template"]["selected_actor_id"],
                    critical_actor_visibility_measurements=candidate.get("critical_actor_visibility_measurements"),
                    critical_actor_id=candidate.get("critical_actor_id"),
                    critical_instance_token=review_row.get("critical_instance_token"),
                    intervention_type=candidate.get("review_intervention_type"),
                    human_interaction_within_3s=candidate.get("interaction_within_3s"),
                    role_state_applicability_gate=(
                        "passed" if candidate.get("role_state_applicability") else None),
                    canonical_twin_effect_gate=(
                        "passed" if effect_gate and effect_gate.get("passed") else None),
                    source_sample_token=sample_token,
                    coordinate_frame="t0_ego",
                    frame_asset="frames",
                    raw_frame_asset="raw_frames",
                    observed_state_asset="observed_state.json",
                    observed_history_asset="observed_history.json",
                    ontology_asset="ontology.json", query_asset="query.json",
                    overlay_version=("coriva-id-overlay-bbox-3" if visibility_policy == "bbox-area-v2"
                                     else "coriva-id-overlay-2")), f, indent=2)
        paths.append(path)
        if selections_csv:
            selection_rows.append({
                **candidate, "scenario_id": scenario_id, "status": "accepted",
                "effect_gate": effect_gate, "output_path": path})

    # Publish the fidelity-error distribution of this build.
    eps_vals = sorted(r["epsilon"] for r in fidelity_rows)
    fidelity_report = {
        "twin_model": twin_model,
        "epsilon_max": epsilon_max,
        "per_scenario": fidelity_rows,
        "summary": {
            "candidates": len(fidelity_rows),
            "admitted": sum(1 for r in fidelity_rows if r["admitted"]),
            "median": eps_vals[len(eps_vals) // 2] if eps_vals else None,
            "mean": (sum(eps_vals) / len(eps_vals)) if eps_vals else None,
            "max": eps_vals[-1] if eps_vals else None,
        },
    }
    with open(os.path.join(out_dir, "fidelity_report.json"), "w", encoding="utf-8") as f:
        json.dump(fidelity_report, f, indent=2)
    effect_reason_counts: Dict[str, int] = {}
    for row in effect_rows:
        reasons = ((row.get("effect") or {}).get("reasons")
                   or (row.get("role_state_applicability") or {}).get("reasons") or [])
        for reason in reasons:
            effect_reason_counts[reason] = effect_reason_counts.get(reason, 0) + 1
    passed_by_intervention: Dict[str, int] = {}
    for row in effect_rows:
        if not row.get("gate_passed"):
            continue
        intervention = str((row.get("role_state_applicability") or {}).get(
            "selected_intervention", "unknown"))
        passed_by_intervention[intervention] = passed_by_intervention.get(intervention, 0) + 1
    released_by_intervention: Dict[str, int] = {}
    released_by_role: Dict[str, int] = {}
    for row in selection_rows:
        if row.get("status") != "accepted":
            continue
        applicability = row.get("role_state_applicability") or {}
        intervention = str(applicability.get("selected_intervention", "unknown"))
        role = str((applicability.get("t0_state") or {}).get("role_type", "unknown"))
        released_by_intervention[intervention] = released_by_intervention.get(intervention, 0) + 1
        released_by_role[role] = released_by_role.get(role, 0) + 1
    effect_report = {
        "visibility_policy": visibility_config,
        "semantics": "canonical_twin_applicability_not_physical_truth",
        "thresholds": admission_thresholds.to_dict(),
        "per_scenario": effect_rows,
        "summary": {
            "candidates": len(effect_rows),
            "role_state_rejected": sum(
                1 for row in effect_rows if row.get("stage") == "role_state_applicability"),
            "paired_effect_evaluated": sum(
                1 for row in effect_rows if row.get("effect") is not None),
            "passed": sum(1 for row in effect_rows if row.get("gate_passed")),
            "paired_effect_rejected": sum(
                1 for row in effect_rows
                if row.get("effect") is not None and not row.get("gate_passed")),
            "rejected": sum(1 for row in effect_rows if not row.get("gate_passed")),
            "released_after_all_gates": sum(
                1 for row in selection_rows if row.get("status") == "accepted"),
            "rejection_reason_counts": effect_reason_counts,
            "passed_by_intervention": passed_by_intervention,
            "released_by_intervention": released_by_intervention,
            "released_by_role": released_by_role,
        },
    }
    with open(os.path.join(out_dir, "nuscenes_effect_report.json"), "w", encoding="utf-8") as f:
        json.dump(effect_report, f, indent=2)
    if selections_csv:
        with open(os.path.join(out_dir, "resolved_selections.json"), "w", encoding="utf-8") as f:
            json.dump({
                "source_csv": os.path.abspath(selections_csv),
                "visibility_policy": visibility_config,
                "review_dir": os.path.abspath(review_dir or ""),
                "future_steps": track_steps(track),
                "track": track,
                "protocol_version": PROTOCOL_VERSION,
                "dt_seconds": dt,
                "selections": selection_rows,
                "review_manifest_required": True,
                "review_manifest": os.path.abspath(review_manifest or ""),
                "human_audit_required":
                    "audit_status=approved and interaction_within_3s=true",
                "critical_actor_policy":
                    "role_state_eligible_dynamic_actor_with_stable_instance_token_visible_all_8_cam_front_frames",
                "scenario_applicability_policy":
                    "role_state_gate_then_paired_canonical_twin_effect_gate_then_human_audit",
                "canonical_twin_effect_report": "nuscenes_effect_report.json",
            }, f, indent=2)

    return {"accepted": len(paths), "rejected": len(rejected_reasons),
            "paths": paths, "rejected_reasons": rejected_reasons,
            "fidelity": fidelity_report["summary"],
            "effect_gate": effect_report["summary"]}


def main():
    p = argparse.ArgumentParser(description="Build CORIVA GTs from nuScenes via the kinematic twin")
    p.add_argument("--dataroot", default="data/nuscenes")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--out-dir", default="output_nuscenes")
    p.add_argument("--samples-per-scene", type=int, default=3)
    p.add_argument("--with-frames", type=int, default=0,
                   help="copy N consecutive CAM_FRONT keyframes ending at T0 "
                        "(reviewed mode always writes 8; automatic mode 0 = off)")
    p.add_argument("--selections-csv", default=None,
                   help="manual review CSV with columns scene_name,t0_sec; "
                        "enables the fixed 8-frame protocol window")
    p.add_argument("--review-dir", default=None,
                   help="directory containing scene-NNNN_CAM_FRONT.json review sidecars")
    p.add_argument("--track", choices=[PRIMARY_TRACK, CHALLENGE_TRACK], default=PRIMARY_TRACK)
    p.add_argument("--review-manifest", default=None,
                   help="required for reviewed L4: stable dynamic actor/intervention metadata, "
                        "interaction_within_3s=true and audit_status=approved")
    p.add_argument("--visibility-policy", choices=VISIBILITY_POLICIES,
                   default=DEFAULT_VISIBILITY_POLICY,
                   help="bbox-area-v2 (default): projected area >=100 px2 at 1024x768; "
                        "center-v1: projected-center visibility")
    p.add_argument("--twin-model", choices=["point_mass", "bicycle"], default="bicycle",
                   help="ego forward model; bicycle adds per-scenario base_steer from past yaw rate")
    p.add_argument("--epsilon-max", type=float, default=2.0,
                   help="fidelity admission gate in metres; <=0 disables (pilot/instrument builds)")
    p.add_argument("--effect-min-position-m", type=float, default=1.0,
                   help="minimum treated/control target-position effect in metres")
    p.add_argument("--effect-min-speed-mps", type=float, default=1.0,
                   help="minimum treated/control target-speed effect in m/s")
    p.add_argument("--effect-max-interaction-gap-m", type=float, default=5.0,
                   help="maximum treated ego-target body-edge gap within the primary horizon")
    args = p.parse_args()
    eps_max = args.epsilon_max if args.epsilon_max > 0 else None
    if min(args.effect_min_position_m, args.effect_min_speed_mps,
           args.effect_max_interaction_gap_m) < 0:
        p.error("nuScenes effect thresholds must be non-negative")
    admission_thresholds = NuScenesAdmissionThresholds(
        min_target_position_effect_m=args.effect_min_position_m,
        min_target_speed_effect_mps=args.effect_min_speed_mps,
        max_interaction_edge_gap_m=args.effect_max_interaction_gap_m,
    )
    r = build_dataset(args.dataroot, args.version, args.limit, args.out_dir,
                      samples_per_scene=args.samples_per_scene, with_frames=args.with_frames,
                      twin_model=args.twin_model, epsilon_max=eps_max,
                      selections_csv=args.selections_csv, review_dir=args.review_dir,
                      track=args.track, review_manifest=args.review_manifest,
                      admission_thresholds=admission_thresholds,
                      visibility_policy=args.visibility_policy)
    f = r["fidelity"]
    print(f"accepted={r['accepted']} rejected={r['rejected']} -> {args.out_dir}")
    if f["candidates"]:
        print(f"fidelity: candidates={f['candidates']} admitted={f['admitted']} "
              f"median={f['median']:.2f} mean={f['mean']:.2f} (model={args.twin_model})")
    else:
        print(f"fidelity: candidates=0 admitted=0 (model={args.twin_model})")
    e = r["effect_gate"]
    print(f"canonical-twin effect gate: candidates={e['candidates']} "
          f"passed={e['passed']} rejected={e['rejected']}")


if __name__ == "__main__":
    main()
