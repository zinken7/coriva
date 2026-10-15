"""
Batch orchestrator: scenario configs → simulated telemetry → GT → QC → files.

For every config: run the simulator (injected `scenario_fn` — the real CARLA
runner on a server, a fake in tests), build the ground truth, apply the QC
gate, and write `out_root/<output_subdir>/ground_truth.json` plus the raw
telemetry. Resumable: scenarios whose GT already exists are skipped, so a
spot-instance kill costs minutes, not the batch. Every scenario appends one
line to `out_root/batch_log.jsonl` (accepted / rejected / error + reasons),
so a run is auditable and rejects are never silent.

Server usage (vast.ai, CARLA running on localhost:2000):
    python3 -m simulation.run_batch --scenarios-dir scenarios/generated \
        --out-root output --host localhost --port 2000
"""

import argparse
import copy
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
import math
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctms.ctms import load_ctms_environment  # noqa: E402
from simulation.build_ground_truth import (  # noqa: E402
    ReferenceBuildError, build_scenario_gt, lane_evidence_gaps, static_pass_through)
from simulation.generate import ScenarioCandidateExhausted  # noqa: E402
from simulation.qc import qc_scenario_gt  # noqa: E402
from simulation.scenario_config import load_scenario_configs  # noqa: E402
from simulation.protocol import (OBSERVED_FRAME_COUNT, PROTOCOL_VERSION,
                                 lateral_action_conflict,
                                 protocol_manifest, track_steps)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OVERLAY_FONT_PATH = os.path.join(
    PROJECT_ROOT, "assets", "fonts", "NotoSans-Regular.ttf")
PUBLISHED_BASE_FILES = (
    "ground_truth.json",
    "raw_telemetry.json",
    "scenario_manifest.json",
    "scenario_config.json",
)
PUBLISHED_PROTOCOL_FILES = (
    "observed_state.json",
    "observed_history.json",
    "query.json",
    "ontology.json",
    "input_manifest.json",
)


def _overlay_carla_frames(scen_dir: str, config: Dict[str, Any], raw: Dict[str, Any],
                          root_actor_ids: Optional[set] = None) -> None:
    """Add deterministic actor IDs to the eight CARLA model frames."""
    from PIL import Image, ImageDraw, ImageFont
    import glob

    factual = next((name for name, b in config.get("branches", {}).items()
                    if b.get("is_factual")), None)
    if factual is None:
        return
    steps = raw.get("branches", {}).get(factual, [])
    if not steps:
        return
    t0 = float(config.get("intervention_time_sec", 0.0))
    frames = sorted(glob.glob(os.path.join(scen_dir, "frames", "*.png")))
    raw_dir = os.path.join(scen_dir, "raw_frames")
    os.makedirs(raw_dir, exist_ok=True)
    # Use the repository-bundled OFL font on every platform.  Falling back to
    # an OS font would make the released overlays host-dependent.
    font = ImageFont.truetype(OVERLAY_FONT_PATH, 18)
    ego_key = next((k for k, v in config["actor_defs"].items()
                    if v.get("canonical_id") == "ego"), "ego")
    offsets = [-3.5 + 0.5 * i for i in range(8)]
    for index, frame in enumerate(frames):
        image = Image.open(frame).convert("RGB")
        image.save(os.path.join(raw_dir, os.path.basename(frame)), quality=95)
        step = min(steps, key=lambda s: abs(float(s.get("time_sec", 0.0)) - (t0 + offsets[index])))
        ego = step.get(ego_key, {})
        ex, ey = float(ego.get("x", 0.0)), float(ego.get("y", 0.0))
        yaw = (math.radians(float(ego["heading_deg"]))
               if ego.get("heading_deg") is not None
               else math.atan2(float(ego.get("vy", 0.0)), float(ego.get("vx", 0.0))))
        c, s = math.cos(yaw), math.sin(yaw)
        width, height = image.size
        focal = width / (2.0 * math.tan(math.radians(100.0) / 2.0))
        draw = ImageDraw.Draw(image)
        for key, actor_def in config["actor_defs"].items():
            if key == ego_key or key not in step:
                continue
            actor_id = str(actor_def.get("canonical_id", key))
            if root_actor_ids is not None and actor_id not in root_actor_ids:
                continue
            actor = step[key]
            projected = actor.get("bbox_2d")
            if isinstance(projected, (list, tuple)) and len(projected) == 4:
                x1, y1, x2, y2 = (int(round(float(value))) for value in projected)
                if x2 <= x1 or y2 <= y1:
                    continue
                box = (x1, y1, x2, y2)
                label_xy = (min(width - 2, x2 + 4), max(2, y1))
            else:
                # Telemetry without bbox_2d.  Protocol-1.0 real CARLA runs
                # always carry bbox_2d from TrajectoryRecorder.
                dx, dy = float(actor.get("x", 0.0)) - ex, float(actor.get("y", 0.0)) - ey
                forward = c * dx + s * dy + 1.5
                lateral = -s * dx + c * dy
                if forward <= 0.5 or abs(math.degrees(math.atan2(lateral, forward))) > 50.0:
                    continue
                u = int(round(width / 2.0 + focal * lateral / forward))
                v = int(round(height * 0.55))
                if not (2 <= u < width - 2):
                    continue
                box = (u - 18, v - 18, u + 18, v + 18)
                label_xy = (u + 20, v - 16)
            draw.rectangle(box, outline=(255, 32, 32), width=3)
            draw.text(label_xy, actor_id, fill=(255, 32, 32), font=font,
                      stroke_width=1, stroke_fill=(255, 255, 255))
        image.save(frame)


def _log(out_root: str, entry: Dict[str, Any]) -> None:
    with open(os.path.join(out_root, "batch_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _write_rejected_diagnostics(
        out_root: str, config: Dict[str, Any], status: str, reasons: Any,
        raw: Optional[Dict[str, Any]] = None,
        gt: Optional[Dict[str, Any]] = None) -> str:
    """Persist compact JSON evidence for a rejected/error attempt.

    Rendered branch frames stay in ``.attempts`` and are deleted by the normal
    cleanup path. Telemetry is gzip-compressed because it is the largest useful
    diagnostic artifact.
    """
    attempt_seed = int(config.get("attempt_seed", config.get("seed", 0)))
    relative = os.path.join(
        ".rejected", config["output_subdir"], f"seed_{attempt_seed}")
    diagnostic_dir = os.path.join(out_root, relative)
    if os.path.isdir(diagnostic_dir):
        shutil.rmtree(diagnostic_dir)
    os.makedirs(diagnostic_dir, exist_ok=True)

    with open(os.path.join(diagnostic_dir, "scenario_config.json"), "w",
              encoding="utf-8") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)
    placement = copy.deepcopy((raw or {}).get("_placement"))
    route = copy.deepcopy((gt or {}).get("scene_geometry"))
    if not isinstance(route, dict):
        route = copy.deepcopy((raw or {}).get("_route_evidence"))
    if not isinstance(route, dict):
        route = {}
    coverage_complete = route.get("route_coverage_complete")
    if coverage_complete is None:
        coverage = route.get("route_coverage_m")
        required = route.get("route_required_m")
        if coverage is not None and required is not None:
            coverage_complete = float(coverage) >= float(required)
    coverage_status = (
        "complete" if coverage_complete is True
        else "incomplete" if coverage_complete is False
        else "unavailable"
    )
    backward_complete = route.get("route_backward_coverage_complete")
    if backward_complete is None:
        backward_coverage = route.get("route_backward_coverage_m")
        backward_required = route.get("route_backward_required_m")
        if backward_coverage is not None and backward_required is not None:
            backward_complete = (
                float(backward_coverage) >= float(backward_required))
    backward_status = (
        "complete" if backward_complete is True
        else "incomplete" if backward_complete is False
        else "unavailable"
    )
    summary = {
        "scenario_id": config.get("scenario_id"),
        "slot_id": config.get("slot_id"),
        "template_id": config.get("template_id"),
        "attempt_seed": config.get("attempt_seed"),
        "status": status,
        "reasons": reasons,
        "runtime_placement": placement,
        "route_evidence": {
            "route_source": route.get("source"),
            "route_mode": route.get("route_mode"),
            "route_id": route.get("route_id"),
            "coverage_status": coverage_status,
            "backward_coverage_status": backward_status,
            "route_coverage_m": route.get("route_coverage_m"),
            "route_required_m": route.get("route_required_m"),
            "route_backward_coverage_m": route.get(
                "route_backward_coverage_m"),
            "route_backward_required_m": route.get(
                "route_backward_required_m"),
            "route_sample_spacing_m": route.get("sample_spacing_m"),
            "candidate_number": (
                placement.get("candidate_number")
                if isinstance(placement, dict) else None),
        },
    }
    with open(os.path.join(diagnostic_dir, "qc_reasons.json"), "w",
              encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    if raw is not None:
        with gzip.open(os.path.join(diagnostic_dir, "raw_telemetry.json.gz"),
                       "wt", encoding="utf-8") as stream:
            json.dump(raw, stream, ensure_ascii=False, separators=(",", ":"))
    if gt is not None:
        with open(os.path.join(diagnostic_dir, "ground_truth.json"), "w",
                  encoding="utf-8") as stream:
            json.dump(gt, stream, ensure_ascii=False, indent=2)
        if gt.get("scene_geometry") is not None:
            with open(os.path.join(diagnostic_dir, "route_geometry.json"), "w",
                      encoding="utf-8") as stream:
                json.dump(gt["scene_geometry"], stream,
                          ensure_ascii=False, indent=2)
    return relative


def _config_hash(config: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _scenario_manifest(config: Dict[str, Any], gt: Dict[str, Any],
                       admission: Optional[Dict[str, Any]],
                       runtime_placement: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "scenario_id": config["scenario_id"],
        "slot_id": config.get("slot_id"),
        "template_id": config.get("template_id", config.get("scenario_type")),
        "scenario_type": config.get("scenario_type"),
        "attempt_seed": config.get("attempt_seed"),
        "simulation_seed": config.get("seed"),
        "config_sha256": _config_hash(config),
        "map": config.get("map"),
        "weather": (config.get("weather") or {}).get("preset"),
        "severity_stratum": config.get("severity_stratum"),
        "target_ttc_sec": config.get("target_ttc_sec"),
        "expected_t0_critical_gap_m": config.get("expected_t0_critical_gap_m"),
        "expected_t0_critical_longitudinal_m": config.get(
            "expected_t0_critical_longitudinal_m"),
        "camera": copy.deepcopy(config.get("camera")),
        "critical_actor_id": config.get("critical_actor_id"),
        "critical_actor_type": (config.get("actor_defs", {})
                                .get(config.get("critical_actor_id"), {})
                                .get("entity_type")),
        "actor_blueprints": {
            str(spec.get("canonical_id", key)): spec.get("model")
            for key, spec in config.get("actor_defs", {}).items()
        },
        "actor_entities": {
            str(spec.get("canonical_id", key)): spec.get("entity_type")
            for key, spec in config.get("actor_defs", {}).items()
        },
        "simulated_actor_count": len(config.get("actor_defs", {})),
        "root_actor_count": len((gt.get("observed_state") or {}).get("actors", [])),
        "runtime_placement": copy.deepcopy(runtime_placement),
        "protocol": copy.deepcopy(gt.get("protocol")),
        "admission": copy.deepcopy(admission),
    }


def _published_scenario_complete(scenario_dir: str, protocol: bool) -> bool:
    """Return whether a scenario has the complete current release contract."""
    required = PUBLISHED_BASE_FILES + (PUBLISHED_PROTOCOL_FILES if protocol else ())
    return all(os.path.isfile(os.path.join(scenario_dir, name)) for name in required)


def _write_dataset_manifest(out_root: str) -> None:
    import glob
    manifests = []
    for path in sorted(glob.glob(os.path.join(out_root, "scenario_*", "scenario_manifest.json"))):
        with open(path, encoding="utf-8") as stream:
            manifests.append(json.load(stream))
    distributions = {}
    for field in ("template_id", "critical_actor_type", "map", "weather",
                  "severity_stratum", "root_actor_count"):
        distributions[field] = dict(sorted(Counter(
            str(item.get(field)) for item in manifests).items()))
    rejected = Counter()
    log_path = os.path.join(out_root, "batch_log.jsonl")
    if os.path.isfile(log_path):
        with open(log_path, encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("status") != "accepted":
                    rejected[str(row.get("status", "unknown"))] += 1
    payload = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "accepted_scenarios": len(manifests),
        "distributions": distributions,
        "rejected_attempts_by_status": dict(sorted(rejected.items())),
        "scenario_ids": [item["scenario_id"] for item in manifests],
    }
    with open(os.path.join(out_root, "dataset_manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def _extract_observed_frames(scen_dir: str, config: Dict[str, Any],
                             fps: int = 10, protocol: bool = False) -> int:
    """Populate <scen_dir>/frames/ — the OBSERVED segment the VLM is allowed
    to see — from the factual branch's recorder PNGs.

    The recorder writes each branch's frames into <scen_dir>/<branch>/ with
    6-digit CARLA frame numbers (chronological when sorted) covering the FULL
    rollout. Non-protocol mode keeps frames strictly before intervention_time_sec;
    protocol mode includes the pre-action T0 camera image and samples the fixed
    -3.5..0.0 s window. Frames are renamed 000.png.. so run_model's glob+sort
    stays chronological regardless of CARLA's global frame counter.
    Returns the number of frames written (0 if the branch has no PNGs, e.g.
    telemetry-only runs)."""
    import glob as _glob
    import shutil

    factual = next((name for name, b in config.get("branches", {}).items()
                    if b.get("is_factual")), None)
    if factual is None:
        return 0
    src_pngs = sorted(_glob.glob(os.path.join(scen_dir, factual, "*.png")))
    if not src_pngs:
        return 0
    n_observed = int(float(config.get("intervention_time_sec", 0.0)) * fps)
    # The protocol path includes the initial camera image, so an intervention
    # at 3.5 s yields indices 0..35 and can end exactly at T0. Non-protocol
    # exports keep the strict-pre-intervention slice.
    pre = src_pngs[:n_observed + 1] if protocol else src_pngs[:n_observed]
    if protocol:
        # The protocol is defined at 2 Hz even when CARLA records at 10 Hz.
        # Take eight equally spaced images ending at the T0 camera image.
        stride = max(1, round(fps * 0.5))
        start = len(pre) - 1 - (OBSERVED_FRAME_COUNT - 1) * stride
        selected = ([pre[start + i * stride] for i in range(OBSERVED_FRAME_COUNT)]
                    if start >= 0 else pre)
    else:
        selected = pre
    if not selected:
        return 0
    frames_dir = os.path.join(scen_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i, src in enumerate(selected):
        shutil.copy(src, os.path.join(frames_dir, f"{i:03d}.png"))
    return len(selected)


def run_batch(configs: List[Dict[str, Any]],
              scenario_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
              out_root: str,
              env: Optional[Dict[str, Any]] = None,
              resume: bool = True,
              track: Optional[str] = None) -> Dict[str, int]:
    """Process every config through simulate → GT → QC → write. Returns
    {accepted, failed, skipped}. One bad scenario never stops the batch."""
    if env is None:
        env = load_ctms_environment("ctms/ontology.yaml", "ctms/ctms_config.yaml",
                                    "ctms/schema_branch.json", "ctms/schema_counterfactual.json")
    os.makedirs(out_root, exist_ok=True)
    accepted = failed = skipped = 0

    for config in configs:
        sid = config["scenario_id"]
        final_scen_dir = os.path.join(out_root, config["output_subdir"])
        if resume and _published_scenario_complete(
                final_scen_dir, protocol=track is not None):
            skipped += 1
            continue

        attempt_rel = os.path.join(
            ".attempts", config["output_subdir"],
            f"seed_{int(config.get('attempt_seed', config.get('seed', 0)))}")
        scen_dir = os.path.join(out_root, attempt_rel)
        if os.path.isdir(scen_dir):
            shutil.rmtree(scen_dir)
        run_config = copy.deepcopy(config)
        run_config["output_subdir"] = attempt_rel
        gt_path = os.path.join(scen_dir, "ground_truth.json")

        def reject(status: str, reasons: Any, started: float,
                   extra: Optional[Dict[str, Any]] = None,
                   diagnostic_raw: Optional[Dict[str, Any]] = None,
                   diagnostic_gt: Optional[Dict[str, Any]] = None) -> None:
            entry = {
                "scenario_id": sid, "slot_id": config.get("slot_id"),
                "template_id": config.get("template_id"),
                "attempt_seed": config.get("attempt_seed"), "status": status,
                "reasons": reasons, "duration_sec": round(time.time() - started, 1),
            }
            if extra:
                entry.update(extra)
            try:
                entry["diagnostic_path"] = _write_rejected_diagnostics(
                    out_root, config, status, reasons,
                    raw=diagnostic_raw, gt=diagnostic_gt)
            except Exception as exc:
                entry["diagnostic_error"] = f"{type(exc).__name__}: {exc}"
            _log(out_root, entry)
            if os.path.isdir(scen_dir):
                shutil.rmtree(scen_dir)

        t0 = time.time()
        raw: Optional[Dict[str, Any]] = None
        gt: Optional[Dict[str, Any]] = None
        try:
            raw = scenario_fn(run_config)
            raw.pop("_controllers", None)
            hits = static_pass_through(raw, run_config.get("actor_defs", {}))
            if hits:
                first = hits[0]
                raise ReferenceBuildError(
                    f"moving actor {first['mover']} passes through static "
                    f"{first['static']} at {first['time_sec']:.1f}s in {first['branch']} "
                    f"(centre distance {first['distance_m']:.2f} m; {len(hits)} step(s))")
            if track is not None and raw.get("_placement") is not None:
                # Protocol runs: the placement's lane evidence must cover
                # what every rollout actually drove, or release would refuse
                # the scenario later; reject-and-resample now instead.
                gaps = lane_evidence_gaps(raw, run_config.get("actor_defs", {}))
                if gaps:
                    raise ReferenceBuildError("lane evidence does not cover the rollouts: "
                                              + "; ".join(gaps[:2]))
            gt = build_scenario_gt(raw, run_config, track=track or "primary",
                                   protocol=track is not None)
            # Protocol runs only: the lateral escape is declared from the
            # template's route offset before the rollout, and on a curve the
            # realised T0 offset can be on the other side. Reject-and-resample
            # rather than publish a scene whose only lateral alternative steers
            # into the critical actor. Non-protocol fixtures have roots in world
            # coordinates and are not held to the protocol-1.0 rule.
            if track is not None:
                conflict = lateral_action_conflict(
                    gt.get("observed_state"), run_config.get("critical_actor_id"),
                    [b["intervention"]["intervention_type"] for b in gt.get("branches", [])])
                if conflict:
                    raise ReferenceBuildError(conflict)
            if track is not None:
                required = track_steps(track)
                short = [b.get("branch_id") for b in gt.get("branches", [])
                         if len(b.get("chain", [])) != required
                         and not (b.get("chain") and b["chain"][-1].get("terminal"))]
                for query in gt.get("l4_queries", []):
                    chain = (query.get("counterfactual_chain") or {}).get("chain", [])
                    if len(chain) != required and not (chain and chain[-1].get("terminal")):
                        short.append("L4_fixed_policy")
                if short:
                    raise ValueError(
                        f"{track} requires {required} future nodes; short branches={short}")
            ok, reasons = qc_scenario_gt(gt, env["ontology"], env["config"], env["branch_schema"])
        except ScenarioCandidateExhausted as e:
            # Placement/visibility exhaustion is deterministic for one
            # generated sample. It is a normal reject-and-resample outcome,
            # not an infrastructure failure worth replaying at the same seed.
            failed += 1
            reject("rejected_placement", [f"{type(e).__name__}: {e}"], t0,
                   diagnostic_raw=raw, diagnostic_gt=gt)
            continue
        except ReferenceBuildError as e:
            # The rollout ran, but it cannot be a valid declared reference
            # (e.g. an in-horizon collision with an actor outside the root).
            failed += 1
            reject("rejected_reference", [str(e)], t0,
                   diagnostic_raw=raw, diagnostic_gt=gt)
            continue
        except Exception as e:  # CARLA hiccup, malformed telemetry, ...
            failed += 1
            reject("error", [f"{type(e).__name__}: {e}"], t0,
                   diagnostic_raw=raw, diagnostic_gt=gt)
            continue

        if not ok:
            failed += 1
            reject("rejected_qc", reasons, t0,
                   diagnostic_raw=raw, diagnostic_gt=gt)
            continue

        try:
            admission = None
            if track is not None and config.get("template_id"):
                from simulation.carla_admission import evaluate_carla_admission
                admission_ok, admission = evaluate_carla_admission(raw, run_config, gt)
                if not admission_ok:
                    failed += 1
                    reject("rejected_admission", admission["reasons"], t0,
                           {"admission": admission},
                           diagnostic_raw=raw, diagnostic_gt=gt)
                    continue

            n_frames = _extract_observed_frames(
                scen_dir, run_config, protocol=track is not None)
            if track is not None and n_frames != OBSERVED_FRAME_COUNT:
                failed += 1
                reject("rejected_protocol", [
                    f"observed_frame_count={n_frames}, expected {OBSERVED_FRAME_COUNT}"],
                    t0, {"observed_frames": n_frames},
                    diagnostic_raw=raw, diagnostic_gt=gt)
                continue

            if track is not None:
                _overlay_carla_frames(
                    scen_dir, run_config, raw,
                    {str(actor.get("id"))
                     for actor in gt["observed_state"].get("actors", [])})
            observed_history = gt.pop("_observed_history", None)
            os.makedirs(scen_dir, exist_ok=True)
            with open(gt_path, "w", encoding="utf-8") as f:
                json.dump(gt, f, indent=2)
            with open(os.path.join(scen_dir, "raw_telemetry.json"),
                      "w", encoding="utf-8") as f:
                json.dump(raw, f)
            if track is not None:
                if observed_history is None:
                    raise RuntimeError("protocol GT did not include observed history")
                from simulation.observed_history import (write_observed_artifacts,
                                                           write_protocol_sidecars)
                write_observed_artifacts(scen_dir, gt["observed_state"], observed_history)
                write_protocol_sidecars(scen_dir, gt, env["ontology"])
                protocol_meta = gt.get("protocol", {})
                with open(os.path.join(scen_dir, "input_manifest.json"),
                          "w", encoding="utf-8") as f:
                    json.dump(protocol_manifest(
                        track=track,
                        condition="video",
                        split="carla",
                        critical_actor_id=protocol_meta.get("critical_actor_id"),
                        intervention_type=protocol_meta.get("intervention_type"),
                        l4_pair_protocol=protocol_meta.get("l4_pair_protocol"),
                        l4_control_reference_source=protocol_meta.get("l4_control_reference_source"),
                        l4_ego_policy=protocol_meta.get("l4_ego_policy"),
                        dataset="carla", coordinate_frame="t0_ego",
                        frame_asset="frames", raw_frame_asset="raw_frames",
                        observed_state_asset="observed_state.json",
                        observed_history_asset="observed_history.json",
                        ontology_asset="ontology.json", query_asset="query.json",
                        overlay_version="coriva-id-overlay-2",
                        camera=copy.deepcopy(config.get("camera"))), f, indent=2)
            with open(os.path.join(scen_dir, "scenario_manifest.json"),
                      "w", encoding="utf-8") as f:
                json.dump(_scenario_manifest(
                    config, gt, admission, raw.get("_placement")), f, indent=2)
            # Store the exact accepted config beside the artifacts it created.
            # The manifest hash can therefore be verified without relying on
            # an external generated_seed* directory remaining available.
            with open(os.path.join(scen_dir, "scenario_config.json"),
                      "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            failed += 1
            reject("error", [f"{type(e).__name__}: {e}"], t0,
                   diagnostic_raw=raw, diagnostic_gt=gt)
            continue

        # Publish only a complete accepted attempt.  Any partial directory
        # without GT is generated scratch and cannot be mixed in.
        if os.path.isdir(final_scen_dir):
            shutil.rmtree(final_scen_dir)
        os.makedirs(os.path.dirname(final_scen_dir), exist_ok=True)
        os.replace(scen_dir, final_scen_dir)
        accepted += 1
        _log(out_root, {"scenario_id": sid, "slot_id": config.get("slot_id"),
                        "template_id": config.get("template_id"),
                        "attempt_seed": config.get("attempt_seed"),
                        "status": "accepted", "reasons": [],
                        "observed_frames": n_frames,
                        "output_path": final_scen_dir,
                        "duration_sec": round(time.time() - t0, 1)})

    _write_dataset_manifest(out_root)
    return {"accepted": accepted, "failed": failed, "skipped": skipped}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scenarios-dir", default="scenarios/generated")
    parser.add_argument("--out-root", default="output")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="process at most N configs (0 = all)")
    parser.add_argument("--track", choices=["primary", "challenge_6s"], default="primary")
    args = parser.parse_args()

    from simulation.generate import FPS, STEP_LENGTH, _default_api, run_scenario

    configs = load_scenario_configs(args.scenarios_dir)
    if args.limit:
        configs = configs[: args.limit]
    api = _default_api()
    World = api["World"]  # CORIVA wrapper around the official CARLA client

    state: Dict[str, Any] = {"world": None, "map": None}

    def scenario_fn(config: Dict[str, Any]) -> Dict[str, Any]:
        map_name = config.get("map", "Town04")
        if state["world"] is None:
            state["world"] = World(host=args.host, port=args.port, timeout=30.0,
                                   map_name=map_name, fps=FPS, step_length=STEP_LENGTH)
            state["world"].setup()
            state["map"] = map_name
        elif map_name != state["map"]:
            state["world"].change_map(map_name)
            state["map"] = map_name
        state["world"].apply_weather(
            (config.get("weather") or {}).get("preset", "ClearNoon"))
        run_config = dict(config)
        required_duration = (float(config["intervention_time_sec"])
                             + track_steps(args.track) * 0.5 + STEP_LENGTH)
        run_config["total_duration_sec"] = max(
            float(config.get("total_duration_sec", 0.0)), required_duration)
        return run_scenario(run_config, state["world"], api, output_dir=args.out_root)

    summary = run_batch(configs, scenario_fn, args.out_root,
                        resume=not args.no_resume, track=args.track)
    print(f"accepted={summary['accepted']} failed={summary['failed']} "
          f"skipped={summary['skipped']} -> {args.out_root}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
