#!/usr/bin/env python3
"""Find replacement T0s for fidelity-rejected scenes, then collect fresh review.

CPU only. Search never changes source approvals, admission thresholds or datasets.
"""
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.package_nuscenes_review import _package_files, MANIFEST, SELECTIONS
from simulation.build_nuscenes_dataset import _sidecar_keyframes, preview_reviewed_candidate
from simulation.nuscenes_adapter import load_tables
from simulation.nuscenes_admission import NuScenesAdmissionThresholds
from simulation.nuscenes_visibility import visibility_policy_manifest
from simulation.observed_clip import write_model_frames
from ctms.ctms import load_ctms_environment


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".recovery-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def select_failures(resolved, fidelity):
    rows = {r["scene_name"]: r for r in resolved["selections"]}
    if len(rows) != len(resolved["selections"]):
        raise ValueError("duplicate scene in build report")
    selected = []
    for measurement in fidelity["per_scenario"]:
        if measurement.get("admitted") is not False:
            continue
        row = rows[measurement["scene_name"]]
        if row.get("status") != "rejected" or row.get("sample_token") != measurement["source_sample_token"]:
            raise ValueError("fidelity and resolved reports disagree")
        if not any(isinstance(r, str) and r.startswith("fidelity_gate:") for r in row.get("reasons", [])):
            raise ValueError("fidelity rejection is missing from resolved report")
        selected.append({**row, "old_epsilon": measurement.get("epsilon")})
    return selected


def search_scene(frames, old_time, old_token, evaluate):
    attempts = []
    ordered = sorted(frames, key=lambda f: (abs(f["video_time_sec"] - old_time), f["video_time_sec"]))
    for frame in ordered:
        if frame["sample_token"] == old_token:
            continue
        result = evaluate(frame)
        # Large root/geometry objects are saved only for the winning proposal.
        attempts.append({**frame, **{k: v for k, v in result.items()
                                    if k not in ("root", "visibility", "effect", "sample_tokens")}})
        if result["passed"]:
            return {"status": "proposal_found", "attempts": attempts,
                    "proposal": {**frame, **result}}
    return {"status": "search_error" if any(a.get("metadata_error") for a in attempts) else "no_valid_t0",
            "attempts": attempts}


def load_context(build_dir, review_root, dataroot, version):
    reports = {name: read_json(build_dir / name) for name in (
        "resolved_selections.json", "fidelity_report.json", "nuscenes_effect_report.json")}
    resolved, fidelity, effect = reports.values()
    if resolved.get("track") != "primary" or resolved.get("dt_seconds") != 0.5 or resolved.get("future_steps") != 6:
        raise ValueError("recovery supports the primary 8-history/6-future protocol only")
    policy = resolved["visibility_policy"]["name"]
    if resolved["visibility_policy"] != visibility_policy_manifest(policy) or effect["visibility_policy"] != resolved["visibility_policy"]:
        raise ValueError("recorded visibility policy differs from current implementation")
    epsilon = fidelity["epsilon_max"]
    if not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("source build must have a finite positive fidelity threshold")
    if fidelity["twin_model"] not in ("bicycle", "point_mass"):
        raise ValueError("unsupported twin model")
    thresholds = NuScenesAdmissionThresholds(**effect["thresholds"])
    _package_files(review_root, False)
    approved = {r["scene_name"]: r for r in read_json(review_root / MANIFEST)["selections"]}
    for row in resolved["selections"]:
        source = approved[row["scene_name"]]
        if abs(float(source["t0_sec"]) - float(row["requested_t0_sec"])) > 1e-9:
            raise ValueError(f"source approved T0 changed: {row['scene_name']}")
        if row.get("critical_instance_token") and row["critical_instance_token"] != source["critical_instance_token"]:
            raise ValueError(f"source actor changed: {row['scene_name']}")
    failures = select_failures(resolved, fidelity)
    names = [r["scene_name"] for r in resolved["selections"]]
    input_paths = [build_dir / name for name in reports]
    input_paths += [review_root / MANIFEST, review_root / SELECTIONS]
    input_paths += [review_root / "videos" / f"{name}_CAM_FRONT.json" for name in names]
    code_paths = sorted((ROOT / "simulation").rglob("*.py")) + sorted((ROOT / "ctms").glob("*.py"))
    code_paths += sorted((ROOT / "ctms").glob("*.yaml")) + sorted((ROOT / "ctms").glob("schema*.json"))
    code_paths += [Path(__file__), ROOT / "scripts/package_nuscenes_review.py"]
    metadata = dataroot / version
    context = {
        "format": "nuscenes-t0-recovery-1", "dataroot": str(dataroot.resolve()), "version": version,
        "source_hashes": {str(p.resolve()): digest(p) for p in input_paths},
        "code_hashes": {str(p.relative_to(ROOT)): digest(p) for p in code_paths},
        "metadata_files": {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                           for p in sorted(metadata.glob("*.json"))},
        "policy": resolved["visibility_policy"], "epsilon_max": epsilon,
        "twin_model": fidelity["twin_model"], "thresholds": thresholds.to_dict(),
        "search_order": "nearest_requested_t0_then_earlier; fixed_actor_and_intervention",
        "max_actors": 5, "radius_m": 60.0,
        "failures": failures, "original_approved": approved,
        "accepted_names": [r["scene_name"] for r in resolved["selections"] if r["status"] == "accepted"],
        "ignored_rejected_names": [r["scene_name"] for r in resolved["selections"]
                                   if r["status"] == "rejected" and r["scene_name"] not in {f['scene_name'] for f in failures}],
    }
    return context


def render_proposal(tables, dataroot, output, proposal, policy):
    name = proposal["scene_name"]
    destination = output / "review" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        marker = read_json(destination / "render.json")
        if marker["proposal_id"] != proposal["proposal_id"]:
            raise ValueError(f"existing render has a different proposal: {name}")
        for rel, expected in marker["frame_hashes"].items():
            if digest(destination / rel) != expected:
                raise ValueError(f"review frame changed: {name}/{rel}")
        return marker
    with tempfile.TemporaryDirectory(prefix=".render-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "scene"
        write_model_frames(tables, proposal["sample_token"], str(dataroot), str(staged), proposal["root"],
                           visibility_policy=policy)
        # The portable review pack needs only the eight overlaid images.
        shutil.rmtree(staged / "raw_frames")
        marker = {"proposal_id": proposal["proposal_id"],
                  "frame_hashes": {f"frames/{i:03d}.jpg": digest(staged / "frames" / f"{i:03d}.jpg") for i in range(8)}}
        write_json(staged / "render.json", marker)
        staged.rename(destination)
    return marker


def publish_summary(output, context):
    states = [read_json(output / "scenes" / f"{row['scene_name']}.json") for row in context["failures"]]
    proposals = [s["proposal"] for s in states if s["status"] == "ready_for_review"]
    summary = {"preserved_accepted": len(context["accepted_names"]), "searched": len(states),
               "ready_for_review": len(proposals), "no_valid_t0": sum(s["status"] == "no_valid_t0" for s in states),
               "render_errors": sum(s["status"] == "render_error" for s in states),
               "search_errors": sum(s["status"] == "search_error" for s in states),
               "maximum_if_all_approved": len(context["accepted_names"]) + len(proposals)}
    report = {"context_id": object_digest(context), "summary": summary,
              "ignored_rejected_names": context["ignored_rejected_names"], "scenes": states}
    write_json(output / "report.json", report)
    # Preserve decisions across resume; append new proposals, never replace existing rows.
    decision_path = output / "review.csv"
    existing = {}
    if decision_path.exists():
        with decision_path.open(newline="", encoding="utf-8-sig") as handle:
            old_rows = list(csv.DictReader(handle))
        existing = {r["scene_name"]: r for r in old_rows}
        if len(existing) != len(old_rows):
            raise ValueError("duplicate review.csv scene")
        new_ids = {p["scene_name"]: p["proposal_id"] for p in proposals}
        if any(new_ids.get(n) != r["proposal_id"] for n, r in existing.items()):
            raise ValueError("review.csv belongs to a different proposal set")
    fields = ["scene_name", "proposal_id", "t0_sec", "decision", "notes"]
    with decision_path.open("a" if decision_path.exists() else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not existing and handle.tell() == 0:
            writer.writeheader()
        for p in proposals:
            if p["scene_name"] not in existing:
                writer.writerow(dict(scene_name=p["scene_name"], proposal_id=p["proposal_id"],
                                     t0_sec=p["video_time_sec"], decision="", notes=""))
    links = []
    for p in proposals:
        name = p["scene_name"]
        links.append(f'<li><a href="review/{name}/index.html">{name}</a>: {p["old_t0_sec"]:g} → {p["video_time_sec"]:g}s</li>')
        images = ''.join(f'<figure><figcaption>T={-3.5+i*0.5:g}s</figcaption><img src="frames/{i:03d}.jpg"></figure>' for i in range(8))
        page = f'''<!doctype html><meta charset="utf-8"><title>{name}</title>
<style>body{{font:16px sans-serif;max-width:1100px;margin:24px auto}}img{{max-width:100%}}figure{{margin:20px 0}}</style>
<h1>{name} — T0={p['video_time_sec']:g}s</h1>
<p>Actor: <b>{html.escape(p['critical_actor_id'])}</b>; intervention: <b>{html.escape(p['intervention_type'])}</b>.</p>
<p>Đề xuất tự động, chưa được duyệt. Xem đủ 8 frame: đúng actor, đủ rõ và intervention hợp lý trong 3 giây.
Trong review.csv điền decision=yes hoặc no; nếu yes, ghi lý do vào notes.</p>
<p>T0 cũ: {p['old_t0_sec']:g}s; fidelity mới: {p['epsilon']:.3f}m. Kiểm tra hình học không đo che khuất.</p>{images}'''
        (output / "review" / name / "index.html").write_text(page, encoding="utf-8")
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>T0 recovery review</title>'
        '<h1>Review T0 thay thế</h1><p>Điền review.csv sau khi xem đủ tám ảnh của từng đề xuất.</p><ul>'
        + ''.join(links) + '</ul>', encoding="utf-8")
    return report


def run_search(args):
    build_dir = args.build_dir.resolve()
    review_root = (args.review_root.resolve() if args.review_root else
                   Path(read_json(build_dir / "resolved_selections.json")["review_manifest"]).parent)
    output, dataroot = args.output.absolute(), args.dataroot.resolve()
    context = load_context(build_dir, review_root, dataroot, args.version)
    if output.exists() or output.is_symlink():
        if not args.resume or read_json(output / "context.json") != context:
            raise ValueError("output exists or inputs/code changed; use a new output (or --resume with unchanged inputs)")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".recovery-init-", dir=output.parent) as temporary:
            staged = Path(temporary) / "package"
            (staged / "source_review/videos").mkdir(parents=True)
            for filename in (MANIFEST, SELECTIONS):
                shutil.copy2(review_root / filename, staged / "source_review" / filename)
            for name in context["original_approved"]:
                shutil.copy2(review_root / "videos" / f"{name}_CAM_FRONT.json", staged / "source_review/videos" / f"{name}_CAM_FRONT.json")
            write_json(staged / "context.json", context)
            staged.rename(output)
    print(f"Loading metadata; searching {len(context['failures'])} fidelity rejections; preserving {len(context['accepted_names'])} accepted.", flush=True)
    tables = load_tables(str(dataroot), args.version)
    env = load_ctms_environment(*(str(ROOT / "ctms" / name) for name in (
        "ontology.yaml", "ctms_config.yaml", "schema_branch.json", "schema_counterfactual.json")))
    thresholds = NuScenesAdmissionThresholds(**context["thresholds"])
    for index, old in enumerate(context["failures"], 1):
        name = old["scene_name"]
        state_path = output / "scenes" / f"{name}.json"
        original = context["original_approved"][name]
        if state_path.exists():
            state = read_json(state_path)
        else:
            side_path = review_root / "videos" / f"{name}_CAM_FRONT.json"
            frames = _sidecar_keyframes(read_json(side_path), str(side_path))
            scene_token = tables["sample"][old["sample_token"]]["scene_token"]
            if any(tables["sample"].get(f["sample_token"], {}).get("scene_token") != scene_token for f in frames):
                raise ValueError(f"sidecar includes unknown/different-scene tokens: {name}")
            def evaluate(frame):
                return preview_reviewed_candidate(tables, frame["sample_token"], original["critical_instance_token"],
                    original["intervention_type"], env, epsilon_max=context["epsilon_max"],
                    twin_model=context["twin_model"], admission_thresholds=thresholds,
                    visibility_policy=context["policy"]["name"])
            baseline = evaluate({"sample_token": old["sample_token"]})
            if (baseline.get("stage") != "fidelity" or baseline.get("epsilon") is None
                    or abs(baseline["epsilon"] - float(old["old_epsilon"])) > 1e-4):
                raise ValueError(f"{name}: cannot reproduce original fidelity rejection; check code, metadata and source build settings")
            state = search_scene(frames, old["requested_t0_sec"], old["sample_token"], evaluate)
            state.update(scene_name=name, original_t0_sec=old["requested_t0_sec"], original_epsilon=old["old_epsilon"])
            if state.get("proposal"):
                p = state["proposal"]
                p.update(scene_name=name, old_t0_sec=old["requested_t0_sec"],
                         critical_instance_token=original["critical_instance_token"],
                         intervention_type=original["intervention_type"])
                p["proposal_id"] = object_digest(p)
            write_json(state_path, state)
        if state.get("proposal"):
            try:
                marker = render_proposal(tables, dataroot, output, state["proposal"], context["policy"]["name"])
                state.update(status="ready_for_review", render=marker)
                state.pop("render_error", None)
            except OSError as exc:
                state.update(status="render_error", render_error=str(exc))
            write_json(state_path, state)
        print(f"[{index}/{len(context['failures'])}] {name}: {state['status']}", flush=True)
    report = publish_summary(output, context)
    print(json.dumps(report["summary"]))
    print(f"Download this folder and open index.html: {output}")
    return 2 if report["summary"]["render_errors"] or report["summary"]["search_errors"] else 0


def read_decisions(path, proposals):
    expected = {p["scene_name"]: p for p in proposals}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"scene_name", "proposal_id", "t0_sec", "decision", "notes"}.issubset(reader.fieldnames or []):
            raise ValueError("review.csv missing required columns")
        rows = list(reader)
    found = {}
    for row in rows:
        name = row["scene_name"]
        if name not in expected or name in found:
            raise ValueError(f"unknown/duplicate review scene: {name}")
        p = expected[name]
        if row["proposal_id"] != p["proposal_id"] or float(row["t0_sec"]) != p["video_time_sec"]:
            raise ValueError(f"review proposal/T0 mismatch: {name}")
        row["decision"] = row["decision"].strip().lower()
        if row["decision"] not in ("yes", "no"):
            raise ValueError(f"{name}: decision must be yes or no (blank is not approval)")
        if row["decision"] == "yes" and not row["notes"].strip():
            raise ValueError(f"{name}: notes required for a new approval")
        found[name] = row
    if set(found) != set(expected):
        raise ValueError("review.csv does not contain every proposal")
    return found


def finalize(recovery, output, reviewer):
    recovery, output = Path(recovery), Path(output)
    if not reviewer.strip():
        raise ValueError("reviewer name is required")
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists; choose a new directory")
    context, report = read_json(recovery / "context.json"), read_json(recovery / "report.json")
    if report["context_id"] != object_digest(context):
        raise ValueError("recovery context changed")
    source = recovery / "source_review"
    _package_files(source, False)
    # Verify portable copies against the original input fingerprints.
    for path, expected_hash in context["source_hashes"].items():
        name = Path(path).name
        if name in (MANIFEST, SELECTIONS) or name.endswith("_CAM_FRONT.json"):
            local = source / ("videos" if name.endswith("_CAM_FRONT.json") else "") / name
            if digest(local) != expected_hash:
                raise ValueError(f"source artifact changed: {name}")
    proposals = [s["proposal"] for s in report["scenes"] if s["status"] == "ready_for_review"]
    for p in proposals:
        if object_digest({k: v for k, v in p.items() if k != "proposal_id"}) != p["proposal_id"]:
            raise ValueError("proposal was modified")
    decisions = read_decisions(recovery / "review.csv", proposals)
    approved = [context["original_approved"][name] for name in context["accepted_names"]]
    for p in proposals:
        name = p["scene_name"]
        if decisions[name]["decision"] != "yes":
            continue
        render = read_json(recovery / "review" / name / "render.json")
        if render["proposal_id"] != p["proposal_id"] or set(render["frame_hashes"]) != {f"frames/{i:03d}.jpg" for i in range(8)}:
            raise ValueError(f"render does not match proposal: {name}")
        for filename, expected in render["frame_hashes"].items():
            if digest(recovery / "review" / name / filename) != expected:
                raise ValueError(f"review image changed: {name}/{filename}")
        approved.append({"scene_name": name, "t0_sec": p["video_time_sec"],
                         "critical_instance_token": p["critical_instance_token"],
                         "intervention_type": p["intervention_type"], "audit_status": "approved",
                         "interaction_within_3s": True, "reviewer": reviewer.strip(),
                         "notes": decisions[name]["notes"].strip(), "recovery_proposal_id": p["proposal_id"],
                         "previous_t0_sec": p["old_t0_sec"]})
    names = [r["scene_name"] for r in approved]
    if not approved or len(set(names)) != len(names):
        raise ValueError("empty or duplicate final scene selection")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".finalize-", dir=output.parent) as temporary:
        staged = Path(temporary) / "package"
        (staged / "videos").mkdir(parents=True)
        manifest = read_json(source / MANIFEST)
        manifest["selections"] = approved
        write_json(staged / MANIFEST, manifest)
        with (staged / SELECTIONS).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["scene_name", "t0_sec"], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(approved)
        for name in names:
            shutil.copy2(source / "videos" / f"{name}_CAM_FRONT.json", staged / "videos" / f"{name}_CAM_FRONT.json")
        write_json(staged / "recovery_provenance.json", {
            "context_id": report["context_id"], "visibility_policy": context["policy"],
            "epsilon_max": context["epsilon_max"], "thresholds": context["thresholds"],
            "twin_model": context["twin_model"], "version": context["version"],
            "preserved": context["accepted_names"], "freshly_approved": names[len(context["accepted_names"]):],
            "reviewer": reviewer, "review_csv_sha256": digest(recovery / "review.csv")})
        if output.exists() or output.is_symlink():
            raise ValueError("output appeared during finalize")
        staged.rename(output)
    return len(approved)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search", help="find nearest passing T0s and render a portable review pack")
    search.add_argument("--build-dir", type=Path, required=True)
    search.add_argument("--review-root", type=Path,
                        help="approved review package; defaults to the manifest directory recorded by the source build")
    search.add_argument("--dataroot", type=Path, required=True)
    search.add_argument("--version", default="v1.0-trainval")
    search.add_argument("--output", type=Path, required=True)
    search.add_argument("--resume", action="store_true")
    finish = commands.add_parser("finalize", help="combine preserved scenes with explicitly reviewed replacements")
    finish.add_argument("--recovery", type=Path, required=True)
    finish.add_argument("--output", type=Path, required=True)
    finish.add_argument("--reviewer", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "search":
            return run_search(args)
        count = finalize(args.recovery, args.output, args.reviewer)
        print(f"Packaged {count} approved scenes -> {args.output}; run final builder into a new dataset directory.")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
