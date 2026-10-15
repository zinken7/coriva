#!/usr/bin/env python3
"""Coordinate CORIVA's two-pass nuScenes review across multiple reviewers.

The script keeps human work narrow:

1. reviewers watch assigned MP4s and record only ``scene_name,t0_sec``;
2. after the coordinator creates overlay v2 artifacts, reviewers inspect all
   eight history frames and choose one admissible candidate.

Canonical CSV/manifest files are produced only by merge commands.  Candidate
labels are rebound to the current overlay manifest by SHA-256, so stale labels
and reviewer-supplied instance tokens/intervention names cannot authorize a
selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import webbrowser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCHEMA_VERSION = 1
T0_ASSIGNMENT_STAGE = "t0_assignment"
T0_REVIEW_STAGE = "t0_review"
ACTOR_ASSIGNMENT_STAGE = "actor_assignment"
ACTOR_REVIEW_STAGE = "actor_review"
REVIEWER_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _number_text(value: float) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else format(number, ".15g")


def _write_minimal_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["scene_name", "t0_sec"])
        for row in rows:
            writer.writerow([row["scene_name"], _number_text(float(row["t0_sec"]))])
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_symlink(source: Path, destination: Path) -> bool:
    source = source.resolve()
    if not source.exists():
        raise ValueError(f"symlink source does not exist: {source}")
    if os.path.lexists(destination):
        if destination.is_symlink() and destination.resolve() == source:
            return False
        raise ValueError(f"dataroot destination already has different content: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=source.is_dir())
    return True


def prepare_trainval_dataroot(source_root: Path, out_root: Path) -> Dict[str, int]:
    """Assemble metadata and CAM_FRONT blob parts as a resumable symlink farm.

    The downloaded archives may be extracted into sibling directories such as
    ``v1.0-trainval_meta`` and ``v1.0-trainval01_blobs``.  CORIVA only consumes
    CAM_FRONT images, so linking those files avoids duplicating the full
    trainval payload while presenting the standard nuScenes dataroot layout.
    """
    source_root = Path(source_root)
    out_root = Path(out_root)
    metadata_roots = sorted(source_root.glob("v1.0-trainval_meta"))
    if len(metadata_roots) != 1:
        raise ValueError(
            f"expected exactly one v1.0-trainval_meta directory in {source_root}")
    metadata = metadata_roots[0]
    version_dir = metadata / "v1.0-trainval"
    maps_dir = metadata / "maps"
    if not (version_dir / "scene.json").is_file() or not maps_dir.is_dir():
        raise ValueError(f"incomplete trainval metadata directory: {metadata}")
    _ensure_symlink(version_dir, out_root / "v1.0-trainval")
    _ensure_symlink(maps_dir, out_root / "maps")

    blob_parts = sorted(source_root.glob("v1.0-trainval*_blobs"))
    if not blob_parts:
        raise ValueError(f"no v1.0-trainval*_blobs directories found in {source_root}")
    counts = {"samples": 0, "sweeps": 0}
    for part in blob_parts:
        for tree in ("samples", "sweeps"):
            camera_dir = part / tree / "CAM_FRONT"
            if not camera_dir.is_dir():
                continue
            for source in sorted(camera_dir.iterdir()):
                if not source.is_file():
                    continue
                _ensure_symlink(source, out_root / tree / "CAM_FRONT" / source.name)
                counts[tree] += 1
    if counts["samples"] == 0 or counts["sweeps"] == 0:
        raise ValueError("downloaded blob parts contain no complete CAM_FRONT samples/sweeps")
    return {
        "blob_parts": len(blob_parts),
        "sample_links": counts["samples"],
        "sweep_links": counts["sweeps"],
    }


def _validate_reviewer(reviewer: str) -> str:
    reviewer = str(reviewer).strip()
    if not reviewer or not REVIEWER_RE.fullmatch(reviewer):
        raise ValueError(
            "reviewer IDs may contain only letters, numbers, dot, underscore and dash")
    return reviewer


def discover_review_scenes(review_dir: Path) -> List[Dict[str, Any]]:
    """Return sorted, complete MP4/sidecar pairs from a review directory."""
    review_dir = Path(review_dir)
    scenes: List[Dict[str, Any]] = []
    for video_path in sorted(review_dir.glob("scene-*_CAM_FRONT.mp4")):
        scene_name = video_path.name.removesuffix("_CAM_FRONT.mp4")
        sidecar_path = review_dir / f"{scene_name}_CAM_FRONT.json"
        if not sidecar_path.is_file():
            raise ValueError(f"missing sidecar for {scene_name}: {sidecar_path}")
        sidecar = _load_json(sidecar_path)
        recorded_name = str(sidecar.get("scene", {}).get("name", ""))
        if recorded_name != scene_name:
            raise ValueError(
                f"sidecar scene mismatch for {scene_name}: recorded {recorded_name!r}")
        skipped = int(sidecar.get("video", {}).get("skipped_missing_source_frames", 0))
        if skipped != 0:
            raise ValueError(f"{scene_name} sidecar reports {skipped} missing source frames")
        scenes.append({
            "scene_name": scene_name,
            "video_path": str(video_path),
            "sidecar_path": str(sidecar_path),
            "duration_sec": float(sidecar.get("video", {}).get("duration_sec", 0.0)),
        })
    if not scenes:
        raise ValueError(f"no scene-*_CAM_FRONT.mp4 files found in {review_dir}")
    return scenes


def partition_assignments(
    scene_names: Sequence[str], reviewers: Sequence[str]
) -> Dict[str, List[str]]:
    """Partition scenes round-robin with no duplicated reviewer or scene."""
    reviewer_ids = [_validate_reviewer(value) for value in reviewers]
    if not reviewer_ids:
        raise ValueError("at least one reviewer is required")
    if len(set(reviewer_ids)) != len(reviewer_ids):
        raise ValueError("reviewer IDs must be unique")
    names = [str(value) for value in scene_names]
    if len(set(names)) != len(names):
        raise ValueError("scene assignments must be unique")
    result = {reviewer: [] for reviewer in reviewer_ids}
    for index, scene_name in enumerate(names):
        result[reviewer_ids[index % len(reviewer_ids)]].append(scene_name)
    return result


def validate_t0(sidecar_path: Path, t0_sec: float) -> Dict[str, Any]:
    """Snap T0 to the nearest keyframe and enforce the 8+6 primary window."""
    requested = float(t0_sec)
    if not math.isfinite(requested) or requested < 0.0:
        raise ValueError("t0_sec must be a finite non-negative number")
    sidecar = _load_json(Path(sidecar_path))
    keyframes = [
        frame for frame in sidecar.get("frames", [])
        if frame.get("is_key_frame") is True
    ]
    if not keyframes:
        raise ValueError(f"sidecar has no keyframes: {sidecar_path}")
    index = min(
        range(len(keyframes)),
        key=lambda idx: (abs(float(keyframes[idx]["video_time_sec"]) - requested), idx),
    )
    if index < 7:
        raise ValueError(
            "selection needs at least 8 history keyframes ending at T0")
    successors = len(keyframes) - index - 1
    if successors < 6:
        raise ValueError("selection needs at least 6 future keyframes after T0")
    resolved = float(keyframes[index]["video_time_sec"])
    return {
        "requested_t0_sec": requested,
        "resolved_t0_sec": resolved,
        "t0_snap_delta_sec": resolved - requested,
        "source_sample_token": keyframes[index].get("source_sample_token"),
        "history_keyframes": 8,
        "future_keyframes": 6,
    }


def _known_scene_names(path: Path) -> set[str]:
    """Read scene names from a canonical CSV or this helper's JSON artifacts."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return {
                str(row["scene_name"])
                for row in csv.DictReader(f)
                if row.get("scene_name")
            }
    value = _load_json(path)
    names: set[str] = set()
    for key in ("scenes", "decisions", "selections"):
        records = value.get(key, [])
        if not isinstance(records, list):
            continue
        for record in records:
            if isinstance(record, str):
                names.add(record)
            elif isinstance(record, dict) and record.get("scene_name"):
                names.add(str(record["scene_name"]))
    return names


def write_t0_assignments(
    review_dir: Path,
    reviewers: Sequence[str],
    out_dir: Path,
    *,
    target: Optional[int] = 260,
    exclude_paths: Sequence[Path] = (),
) -> List[Path]:
    scenes = discover_review_scenes(Path(review_dir))
    excluded: set[str] = set()
    for path in exclude_paths:
        excluded.update(_known_scene_names(Path(path)))
    names = [row["scene_name"] for row in scenes if row["scene_name"] not in excluded]
    if target is not None:
        if target <= 0:
            raise ValueError("target must be > 0")
        names = names[:target]
    assignments = partition_assignments(names, reviewers)
    outputs: List[Path] = []
    for reviewer, assigned in assignments.items():
        path = Path(out_dir) / f"t0_assignment_{reviewer}.json"
        _write_json_atomic(path, {
            "schema_version": SCHEMA_VERSION,
            "stage": T0_ASSIGNMENT_STAGE,
            "reviewer": reviewer,
            "scenes": [{"scene_name": scene_name} for scene_name in assigned],
        })
        outputs.append(path)
    return outputs


def _load_or_start_contribution(
    output_path: Path, stage: str, reviewer: str
) -> Dict[str, Any]:
    reviewer = _validate_reviewer(reviewer)
    if output_path.exists():
        value = _load_json(output_path)
        if value.get("schema_version") != SCHEMA_VERSION or value.get("stage") != stage:
            raise ValueError(f"incompatible contribution file: {output_path}")
        if value.get("reviewer") != reviewer:
            raise ValueError(f"reviewer mismatch in {output_path}")
        if not isinstance(value.get("decisions"), list):
            raise ValueError(f"decisions must be a list in {output_path}")
        return value
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "reviewer": reviewer,
        "decisions": [],
    }


def review_t0_interactive(
    assignment_path: Path,
    review_dir: Path,
    output_path: Path,
    *,
    open_video: bool = True,
) -> Dict[str, Any]:
    assignment = _load_json(Path(assignment_path))
    if assignment.get("schema_version") != SCHEMA_VERSION \
            or assignment.get("stage") != T0_ASSIGNMENT_STAGE:
        raise ValueError(f"not a T0 assignment: {assignment_path}")
    reviewer = _validate_reviewer(str(assignment.get("reviewer", "")))
    contribution = _load_or_start_contribution(
        Path(output_path), T0_REVIEW_STAGE, reviewer)
    decided = {str(row.get("scene_name")) for row in contribution["decisions"]}

    for assigned in assignment.get("scenes", []):
        scene_name = str(assigned.get("scene_name", ""))
        if not scene_name or scene_name in decided:
            continue
        video_path = Path(review_dir) / f"{scene_name}_CAM_FRONT.mp4"
        sidecar_path = Path(review_dir) / f"{scene_name}_CAM_FRONT.json"
        if not video_path.is_file() or not sidecar_path.is_file():
            raise ValueError(f"missing MP4/sidecar pair for {scene_name}")
        if open_video:
            webbrowser.open(video_path.resolve().as_uri())
        print(f"\n{scene_name}: {video_path}")
        while True:
            answer = input("T0 seconds, s=skip, q=save and quit: ").strip()
            if answer.lower() == "q":
                _write_json_atomic(Path(output_path), contribution)
                return contribution
            if answer.lower() == "s":
                contribution["decisions"].append({
                    "scene_name": scene_name,
                    "status": "skipped",
                })
                break
            try:
                preview = validate_t0(sidecar_path, float(answer))
            except (ValueError, TypeError) as exc:
                print(f"Invalid T0: {exc}")
                continue
            contribution["decisions"].append({
                "scene_name": scene_name,
                "status": "selected",
                "t0_sec": preview["requested_t0_sec"],
                "resolved_keyframe_preview_sec": preview["resolved_t0_sec"],
            })
            print(
                f"Selected {scene_name} T0={_number_text(preview['requested_t0_sec'])} "
                f"(nearest keyframe {preview['resolved_t0_sec']:.3f}s)")
            break
        _write_json_atomic(Path(output_path), contribution)
    return contribution


def merge_t0_contributions(
    contribution_paths: Sequence[Path], output_csv: Path
) -> List[Dict[str, Any]]:
    seen: Dict[str, str] = {}
    selected: List[Dict[str, Any]] = []
    for path in contribution_paths:
        value = _load_json(Path(path))
        if value.get("schema_version") != SCHEMA_VERSION \
                or value.get("stage") != T0_REVIEW_STAGE:
            raise ValueError(f"not a T0 review contribution: {path}")
        reviewer = _validate_reviewer(str(value.get("reviewer", "")))
        decisions = value.get("decisions")
        if not isinstance(decisions, list):
            raise ValueError(f"decisions must be a list: {path}")
        for decision in decisions:
            scene_name = str(decision.get("scene_name", ""))
            if not scene_name:
                raise ValueError(f"missing scene_name in {path}")
            if scene_name in seen:
                raise ValueError(
                    f"duplicate scene decision for {scene_name}: "
                    f"{seen[scene_name]} and {reviewer}")
            seen[scene_name] = reviewer
            status = decision.get("status")
            if status == "skipped":
                continue
            if status != "selected":
                raise ValueError(f"unknown T0 review status for {scene_name}: {status!r}")
            t0_sec = float(decision.get("t0_sec"))
            if not math.isfinite(t0_sec) or t0_sec < 0.0:
                raise ValueError(f"invalid t0_sec for {scene_name}: {t0_sec!r}")
            selected.append({"scene_name": scene_name, "t0_sec": t0_sec})
    selected.sort(key=lambda row: row["scene_name"])
    _write_minimal_csv(Path(output_csv), selected)
    return selected


def _load_overlay_index(overlay_dir: Path) -> Dict[str, Dict[str, Any]]:
    index_path = Path(overlay_dir) / "index.json"
    index = _load_json(index_path)
    valid = index.get("valid")
    if not isinstance(valid, list):
        raise ValueError(f"index.json valid must be a list: {index_path}")
    result: Dict[str, Dict[str, Any]] = {}
    for row in valid:
        scene_name = str(row.get("scene_name", ""))
        if not scene_name:
            raise ValueError(f"index row missing scene_name: {index_path}")
        if scene_name in result:
            raise ValueError(f"duplicate valid scene in overlay index: {scene_name}")
        result[scene_name] = row
    return result


def plan_overlay_resume(selections_csv: Path, out_dir: Path, *,
                        allow_empty_t0_changes: bool = False,
                        allow_empty_removals: bool = False) -> tuple[list[dict], list[dict]]:
    """Validate saved scenes using artifacts only; never require their raw images."""
    rows = _read_minimal_selections(selections_csv)
    by_name = {row["scene_name"]: row for row in rows}
    for name in by_name:
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"invalid scene_name: {name}")
    saved = _load_overlay_index(out_dir) if (out_dir / "index.json").is_file() else {}
    if os.path.lexists(out_dir) and not (out_dir / "index.json").is_file():
        raise ValueError(f"Existing overlay folder has no index.json: {out_dir}")
    changed = set()
    for name, row in saved.items():
        manifest = _load_json(out_dir / name / "overlay_manifest.json")
        removing = name not in by_name
        if removing and (not allow_empty_removals or row.get("admissible_stable_candidates") or manifest.get("admissible_stable_candidates")):
            raise ValueError(f"{name}: scene bị xóa khỏi CSV; chỉ được bỏ scene trước đó không có admissible candidate.")
        selection = manifest.get("selection", {})
        recorded = row.get("requested_t0_sec")
        if recorded is None or recorded != selection.get("requested_t0_sec"):
            raise ValueError(f"{name}: T0 differs between index and manifest")
        if not removing and not math.isclose(float(recorded), by_name[name]["t0_sec"], rel_tol=0, abs_tol=1e-9):
            if not allow_empty_t0_changes or row.get("admissible_stable_candidates") or manifest.get("admissible_stable_candidates"):
                raise ValueError(f"{name}: T0 đã thay đổi; chỉ được tạo lại scene trước đó không có admissible candidate.")
            changed.add(name)
        if row.get("sample_token") != manifest.get("sample_token"):
            raise ValueError(f"{name}: sample_token differs between index and manifest")
        for key in ("stable_candidates", "role_state_eligible_stable_candidates", "admissible_stable_candidates"):
            if row.get(key) != manifest.get(key):
                raise ValueError(f"{name}: {key} differs between index and manifest")
        for i in range(8):
            frame = out_dir / name / f"{i:03d}.jpg"
            if not frame.is_file() or frame.stat().st_size == 0:
                raise ValueError(f"{name}: thiếu ảnh overlay {frame}; cần khôi phục output trước khi chạy tiếp.")
    pending = [row for row in rows if row["scene_name"] not in saved or row["scene_name"] in changed]
    for row in pending:
        if row["scene_name"] not in changed and os.path.lexists(out_dir / row["scene_name"]):
            raise ValueError(f"Unindexed scene folder already exists: {out_dir / row['scene_name']}")
    return [saved[row["scene_name"]] for row in rows if row["scene_name"] in saved], pending


def make_overlay_round(
    dataroot: Path,
    review_dir: Path,
    selections_csv: Path,
    out_dir: Path,
    *,
    version: str = "v1.0-trainval",
    resume: bool = False,
    allow_empty_t0_changes: bool = False,
    allow_empty_removals: bool = False,
) -> Dict[str, int]:
    """Render a fresh round, or append new scenes while preserving saved reviews."""
    dataroot = Path(dataroot)
    review_dir = Path(review_dir)
    selections_csv = Path(selections_csv)
    out_dir = Path(out_dir)
    if os.path.lexists(out_dir) and not resume:
        raise ValueError(f"overlay output already exists; choose a new round: {out_dir}")

    valid, pending = plan_overlay_resume(selections_csv, out_dir,
        allow_empty_t0_changes=allow_empty_t0_changes, allow_empty_removals=allow_empty_removals)
    saved_names = {row['scene_name'] for row in valid}
    old_index = _load_json(out_dir / "index.json") if (out_dir / "index.json").is_file() else {}
    removed_records = list(old_index.get("removed", []))
    removals = [row for row in old_index.get("valid", []) if row["scene_name"] not in saved_names]
    removal_count = len(removals)
    rejected = list(old_index.get("rejected", []))

    def save_index():
        payload = {"valid": valid, "rejected": rejected}
        if removed_records:
            payload["removed"] = removed_records
        _write_json_atomic(out_dir / "index.json", payload)

    # Keep only hashes for synchronising the inspection copy; old images are temporary.
    if removals:
        with tempfile.TemporaryDirectory(prefix=".remove-", dir=out_dir) as temporary:
            moved = []
            try:
                for row in removals:
                    name = row["scene_name"]
                    source = out_dir / name
                    digest = file_sha256(source / "overlay_manifest.json")
                    frame_hashes = {f"{i:03d}.jpg": file_sha256(source / f"{i:03d}.jpg") for i in range(8)}
                    removed_records.append({**row, "removed_manifest_sha256": digest,
                        "frame_sha256": frame_hashes,
                        "reason": "removed_from_selection_csv_no_admissible_candidates"})
                    backup = Path(temporary) / name
                    source.rename(backup)
                    moved.append((source, backup))
                save_index()
            except OSError:
                for source, backup in reversed(moved):
                    backup.rename(source)
                raise
    changed_names = saved_names & {row['scene_name'] for row in pending}
    skipped = len(valid) - len(changed_names)
    created = rebuilt = 0
    if resume and not pending:
        return {"valid": skipped, "rejected": 0, "created": 0, "rebuilt": 0, "removed": removal_count, "skipped": skipped,
                "with_admissible_candidates": sum(bool(r["admissible_stable_candidates"]) for r in valid)}

    from simulation.build_nuscenes_dataset import _reviewed_candidates
    from simulation.nuscenes_adapter import load_tables
    from simulation.review_overlay import build_review_overlay

    tables = load_tables(str(dataroot), version)
    with tempfile.TemporaryDirectory(prefix="coriva-selections-") as temporary:
        pending_csv = Path(temporary) / "selections.csv"
        _write_minimal_csv(pending_csv, pending)
        rows = _reviewed_candidates(tables, str(pending_csv), str(review_dir))
    rejected = []
    out_dir.mkdir(parents=True, exist_ok=True)
    # Commit after each complete scene so a later render failure can be resumed.
    save_index()
    for row in rows:
        if row.get("error"):
            rejected.append(row)
            save_index()
            print(f"REJECT {row['scene_name']} {row['error']}")
            continue
        scene_out = out_dir / str(row["scene_name"])
        manifest_path = scene_out / "overlay_manifest.json"
        replacing = row["scene_name"] in changed_names
        previous_hash = file_sha256(manifest_path) if replacing else None
        previous_manifest = _load_json(manifest_path) if replacing else None
        with tempfile.TemporaryDirectory(prefix=".render-", dir=out_dir) as temporary:
            staged = Path(temporary) / "scene"
            overlay = build_review_overlay(
                tables, row["sample_token"], str(dataroot), str(staged), num_frames=8)
            overlay["selection"] = row
            if replacing:
                overlay["superseded_overlay_manifest_sha256s"] = list(dict.fromkeys(
                    previous_manifest.get("superseded_overlay_manifest_sha256s", []) + [previous_hash]))
                overlay["superseded_frame_sha256"] = {
                    **previous_manifest.get("superseded_frame_sha256", {}),
                    previous_hash: {f"{i:03d}.jpg": file_sha256(scene_out / f"{i:03d}.jpg") for i in range(8)}}
            _write_json_atomic(staged / "overlay_manifest.json", overlay)
            backup = Path(temporary) / "previous"
            previous_valid = valid
            if replacing:
                scene_out.rename(backup)
            try:
                staged.rename(scene_out)
                valid = [old for old in valid if old['scene_name'] != row['scene_name']]
                valid.append({
                    "scene_name": row["scene_name"],
                    "requested_t0_sec": row["requested_t0_sec"],
                    "resolved_t0_sec": row["resolved_t0_sec"],
                    "sample_token": row["sample_token"],
                    "stable_candidates": overlay["stable_candidates"],
                    "role_state_eligible_stable_candidates": overlay["role_state_eligible_stable_candidates"],
                    "admissible_stable_candidates": overlay["admissible_stable_candidates"],
                    "manifest": str(manifest_path.resolve()),
                })
                save_index()
            except OSError:
                valid = previous_valid
                if scene_out.exists() and not staged.exists():
                    scene_out.rename(staged)
                if replacing:
                    backup.rename(scene_out)
                raise
        if replacing:
            rebuilt += 1
        else:
            created += 1
        print(f"OK {row['scene_name']} {overlay['admissible_stable_candidates']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_index()
    print(f"valid {len(valid)} rejected {len(rejected)}")
    if rejected:
        raise ValueError(
            f"{len(rejected)} T0 selections are invalid; inspect {out_dir / 'index.json'}")
    return {
        "valid": len(valid),
        "created": created,
        "rebuilt": rebuilt,
        "removed": removal_count,
        "skipped": skipped,
        "rejected": len(rejected),
        "with_admissible_candidates": sum(
            1 for row in valid if row["admissible_stable_candidates"]),
    }


def _overlay_manifest_path(overlay_dir: Path, scene_name: str) -> Path:
    return Path(overlay_dir) / scene_name / "overlay_manifest.json"


def write_actor_assignments(
    overlay_dir: Path,
    reviewers: Sequence[str],
    out_dir: Path,
    *,
    exclude_paths: Sequence[Path] = (),
) -> List[Path]:
    index = _load_overlay_index(Path(overlay_dir))
    excluded: set[str] = set()
    for path in exclude_paths:
        excluded.update(_known_scene_names(Path(path)))
    scene_names = [
        scene_name for scene_name, row in sorted(index.items())
        if row.get("admissible_stable_candidates") and scene_name not in excluded
    ]
    assignments = partition_assignments(scene_names, reviewers)
    outputs: List[Path] = []
    for reviewer, assigned in assignments.items():
        scenes = []
        for scene_name in assigned:
            manifest_path = _overlay_manifest_path(Path(overlay_dir), scene_name)
            manifest = _load_json(manifest_path)
            candidates = list(manifest.get("admissible_stable_candidates", []))
            if not candidates:
                raise ValueError(f"no admissible candidates in current manifest: {scene_name}")
            scenes.append({
                "scene_name": scene_name,
                "admissible_stable_candidates": candidates,
                "overlay_manifest_sha256": file_sha256(manifest_path),
            })
        path = Path(out_dir) / f"actor_assignment_{reviewer}.json"
        _write_json_atomic(path, {
            "schema_version": SCHEMA_VERSION,
            "stage": ACTOR_ASSIGNMENT_STAGE,
            "reviewer": reviewer,
            "scenes": scenes,
        })
        outputs.append(path)
    return outputs


def _candidate_summary(manifest: Mapping[str, Any], candidate_id: str) -> Dict[str, Any]:
    candidate = manifest.get("candidates", {}).get(candidate_id)
    if not isinstance(candidate, dict):
        raise ValueError(f"candidate missing from overlay manifest: {candidate_id}")
    state = candidate.get("t0_state", {})
    effect = candidate.get("paired_canonical_twin_effect_gate") or {}
    return {
        "candidate": candidate_id,
        "entity": candidate.get("entity_type"),
        "role": candidate.get("role_type"),
        "intervention": candidate.get("recommended_intervention"),
        "x": state.get("x"),
        "y": state.get("y"),
        "vx": state.get("vx"),
        "vy": state.get("vy"),
        "min_gap": effect.get("min_treated_ego_target_edge_gap_m"),
        "collision_depth": effect.get("treated_collision_depth"),
    }


def write_actor_review_page(
    overlay_dir: Path, scene_name: str, pages_dir: Path
) -> Path:
    manifest_path = _overlay_manifest_path(Path(overlay_dir), scene_name)
    manifest = _load_json(manifest_path)
    allowed = list(manifest.get("admissible_stable_candidates", []))
    if not allowed:
        raise ValueError(f"no admissible candidates for {scene_name}")
    rows = []
    for candidate_id in allowed:
        summary = _candidate_summary(manifest, candidate_id)
        cells = "".join(
            f"<td>{html.escape(str(summary[key]))}</td>"
            for key in ("candidate", "entity", "role", "intervention", "x", "y",
                        "vx", "vy", "min_gap", "collision_depth")
        )
        rows.append(f"<tr>{cells}</tr>")
    images = []
    scene_dir = Path(overlay_dir) / scene_name
    for index in range(8):
        image_path = scene_dir / f"{index:03d}.jpg"
        if not image_path.is_file():
            raise ValueError(f"missing overlay history frame: {image_path}")
        relative_image = Path(os.path.relpath(
            image_path.resolve(), Path(pages_dir).resolve())).as_posix()
        images.append(
            f'<figure><img src="{html.escape(relative_image)}" '
            f'alt="{scene_name} frame {index:03d}"><figcaption>{index:03d}.jpg</figcaption></figure>')
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(scene_name)}</title>
<style>
body {{ font-family: sans-serif; margin: 24px; color: #171717; }}
.warning {{ padding: 12px; background: #fff4ce; border: 1px solid #d6a700; }}
.gallery {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
figure {{ margin: 0; }} img {{ width: 100%; height: auto; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
th, td {{ border: 1px solid #bbb; padding: 6px; text-align: left; }}
</style></head><body>
<h1>{html.escape(scene_name)} — actor review</h1>
<p class="warning">Review only the eight history frames 000–007. Do not use the recorded future to choose an intervention.</p>
<h2>Admissible candidates</h2>
<table><thead><tr><th>candidate</th><th>entity</th><th>role</th><th>intervention</th>
<th>x</th><th>y</th><th>vx</th><th>vy</th><th>min gap</th><th>collision depth</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Eight-frame history</h2><div class="gallery">{''.join(images)}</div>
</body></html>"""
    page_path = Path(pages_dir) / f"{scene_name}.html"
    _write_text_atomic(page_path, document)
    return page_path


def review_actors_interactive(
    assignment_path: Path,
    overlay_dir: Path,
    output_path: Path,
    pages_dir: Path,
    *,
    open_page: bool = True,
) -> Dict[str, Any]:
    assignment = _load_json(Path(assignment_path))
    if assignment.get("schema_version") != SCHEMA_VERSION \
            or assignment.get("stage") != ACTOR_ASSIGNMENT_STAGE:
        raise ValueError(f"not an actor assignment: {assignment_path}")
    reviewer = _validate_reviewer(str(assignment.get("reviewer", "")))
    contribution = _load_or_start_contribution(
        Path(output_path), ACTOR_REVIEW_STAGE, reviewer)
    decided = {str(row.get("scene_name")) for row in contribution["decisions"]}

    current_index = _load_overlay_index(Path(overlay_dir))
    for assigned in assignment.get("scenes", []):
        scene_name = str(assigned.get("scene_name", ""))
        if not scene_name or scene_name in decided:
            continue
        if scene_name not in current_index:
            raise ValueError(f"scene is not valid in current overlay index: {scene_name}")
        manifest_path = _overlay_manifest_path(Path(overlay_dir), scene_name)
        current_hash = file_sha256(manifest_path)
        if assigned.get("overlay_manifest_sha256") != current_hash:
            raise ValueError(f"stale actor assignment for {scene_name}; regenerate assignments")
        manifest = _load_json(manifest_path)
        allowed = list(manifest.get("admissible_stable_candidates", []))
        page_path = write_actor_review_page(Path(overlay_dir), scene_name, Path(pages_dir))
        if open_page:
            webbrowser.open(page_path.resolve().as_uri())
        print(f"\n{scene_name}: {page_path}")
        for candidate_id in allowed:
            print(json.dumps(_candidate_summary(manifest, candidate_id), ensure_ascii=False))
        while True:
            answer = input("Candidate label, s=skip, q=save and quit: ").strip()
            if answer.lower() == "q":
                _write_json_atomic(Path(output_path), contribution)
                return contribution
            if answer.lower() == "s":
                contribution["decisions"].append({
                    "scene_name": scene_name,
                    "status": "skipped",
                    "overlay_manifest_sha256": current_hash,
                })
                break
            if answer not in allowed:
                print(f"Choose one of: {', '.join(allowed)}")
                continue
            interaction = input("Can the intervention interact with ego within 3s? [y/N]: ").strip()
            if interaction.lower() != "y":
                print("Not approved; choose another candidate or skip the scene.")
                continue
            notes = input("Short audit reason: ").strip()
            if not notes:
                print("A short audit reason is required.")
                continue
            contribution["decisions"].append({
                "scene_name": scene_name,
                "status": "approved",
                "candidate_label": answer,
                "interaction_within_3s": True,
                "notes": notes,
                "overlay_manifest_sha256": current_hash,
            })
            break
        _write_json_atomic(Path(output_path), contribution)
    return contribution


def _read_minimal_selections(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["scene_name", "t0_sec"]:
            raise ValueError(
                f"selection CSV must contain exactly scene_name,t0_sec: {path}")
        rows = []
        seen = set()
        for row in reader:
            scene_name = str(row.get("scene_name", ""))
            if not scene_name or scene_name in seen:
                raise ValueError(f"missing or duplicate scene in selection CSV: {scene_name!r}")
            seen.add(scene_name)
            t0_sec = float(row["t0_sec"])
            if not math.isfinite(t0_sec) or t0_sec < 0.0:
                raise ValueError(f"invalid t0_sec for {scene_name}")
            rows.append({"scene_name": scene_name, "t0_sec": t0_sec})
    return rows


def merge_actor_contributions(
    overlay_dir: Path,
    input_selections_csv: Path,
    contribution_paths: Sequence[Path],
    output_selections_csv: Path,
    output_manifest_json: Path,
) -> List[Dict[str, Any]]:
    input_rows = _read_minimal_selections(Path(input_selections_csv))
    input_by_scene = {row["scene_name"]: row for row in input_rows}
    input_order = {row["scene_name"]: index for index, row in enumerate(input_rows)}
    overlay_index = _load_overlay_index(Path(overlay_dir))
    seen: Dict[str, str] = {}
    canonical: List[Dict[str, Any]] = []

    for path in contribution_paths:
        value = _load_json(Path(path))
        if value.get("schema_version") != SCHEMA_VERSION \
                or value.get("stage") != ACTOR_REVIEW_STAGE:
            raise ValueError(f"not an actor review contribution: {path}")
        reviewer = _validate_reviewer(str(value.get("reviewer", "")))
        decisions = value.get("decisions")
        if not isinstance(decisions, list):
            raise ValueError(f"decisions must be a list: {path}")
        for decision in decisions:
            scene_name = str(decision.get("scene_name", ""))
            if not scene_name:
                raise ValueError(f"missing scene_name in {path}")
            if scene_name in seen:
                raise ValueError(
                    f"duplicate scene decision for {scene_name}: "
                    f"{seen[scene_name]} and {reviewer}")
            seen[scene_name] = reviewer
            status = decision.get("status")
            if status == "skipped":
                continue
            if status != "approved":
                raise ValueError(f"unknown actor review status for {scene_name}: {status!r}")
            if decision.get("interaction_within_3s") is not True:
                raise ValueError(f"interaction_within_3s not approved for {scene_name}")
            notes = str(decision.get("notes", "")).strip()
            if not notes:
                raise ValueError(f"missing audit notes for {scene_name}")
            if scene_name not in input_by_scene:
                raise ValueError(f"actor review scene is absent from input selections: {scene_name}")
            index_row = overlay_index.get(scene_name)
            if index_row is None:
                raise ValueError(f"actor review scene is not valid in current index: {scene_name}")
            manifest_path = _overlay_manifest_path(Path(overlay_dir), scene_name)
            current_hash = file_sha256(manifest_path)
            if decision.get("overlay_manifest_sha256") != current_hash:
                raise ValueError(f"stale actor decision for {scene_name}; re-review current overlay")
            manifest = _load_json(manifest_path)
            if manifest.get("schema_version") != 2 \
                    or manifest.get("overlay_version") != "coriva-id-overlay-2":
                raise ValueError(f"unsupported overlay version for {scene_name}")
            convention = manifest.get("coordinate_convention", {})
            if convention.get("y_axis") != "positive_right_at_t0":
                raise ValueError(f"wrong overlay coordinate convention for {scene_name}")
            candidate_id = str(decision.get("candidate_label", ""))
            index_allowed = list(index_row.get("admissible_stable_candidates", []))
            manifest_allowed = list(manifest.get("admissible_stable_candidates", []))
            if candidate_id not in index_allowed or candidate_id not in manifest_allowed:
                raise ValueError(f"candidate {candidate_id!r} is not admissible for {scene_name}")
            candidate = manifest.get("candidates", {}).get(candidate_id)
            if not isinstance(candidate, dict):
                raise ValueError(f"candidate missing from manifest: {scene_name} {candidate_id}")
            applicability = candidate.get("role_state_applicability") or {}
            effect = candidate.get("paired_canonical_twin_effect_gate") or {}
            selected_intervention = applicability.get("selected_intervention")
            if applicability.get("passed") is not True:
                raise ValueError(f"role/state gate is not passed: {scene_name} {candidate_id}")
            if effect.get("passed") is not True:
                raise ValueError(f"paired-effect gate is not passed: {scene_name} {candidate_id}")
            if not selected_intervention \
                    or candidate.get("recommended_intervention") != selected_intervention:
                raise ValueError(f"intervention mismatch: {scene_name} {candidate_id}")
            selection = manifest.get("selection", {})
            requested_t0 = float(selection.get("requested_t0_sec"))
            if abs(requested_t0 - input_by_scene[scene_name]["t0_sec"]) > 1e-9:
                raise ValueError(f"T0 mismatch between CSV and overlay: {scene_name}")
            instance_token = str(candidate.get("instance_token", ""))
            if not instance_token:
                raise ValueError(f"missing instance token: {scene_name} {candidate_id}")
            canonical.append({
                "scene_name": scene_name,
                "t0_sec": input_by_scene[scene_name]["t0_sec"],
                "critical_instance_token": instance_token,
                "intervention_type": str(selected_intervention),
                "audit_status": "approved",
                "interaction_within_3s": True,
                "reviewer": reviewer,
                "notes": notes,
            })

    canonical.sort(key=lambda row: input_order[row["scene_name"]])
    approved_rows = [
        {"scene_name": row["scene_name"], "t0_sec": row["t0_sec"]}
        for row in canonical
    ]
    _write_minimal_csv(Path(output_selections_csv), approved_rows)
    _write_json_atomic(Path(output_manifest_json), {
        "schema_version": SCHEMA_VERSION,
        "selections": canonical,
    })
    return approved_rows


def merge_approved_rounds(
    selection_paths: Sequence[Path],
    manifest_paths: Sequence[Path],
    output_selections_csv: Path,
    output_manifest_json: Path,
) -> List[Dict[str, Any]]:
    """Join approved round outputs without weakening the canonical manifest."""
    if not selection_paths or len(selection_paths) != len(manifest_paths):
        raise ValueError(
            "provide the same non-zero number of selection CSVs and review manifests")

    approved_rows: List[Dict[str, Any]] = []
    canonical: List[Dict[str, Any]] = []
    seen: set[str] = set()
    required_manifest_fields = (
        "critical_instance_token",
        "intervention_type",
        "reviewer",
        "notes",
    )
    for selection_path, manifest_path in zip(selection_paths, manifest_paths):
        rows = _read_minimal_selections(Path(selection_path))
        manifest = _load_json(Path(manifest_path))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported review manifest schema: {manifest_path}")
        selections = manifest.get("selections")
        if not isinstance(selections, list):
            raise ValueError(f"manifest selections must be a list: {manifest_path}")
        by_scene: Dict[str, Dict[str, Any]] = {}
        for selection in selections:
            if not isinstance(selection, dict):
                raise ValueError(f"invalid manifest selection: {manifest_path}")
            scene_name = str(selection.get("scene_name", ""))
            if not scene_name or scene_name in by_scene:
                raise ValueError(
                    f"missing or duplicate scene in review manifest: {scene_name!r}")
            by_scene[scene_name] = selection

        if set(by_scene) != {row["scene_name"] for row in rows}:
            raise ValueError(
                f"selection CSV and review manifest do not contain the same scenes: "
                f"{selection_path} {manifest_path}")
        for row in rows:
            scene_name = row["scene_name"]
            if scene_name in seen:
                raise ValueError(f"duplicate approved scene across rounds: {scene_name}")
            selection = by_scene[scene_name]
            manifest_t0 = float(selection.get("t0_sec"))
            if not math.isfinite(manifest_t0) \
                    or abs(manifest_t0 - row["t0_sec"]) > 1e-9:
                raise ValueError(f"T0 mismatch in approved round: {scene_name}")
            if selection.get("audit_status") != "approved" \
                    or selection.get("interaction_within_3s") is not True:
                raise ValueError(f"scene is not human-approved: {scene_name}")
            for field in required_manifest_fields:
                if not str(selection.get(field, "")).strip():
                    raise ValueError(f"missing {field} for approved scene: {scene_name}")
            seen.add(scene_name)
            approved_rows.append(row)
            canonical.append(selection)

    _write_minimal_csv(Path(output_selections_csv), approved_rows)
    _write_json_atomic(Path(output_manifest_json), {
        "schema_version": SCHEMA_VERSION,
        "selections": canonical,
    })
    return approved_rows


def _add_common_reviewers(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reviewers", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--exclude", action="append", default=[], type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare_data = sub.add_parser(
        "prepare-data", help="assemble downloaded trainval folders into a CAM_FRONT dataroot")
    prepare_data.add_argument("--source", required=True, type=Path)
    prepare_data.add_argument("--out", required=True, type=Path)

    assign_t0 = sub.add_parser("assign-t0", help="split MP4 scenes across reviewers")
    assign_t0.add_argument("--review-dir", required=True, type=Path)
    _add_common_reviewers(assign_t0)
    assign_t0.add_argument("--target", type=int, default=260)

    review_t0 = sub.add_parser("review-t0", help="interactively record T0 decisions")
    review_t0.add_argument("--assignment", required=True, type=Path)
    review_t0.add_argument("--review-dir", required=True, type=Path)
    review_t0.add_argument("--output", required=True, type=Path)
    review_t0.add_argument("--no-open", action="store_true")

    merge_t0 = sub.add_parser("merge-t0", help="merge reviewer T0 contributions")
    merge_t0.add_argument("--contributions", nargs="+", required=True, type=Path)
    merge_t0.add_argument("--output", required=True, type=Path)
    merge_t0.add_argument("--target", type=int, default=260)

    make_overlays = sub.add_parser(
        "make-overlays", help="resolve T0 selections and create a fresh overlay-v2 round")
    make_overlays.add_argument("--dataroot", required=True, type=Path)
    make_overlays.add_argument("--review-dir", required=True, type=Path)
    make_overlays.add_argument("--selections", required=True, type=Path)
    make_overlays.add_argument("--out-dir", required=True, type=Path)
    make_overlays.add_argument("--version", default="v1.0-trainval")

    assign_actors = sub.add_parser(
        "assign-actors", help="split current admissible overlay scenes across reviewers")
    assign_actors.add_argument("--overlay-dir", required=True, type=Path)
    _add_common_reviewers(assign_actors)

    review_actors = sub.add_parser(
        "review-actors", help="interactively approve one actor per assigned scene")
    review_actors.add_argument("--assignment", required=True, type=Path)
    review_actors.add_argument("--overlay-dir", required=True, type=Path)
    review_actors.add_argument("--output", required=True, type=Path)
    review_actors.add_argument("--pages-dir", required=True, type=Path)
    review_actors.add_argument("--no-open", action="store_true")

    merge_actors = sub.add_parser(
        "merge-actors", help="produce canonical approved CSV and review manifest")
    merge_actors.add_argument("--overlay-dir", required=True, type=Path)
    merge_actors.add_argument("--input-selections", required=True, type=Path)
    merge_actors.add_argument("--contributions", nargs="+", required=True, type=Path)
    merge_actors.add_argument("--output-selections", required=True, type=Path)
    merge_actors.add_argument("--output-manifest", required=True, type=Path)
    merge_actors.add_argument(
        "--target", type=int, default=260,
        help="human-approved review quota; accepted quota is measured by the builder")

    merge_rounds = sub.add_parser(
        "merge-rounds", help="combine canonical approved CSV/manifest round pairs")
    merge_rounds.add_argument("--selections", nargs="+", required=True, type=Path)
    merge_rounds.add_argument("--manifests", nargs="+", required=True, type=Path)
    merge_rounds.add_argument("--output-selections", required=True, type=Path)
    merge_rounds.add_argument("--output-manifest", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare-data":
            summary = prepare_trainval_dataroot(args.source, args.out)
            print(
                f"blob_parts={summary['blob_parts']} "
                f"samples_CAM_FRONT={summary['sample_links']} "
                f"sweeps_CAM_FRONT={summary['sweep_links']} out={args.out}")
        elif args.command == "assign-t0":
            paths = write_t0_assignments(
                args.review_dir, args.reviewers, args.out_dir,
                target=args.target, exclude_paths=args.exclude)
            assigned = sum(len(_load_json(path)["scenes"]) for path in paths)
            print(f"assigned_t0={assigned} reviewers={len(paths)} requested={args.target}")
        elif args.command == "review-t0":
            value = review_t0_interactive(
                args.assignment, args.review_dir, args.output,
                open_video=not args.no_open)
            print(f"saved_t0_decisions={len(value['decisions'])} output={args.output}")
        elif args.command == "merge-t0":
            rows = merge_t0_contributions(args.contributions, args.output)
            print(
                f"selected_t0={len(rows)} target={args.target} "
                f"remaining={max(0, args.target - len(rows))} output={args.output}")
        elif args.command == "make-overlays":
            summary = make_overlay_round(
                args.dataroot, args.review_dir, args.selections, args.out_dir,
                version=args.version)
            print(
                f"overlay_valid={summary['valid']} "
                f"overlay_admissible={summary['with_admissible_candidates']} "
                f"output={args.out_dir}")
        elif args.command == "assign-actors":
            paths = write_actor_assignments(
                args.overlay_dir, args.reviewers, args.out_dir,
                exclude_paths=args.exclude)
            assigned = sum(len(_load_json(path)["scenes"]) for path in paths)
            print(f"assigned_actor_scenes={assigned} reviewers={len(paths)}")
        elif args.command == "review-actors":
            value = review_actors_interactive(
                args.assignment, args.overlay_dir, args.output, args.pages_dir,
                open_page=not args.no_open)
            print(f"saved_actor_decisions={len(value['decisions'])} output={args.output}")
        elif args.command == "merge-actors":
            rows = merge_actor_contributions(
                args.overlay_dir, args.input_selections, args.contributions,
                args.output_selections, args.output_manifest)
            print(
                f"approved={len(rows)} target={args.target} "
                f"remaining={max(0, args.target - len(rows))} "
                f"selections={args.output_selections} manifest={args.output_manifest}")
        elif args.command == "merge-rounds":
            rows = merge_approved_rounds(
                args.selections, args.manifests,
                args.output_selections, args.output_manifest)
            print(
                f"approved_all_rounds={len(rows)} "
                f"selections={args.output_selections} manifest={args.output_manifest}")
        else:  # pragma: no cover - argparse enforces the command set.
            raise ValueError(f"unknown command: {args.command}")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
