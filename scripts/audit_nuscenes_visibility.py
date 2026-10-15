#!/usr/bin/env python3
"""Audit approved actors using metadata only; does not change admission or review.

The experimental area threshold is measured at a 1024x768 reference resolution.
Projected box area is not an occlusion measurement or a full dataset admission.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.package_nuscenes_review import _package_files, MANIFEST, SELECTIONS
from simulation.nuscenes_visibility import (
    measure_box, REFERENCE_WIDTH, REFERENCE_HEIGHT, NEAR_PLANE)


def iter_records(path, chunk_size=1024 * 1024):
    """Stream a JSON array of objects without loading GB-sized tables into RAM."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8-sig") as handle:
        buffer, pos, eof = "", 0, False

        def refill():
            nonlocal buffer, pos, eof
            chunk = handle.read(chunk_size)
            buffer = buffer[pos:] + chunk
            pos = 0
            eof = not chunk

        def peek():
            nonlocal pos
            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos < len(buffer):
                    return buffer[pos]
                if eof:
                    return ""
                refill()

        if peek() != "[":
            raise ValueError(f"Expected JSON array: {path}")
        pos += 1
        if peek() != "]":
            while True:
                if peek() != "{":
                    raise ValueError(f"Expected table record: {path}")
                while True:
                    try:
                        record, end = decoder.raw_decode(buffer, pos)
                        break
                    except json.JSONDecodeError:
                        if eof:
                            raise ValueError(f"Invalid/incomplete JSON: {path}") from None
                        refill()
                pos = end
                yield record
                separator = peek()
                if separator == "]":
                    break
                if separator != ",":
                    raise ValueError(f"Expected comma or closing bracket: {path}")
                pos += 1
        pos += 1
        if peek():
            raise ValueError(f"Trailing content in JSON: {path}")


def _history(source, row, samples, scenes):
    name = row["scene_name"]
    sidecar = json.loads((source / "videos" / f"{name}_CAM_FRONT.json").read_text())
    fps = float(sidecar.get("video", {}).get("fps", 0))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("invalid sidecar fps")
    frames, seen = [], set()
    for frame in sidecar.get("frames", []):
        token = frame.get("source_sample_token")
        if not frame.get("is_key_frame") or not token or token in seen:
            continue
        time = frame.get("video_time_sec")
        time = float(time) if time is not None else int(frame["video_frame_index"]) / fps
        if not math.isfinite(time):
            raise ValueError("invalid sidecar keyframe time")
        frames.append((time, token))
        seen.add(token)
    if not frames:
        raise ValueError("no keyframes in sidecar")
    time, token = min(frames, key=lambda f: (abs(f[0] - row["t0_sec"]), f[0]))
    tokens = []
    for _ in range(8):
        sample = samples.get(token)
        if sample is None or sample["scene_token"] != scenes[name]["token"]:
            raise ValueError("missing history sample or history crosses scene boundary")
        tokens.append(token)
        token = sample.get("prev")
    if len(set(tokens)) != 8:
        raise ValueError("duplicate sample in history")
    return list(reversed(tokens)), time


def audit(source, metadata, min_area):
    source, metadata = Path(source), Path(metadata)
    _package_files(source, False)  # Check approved CSV/manifest pair and sidecars; no images.
    approved = json.loads((source / MANIFEST).read_text())["selections"]
    scenes = {r["name"]: r for r in iter_records(metadata / "scene.json")}
    samples = {r["token"]: r for r in iter_records(metadata / "sample.json")}
    results, wanted = [], set()
    for row in approved:
        result = {"scene_name": row["scene_name"], "t0_sec": float(row["t0_sec"]),
                  "critical_instance_token": row["critical_instance_token"], "frames": []}
        try:
            tokens, time = _history(source, result, samples, scenes)
            result.update(sample_tokens=tokens, resolved_t0_sec=time)
            wanted.update(tokens)
        except (KeyError, ValueError, TypeError) as exc:
            result.update(status="error", reason=str(exc))
        results.append(result)

    cameras = {}
    for record in iter_records(metadata / "sample_data.json"):
        if (record["sample_token"] in wanted and record.get("is_key_frame")
                and record.get("filename", "").startswith("samples/CAM_FRONT/")):
            token = record["sample_token"]
            if token in cameras:
                raise ValueError(f"duplicate CAM_FRONT keyframe: {token}")
            cameras[token] = record
    needed_poses = {r["ego_pose_token"] for r in cameras.values()}
    poses = {r["token"]: r for r in iter_records(metadata / "ego_pose.json") if r["token"] in needed_poses}
    calibrations = {r["token"]: r for r in iter_records(metadata / "calibrated_sensor.json")}
    pairs = {(token, row["critical_instance_token"]) for row in results for token in row.get("sample_tokens", [])}
    annotations = {}
    for record in iter_records(metadata / "sample_annotation.json"):
        pair = record["sample_token"], record["instance_token"]
        if pair in pairs:
            if pair in annotations:
                raise ValueError(f"duplicate actor annotation: {pair}")
            annotations[pair] = record
    for row in results:
        if row.get("status") == "error":
            continue
        for index, token in enumerate(row["sample_tokens"]):
            frame = {"frame_index": index, "sample_token": token}
            try:
                camera = cameras[token]
                ann = annotations[token, row["critical_instance_token"]]
                measurement = measure_box(ann, poses[camera["ego_pose_token"]],
                                          calibrations[camera["calibrated_sensor_token"]],
                                          camera["width"], camera["height"])
                frame.update(measurement, width=camera["width"], height=camera["height"],
                             area_pass=measurement["area_reference_px2"] >= min_area)
            except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError) as exc:
                frame["error"] = f"missing/invalid frame metadata: {exc}"
            row["frames"].append(frame)
        if any("error" in f for f in row["frames"]):
            row["status"] = "error"
        elif any(f["crosses_near_plane"] for f in row["frames"]):
            row["status"] = "review"
        else:
            row["status"] = "pass" if all(f["area_pass"] for f in row["frames"]) else "fail"
        row["center_pass_all_8"] = all(f.get("center_visible", False) for f in row["frames"])
        row["failed_frame_indices"] = [f["frame_index"] for f in row["frames"] if not f.get("area_pass", False)]
        areas = [f["area_reference_px2"] for f in row["frames"] if "area_reference_px2" in f]
        row["min_area_reference_px2"] = min(areas) if len(areas) == 8 else None
    counts = Counter(row["status"] for row in results)
    return {
        "audit_version": "nuscenes-projected-box-audit-1", "scope": "approved critical actor visibility only",
        "threshold": {"min_area_reference_px2": min_area, "reference_width": REFERENCE_WIDTH,
                      "reference_height": REFERENCE_HEIGHT, "near_plane_m": NEAR_PLANE,
                      "experimental": True},
        "limitations": ["No occlusion measurement", "No role/effect/fidelity checks",
                        "Does not change step2 or builder admission", "Near-plane crossings need review"],
        "inputs": {"metadata_dir": str(metadata.resolve()), "review_dir": str(source.resolve()),
                   "approved_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                       for name in (SELECTIONS, MANIFEST)}},
        "summary": {"total": len(results), **{status: counts[status] for status in ("pass", "fail", "review", "error")}},
        "scenes": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--metadata-dir", type=Path,
                        default=PROJECT_ROOT / "trainval/v1.0-trainval_meta/v1.0-trainval",
                        help="Directory directly containing scene.json, sample.json, etc.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/visibility_audit",
                        help="New report directory (must not exist)")
    parser.add_argument("--min-area-px2", type=float, default=100,
                        help="Experimental threshold at 1024x768 reference resolution (default: 100)")
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.min_area_px2) or args.min_area_px2 <= 0:
            raise ValueError("--min-area-px2 must be finite and positive")
        output = args.output.expanduser().absolute()
        if output.exists() or output.is_symlink():
            raise ValueError(f"Thư mục đã tồn tại; chọn --output mới: {output}")
        print("Đang đọc metadata; không đọc ảnh/video...", flush=True)
        report = audit(args.source.expanduser(), args.metadata_dir.expanduser(), args.min_area_px2)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".visibility-audit-", dir=output.parent) as temporary:
            staged = Path(temporary) / "report"
            staged.mkdir()
            (staged / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            fields = ["scene_name", "t0_sec", "critical_instance_token", "status", "center_pass_all_8",
                      "min_area_reference_px2", "failed_frame_indices", "reason"]
            with (staged / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(report["scenes"])
            if output.exists() or output.is_symlink():
                raise ValueError(f"Thư mục đã tồn tại: {output}")
            staged.rename(output)
        summary = report["summary"]
        print(" ".join(f"{key}={value}" for key, value in summary.items()))
        print(f"Báo cáo: {output}")
        return 2 if summary["error"] else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
