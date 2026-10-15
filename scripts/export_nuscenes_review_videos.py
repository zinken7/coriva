#!/usr/bin/env python3
"""Export full-scene nuScenes camera videos for manual event review.

The exporter works with the identical metadata layout used by v1.0-mini and
v1.0-trainval.  By default it includes both annotated keyframes from
``samples/<CHANNEL>/`` and the intervening camera frames from
``sweeps/<CHANNEL>/``.  A JSON sidecar maps every encoded video frame back to
its nuScenes timestamp and nearest annotated keyframe, so a reviewer can mark
an event and later choose an exact T0 sample.

Examples
--------
Export one mini scene as a quick smoke test::

    python3 scripts/export_nuscenes_review_videos.py \
        --dataroot data/nuscenes --version v1.0-mini \
        --out-dir review_videos_mini --limit 1

Export all trainval CAM_FRONT scenes::

    python3 scripts/export_nuscenes_review_videos.py \
        --dataroot /path/to/nuscenes --version v1.0-trainval \
        --out-dir review_videos_trainval

Export one named scene at annotated-keyframe rate only::

    python3 scripts/export_nuscenes_review_videos.py \
        --dataroot data/nuscenes --version v1.0-mini \
        --scene scene-0061 --keyframes-only
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import shutil
import subprocess
import sys
import tempfile
from json import JSONDecodeError
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


DEFAULT_CAMERA_FPS = 12.0


def _load_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, list):
        raise ValueError(f"expected a JSON array in {path}")
    return value


def _iter_json_array(path: str, chunk_size: int = 1 << 20) -> Iterator[Dict[str, Any]]:
    """Stream objects from a top-level JSON array using only the stdlib.

    ``sample_data.json`` is large in v1.0-trainval.  Streaming it prevents the
    exporter from retaining the full sensor table in memory when only one
    camera channel (and often only a few selected scenes) is required.
    """
    decoder = json.JSONDecoder()
    buffer = ""
    pos = 0
    started = False
    eof = False

    with open(path, "r", encoding="utf-8") as f:
        while True:
            if not eof and (pos >= len(buffer) or len(buffer) - pos < chunk_size // 4):
                buffer = buffer[pos:]
                pos = 0
                chunk = f.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            while pos < len(buffer) and buffer[pos].isspace():
                pos += 1

            if not started:
                if pos >= len(buffer):
                    if eof:
                        raise ValueError(f"empty JSON file: {path}")
                    continue
                if buffer[pos] != "[":
                    raise ValueError(f"expected a top-level JSON array in {path}")
                pos += 1
                started = True
                continue

            while pos < len(buffer) and (buffer[pos].isspace() or buffer[pos] == ","):
                pos += 1

            if pos < len(buffer) and buffer[pos] == "]":
                return
            if pos >= len(buffer):
                if eof:
                    raise ValueError(f"unterminated JSON array in {path}")
                continue

            try:
                value, end = decoder.raw_decode(buffer, pos)
            except JSONDecodeError:
                if eof:
                    raise ValueError(f"invalid or truncated JSON in {path}")
                # Preserve the incomplete object and read another chunk.
                buffer = buffer[pos:]
                pos = 0
                chunk = f.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
                continue

            if not isinstance(value, dict):
                raise ValueError(f"expected JSON objects in {path}, got {type(value).__name__}")
            yield value
            pos = end


def _select_scenes(
    scenes: Sequence[Dict[str, Any]],
    selectors: Optional[Sequence[str]],
    offset: int,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be > 0")

    if selectors:
        by_identifier: Dict[str, Dict[str, Any]] = {}
        for scene in scenes:
            by_identifier[str(scene["token"])] = scene
            by_identifier[str(scene["name"])] = scene
        missing = [selector for selector in selectors if selector not in by_identifier]
        if missing:
            raise ValueError("unknown scene name/token: " + ", ".join(missing))
        # Preserve the user's order while removing accidental duplicates.
        selected = []
        seen = set()
        for selector in selectors:
            scene = by_identifier[selector]
            if scene["token"] not in seen:
                selected.append(scene)
                seen.add(scene["token"])
    else:
        selected = list(scenes)

    selected = selected[offset:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def _collect_scene_frames(
    sample_data_path: str,
    sample_to_scene: Dict[str, str],
    selected_scene_tokens: Iterable[str],
    channel: str,
    keyframes_only: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    selected = set(selected_scene_tokens)
    frames: Dict[str, List[Dict[str, Any]]] = {token: [] for token in selected}
    allowed_prefixes = (f"samples/{channel}/",) if keyframes_only else (
        f"samples/{channel}/", f"sweeps/{channel}/")

    for record in _iter_json_array(sample_data_path):
        filename = str(record.get("filename", ""))
        if not filename.startswith(allowed_prefixes):
            continue
        if keyframes_only and not record.get("is_key_frame"):
            continue
        scene_token = sample_to_scene.get(str(record.get("sample_token", "")))
        if scene_token in selected:
            frames[scene_token].append(record)

    for records in frames.values():
        records.sort(key=lambda r: (int(r["timestamp"]), str(r.get("token", ""))))
    return frames


def _infer_fps(records: Sequence[Dict[str, Any]]) -> float:
    timestamps = [int(record["timestamp"]) for record in records]
    if len(timestamps) < 2:
        return DEFAULT_CAMERA_FPS
    elapsed_seconds = (timestamps[-1] - timestamps[0]) / 1e6
    if elapsed_seconds <= 0:
        return DEFAULT_CAMERA_FPS
    # nuScenes cameras are nominally 12 Hz, but individual scenes contain
    # small timestamp jitter and occasional gaps.  A constant-rate review MP4
    # cannot reproduce every irregular interval, so preserve the total scene
    # span: (N - 1) frame intervals cover (last_timestamp - first_timestamp).
    return round((len(timestamps) - 1) / elapsed_seconds, 6)


def _nearest_keyframe(
    timestamp: int,
    keyframe_timestamps: Sequence[int],
    keyframe_tokens: Sequence[str],
) -> Tuple[Optional[str], Optional[float]]:
    if not keyframe_timestamps:
        return None, None
    insertion = bisect.bisect_left(keyframe_timestamps, timestamp)
    candidates = []
    if insertion < len(keyframe_timestamps):
        candidates.append(insertion)
    if insertion > 0:
        candidates.append(insertion - 1)
    idx = min(candidates, key=lambda i: (abs(keyframe_timestamps[i] - timestamp), i))
    return keyframe_tokens[idx], round((timestamp - keyframe_timestamps[idx]) / 1e6, 6)


def _write_json_atomic(path: str, value: Dict[str, Any]) -> None:
    out_dir = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".review_sidecar_", suffix=".json", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _encode_video(
    dataroot: str,
    records: Sequence[Dict[str, Any]],
    fps: float,
    out_path: str,
    crf: int,
) -> None:
    sequence_dir = tempfile.mkdtemp(prefix="coriva_nuscenes_review_")
    output_dir = os.path.dirname(out_path) or "."
    fd, tmp_video = tempfile.mkstemp(prefix=".review_video_", suffix=".mp4", dir=output_dir)
    os.close(fd)
    os.unlink(tmp_video)  # ffmpeg creates the output itself.
    try:
        for index, record in enumerate(records):
            src = os.path.abspath(os.path.join(dataroot, record["filename"]))
            dst = os.path.join(sequence_dir, f"{index:06d}.jpg")
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copyfile(src, dst)

        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(fps),
                "-i", os.path.join(sequence_dir, "%06d.jpg"),
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
                "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                "-movflags", "+faststart", tmp_video,
            ],
            check=True,
        )
        os.replace(tmp_video, out_path)
    finally:
        shutil.rmtree(sequence_dir, ignore_errors=True)
        try:
            os.unlink(tmp_video)
        except OSError:
            pass


def _build_sidecar(
    scene: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    samples_by_token: Dict[str, Dict[str, Any]],
    version: str,
    channel: str,
    fps: float,
    video_filename: str,
    skipped_missing: int,
) -> Dict[str, Any]:
    scene_samples = sorted(
        (sample for sample in samples_by_token.values() if sample["scene_token"] == scene["token"]),
        key=lambda sample: int(sample["timestamp"]),
    )
    keyframe_timestamps = [int(sample["timestamp"]) for sample in scene_samples]
    keyframe_tokens = [str(sample["token"]) for sample in scene_samples]
    first_timestamp = int(records[0]["timestamp"])

    timeline = []
    for index, record in enumerate(records):
        timestamp = int(record["timestamp"])
        nearest_token, nearest_delta = _nearest_keyframe(
            timestamp, keyframe_timestamps, keyframe_tokens)
        timeline.append({
            "video_frame_index": index,
            "video_time_sec": round(index / fps, 6),
            "sensor_elapsed_sec": round((timestamp - first_timestamp) / 1e6, 6),
            "timestamp_us": timestamp,
            "sample_data_token": record.get("token"),
            "source_sample_token": record.get("sample_token"),
            "nearest_keyframe_sample_token": nearest_token,
            "nearest_keyframe_delta_sec": nearest_delta,
            "is_key_frame": bool(record.get("is_key_frame")),
            "filename": record.get("filename"),
        })

    return {
        "schema_version": 1,
        "dataset_version": version,
        "channel": channel,
        "scene": {
            "token": scene["token"],
            "name": scene["name"],
            "description": scene.get("description", ""),
            "nbr_samples": scene.get("nbr_samples"),
            "first_sample_token": scene.get("first_sample_token"),
            "last_sample_token": scene.get("last_sample_token"),
        },
        "video": {
            "filename": video_filename,
            "fps": fps,
            "frame_count": len(records),
            "duration_sec": round(len(records) / fps, 6),
            "skipped_missing_source_frames": skipped_missing,
        },
        "frames": timeline,
    }


def export_review_videos(args: argparse.Namespace) -> int:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH")
    if args.fps is not None and args.fps <= 0:
        raise ValueError("fps must be > 0")
    if not 0 <= args.crf <= 51:
        raise ValueError("crf must be between 0 and 51")

    metadata_dir = os.path.join(args.dataroot, args.version)
    scene_path = os.path.join(metadata_dir, "scene.json")
    sample_path = os.path.join(metadata_dir, "sample.json")
    sample_data_path = os.path.join(metadata_dir, "sample_data.json")
    for path in (scene_path, sample_path, sample_data_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"required nuScenes metadata file not found: {path}")

    scenes = _load_json_array(scene_path)
    samples = _load_json_array(sample_path)
    selected_scenes = _select_scenes(scenes, args.scene, args.offset, args.limit)
    if not selected_scenes:
        print("No scenes selected.")
        return 0

    selected_tokens = {str(scene["token"]) for scene in selected_scenes}
    samples_by_token = {str(sample["token"]): sample for sample in samples}
    sample_to_scene = {
        token: str(sample["scene_token"])
        for token, sample in samples_by_token.items()
        if str(sample["scene_token"]) in selected_tokens
    }

    print(
        f"Scanning {sample_data_path} for {len(selected_scenes)} scene(s), "
        f"channel={args.channel}..."
    )
    frames_by_scene = _collect_scene_frames(
        sample_data_path,
        sample_to_scene,
        selected_tokens,
        args.channel,
        args.keyframes_only,
    )
    os.makedirs(args.out_dir, exist_ok=True)

    exported = skipped = failed = 0
    for position, scene in enumerate(selected_scenes, start=1):
        scene_name = str(scene["name"])
        video_path = os.path.join(args.out_dir, f"{scene_name}_{args.channel}.mp4")
        sidecar_path = os.path.join(args.out_dir, f"{scene_name}_{args.channel}.json")
        if not args.overwrite and os.path.exists(video_path) and os.path.exists(sidecar_path):
            print(f"[{position}/{len(selected_scenes)}] [skip] {scene_name}: output exists")
            skipped += 1
            continue
        if not args.overwrite and (os.path.exists(video_path) or os.path.exists(sidecar_path)):
            print(
                f"[{position}/{len(selected_scenes)}] [error] {scene_name}: partial output exists; "
                "use --overwrite to replace it",
                file=sys.stderr,
            )
            failed += 1
            continue

        records = list(frames_by_scene.get(str(scene["token"]), []))
        missing = [
            record for record in records
            if not os.path.isfile(os.path.join(args.dataroot, str(record.get("filename", ""))))
        ]
        if missing and not args.skip_missing:
            example = os.path.join(args.dataroot, str(missing[0].get("filename", "")))
            print(
                f"[{position}/{len(selected_scenes)}] [error] {scene_name}: "
                f"{len(missing)} source image(s) missing; first: {example}",
                file=sys.stderr,
            )
            failed += 1
            continue
        if missing:
            missing_tokens = {str(record.get("token")) for record in missing}
            records = [record for record in records if str(record.get("token")) not in missing_tokens]
        if not records:
            print(
                f"[{position}/{len(selected_scenes)}] [error] {scene_name}: "
                f"no {args.channel} frames found",
                file=sys.stderr,
            )
            failed += 1
            continue

        fps = float(args.fps) if args.fps is not None else _infer_fps(records)
        try:
            print(
                f"[{position}/{len(selected_scenes)}] [export] {scene_name}: "
                f"{len(records)} frames @ {fps:.3f} fps"
            )
            _encode_video(args.dataroot, records, fps, video_path, args.crf)
            sidecar = _build_sidecar(
                scene,
                records,
                samples_by_token,
                args.version,
                args.channel,
                fps,
                os.path.basename(video_path),
                len(missing),
            )
            _write_json_atomic(sidecar_path, sidecar)
            exported += 1
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            print(f"[{position}/{len(selected_scenes)}] [error] {scene_name}: {exc}", file=sys.stderr)
            failed += 1

    print(
        f"Done: exported={exported} skipped={skipped} failed={failed} "
        f"out_dir={args.out_dir}"
    )
    return 1 if failed else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export full-scene nuScenes camera videos and timestamp sidecars for manual review"
    )
    parser.add_argument("--dataroot", default="data/nuscenes", help="nuScenes dataset root")
    parser.add_argument("--version", default="v1.0-mini", help="e.g. v1.0-mini or v1.0-trainval")
    parser.add_argument("--out-dir", default="review_videos", help="output directory")
    parser.add_argument("--channel", default="CAM_FRONT", help="camera channel to export")
    parser.add_argument(
        "--scene", action="append", default=None, metavar="NAME_OR_TOKEN",
        help="export one scene name/token; repeat to select multiple scenes",
    )
    parser.add_argument("--offset", type=int, default=0, help="skip this many selected scenes")
    parser.add_argument("--limit", type=int, default=None, help="maximum number of scenes to export")
    parser.add_argument(
        "--fps", type=float, default=None,
        help="encoded FPS; default infers the median source-camera rate per scene",
    )
    parser.add_argument(
        "--keyframes-only", action="store_true",
        help="use annotated samples/<CHANNEL> keyframes only (normally both samples and sweeps are used)",
    )
    parser.add_argument(
        "--skip-missing", action="store_true",
        help="encode around missing source images instead of rejecting that scene",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing video/sidecar outputs")
    parser.add_argument("--crf", type=int, default=18, help="H.264 quality (0 best, 51 worst; default 18)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return export_review_videos(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
