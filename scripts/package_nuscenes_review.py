#!/usr/bin/env python3
"""Copy approved review artifacts for a nuScenes build on another machine.

Uses only the Python standard library; raw nuScenes data is not included.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile


DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "results"
SELECTIONS = "review_selections_approved.csv"
MANIFEST = "review_manifest_approved.json"


def _t0(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"T0 không hợp lệ: {value!r}") from None
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"T0 không hợp lệ: {value!r}")
    return result


def _load_json(path):
    with path.open(encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON phải là object: {path}")
    return value


def _package_files(source, include_videos):
    with (source / SELECTIONS).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"scene_name", "t0_sec"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{SELECTIONS}: thiếu scene_name hoặc t0_sec")
        scenes = {}
        for row in reader:
            name = (row.get("scene_name") or "").strip()
            if not re.fullmatch(r"scene-\d{4}", name):
                raise ValueError(f"Tên scene không hợp lệ: {name!r}")
            if name in scenes:
                raise ValueError(f"Scene trùng trong CSV: {name}")
            scenes[name] = _t0(row.get("t0_sec"))
    if not scenes:
        raise ValueError(f"{SELECTIONS}: chưa có scene approved; hoàn thành step3 trước")

    selections = _load_json(source / MANIFEST).get("selections")
    if not isinstance(selections, list):
        raise ValueError(f"{MANIFEST}: thiếu danh sách selections")
    by_scene = {}
    for row in selections:
        if not isinstance(row, dict) or not isinstance(row.get("scene_name"), str):
            raise ValueError(f"{MANIFEST}: dòng selection không hợp lệ")
        name = row["scene_name"]
        if name in by_scene:
            raise ValueError(f"Scene trùng trong manifest: {name}")
        by_scene[name] = row
    if set(by_scene) != set(scenes):
        raise ValueError("CSV và manifest approved không cùng danh sách scene")

    files = [Path(SELECTIONS), Path(MANIFEST)]
    for name, t0 in scenes.items():
        row = by_scene[name]
        if abs(_t0(row.get("t0_sec")) - t0) > 1e-9:
            raise ValueError(f"{name}: T0 trong CSV và manifest không khớp")
        if row.get("audit_status") != "approved" or row.get("interaction_within_3s") is not True:
            raise ValueError(f"{name}: chưa approved")
        for field in ("critical_instance_token", "intervention_type"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{name}: thiếu {field}")
        relative = Path("videos") / f"{name}_CAM_FRONT.json"
        sidecar = _load_json(source / relative)
        scene = sidecar.get("scene")
        if not isinstance(scene, dict) or scene.get("name") != name:
            raise ValueError(f"{relative}: tên scene không khớp")
        files.append(relative)
        if include_videos:
            files.append(relative.with_suffix(".mp4"))

    for relative in files:
        path = source / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"File thiếu hoặc rỗng: {path}")
    return files, len(scenes)


def package_review(source, output, include_videos=False):
    """Validate the pair, copy byte-for-byte, then publish a complete directory."""
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"Thư mục đích đã tồn tại; chọn --output mới: {output}")
    files, count = _package_files(source, include_videos)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".nuscenes-package-", dir=output.parent) as temporary:
        staged = Path(temporary) / "package"
        (staged / "videos").mkdir(parents=True)
        for relative in files:
            shutil.copy2(source / relative, staged / relative)
        if output.exists() or output.is_symlink():
            raise ValueError(f"Thư mục đích đã tồn tại: {output}")
        staged.rename(output)
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Thư mục chứa hai file approved và videos/ (mặc định: Coding/results)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Thư mục gói upload mới; bên trong sẽ có videos/ và hai file approved")
    parser.add_argument("--include-videos", action="store_true",
                        help="Copy thêm MP4 của các scene approved để xem lại")
    args = parser.parse_args(argv)
    try:
        count = package_review(args.source, args.output, args.include_videos)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    media = "JSON + MP4" if args.include_videos else "JSON"
    print(f"Đã copy {count} scene ({media}) + 2 file approved → {args.output.expanduser().absolute()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
