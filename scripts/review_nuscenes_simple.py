#!/usr/bin/env python3
"""Three-step local nuScenes review that writes everything under results/."""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import webbrowser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = PROJECT_ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
SOURCE_DIR = PROJECT_ROOT / "trainval"
DATAROOT = PROJECT_ROOT / "data" / "nuscenes_trainval_full"
RESULTS_DIR = PROJECT_ROOT / "results"
VIDEOS_DIR = RESULTS_DIR / "videos"
SELECTIONS_CSV = RESULTS_DIR / "review_selections.csv"
OVERLAYS_DIR = RESULTS_DIR / "overlays"
NO_CANDIDATES_DIR = RESULTS_DIR / "overlays_no_candidates"
ACTOR_REVIEW_DIR = RESULTS_DIR / "actor_review"
ACTOR_REVIEW_CSV = RESULTS_DIR / "actor_review.csv"
ACTOR_DECISIONS_JSON = RESULTS_DIR / "actor_decisions.json"
APPROVED_CSV = RESULTS_DIR / "review_selections_approved.csv"
APPROVED_MANIFEST = RESULTS_DIR / "review_manifest_approved.json"
PART_RE = re.compile(r"^v1\.0-trainval(\d{2})_blobs$")


def _inside_project_venv() -> bool:
    return Path(sys.prefix).resolve() == VENV_DIR.resolve()


def _bootstrap_or_reexec(command: str) -> None:
    if _inside_project_venv():
        return
    if command == "step1":
        if not VENV_PYTHON.is_file():
            subprocess.run(
                [sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        subprocess.run(
            [str(VENV_PYTHON), "-m", "pip", "install", "-r",
             str(PROJECT_ROOT / "requirements.txt")],
            check=True,
        )
    elif not VENV_PYTHON.is_file():
        raise RuntimeError("Chưa có .venv. Hãy chạy step1 trước.")
    os.execv(
        str(VENV_PYTHON),
        [str(VENV_PYTHON), str(Path(__file__).resolve()), command],
    )


def _require_external_tools() -> None:
    missing = [name for name in ("ffmpeg", "magick") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            f"Thiếu {', '.join(missing)}. Mac: brew install ffmpeg imagemagick; "
            "Ubuntu: sudo apt-get install ffmpeg imagemagick")


def _downloaded_part_numbers(source_dir: Path) -> list[int]:
    parts: list[int] = []
    for path in source_dir.iterdir() if source_dir.is_dir() else []:
        match = PART_RE.fullmatch(path.name)
        if match and path.is_dir():
            parts.append(int(match.group(1)))
    parts = sorted(set(parts))
    if not parts or any(part < 1 or part > 10 for part in parts):
        raise ValueError(
            "Không tìm thấy blob part hợp lệ trong trainval/ "
            "(cần v1.0-trainval01_blobs ... v1.0-trainval10_blobs).")
    return parts


def scene_names_for_parts(scene_rows: list[dict], parts: list[int]) -> list[str]:
    """Map official 10 trainval blob parts to their 85-scene metadata slices."""
    if len(scene_rows) != 850:
        raise ValueError(f"Metadata trainval phải có 850 scene, hiện có {len(scene_rows)}")
    names: list[str] = []
    for part in parts:
        start = (part - 1) * 85
        names.extend(str(row["name"]) for row in scene_rows[start:start + 85])
    return names


def _create_selections_csv_if_missing() -> None:
    if SELECTIONS_CSV.exists():
        return
    SELECTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SELECTIONS_CSV.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerow(["scene_name", "t0_sec"])


def step1() -> None:
    _require_external_tools()
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.export_nuscenes_review_videos import _build_parser, export_review_videos
    from scripts.nuscenes_review_queue import prepare_trainval_dataroot

    parts = _downloaded_part_numbers(SOURCE_DIR)
    summary = prepare_trainval_dataroot(SOURCE_DIR, DATAROOT)
    scene_rows = json.loads(
        (DATAROOT / "v1.0-trainval" / "scene.json").read_text(encoding="utf-8"))
    scene_names = scene_names_for_parts(scene_rows, parts)
    pending_scenes = [name for name in scene_names if not (
        (VIDEOS_DIR / f"{name}_CAM_FRONT.mp4").is_file()
        and (VIDEOS_DIR / f"{name}_CAM_FRONT.json").is_file())]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    arguments = [
        "--dataroot", str(DATAROOT),
        "--version", "v1.0-trainval",
        "--out-dir", str(VIDEOS_DIR),
        "--crf", "22",
    ]
    for scene_name in pending_scenes:
        arguments.extend(["--scene", scene_name])
    return_code = export_review_videos(_build_parser().parse_args(arguments)) if pending_scenes else 0
    if return_code != 0:
        raise RuntimeError("Tạo video có lỗi; xem các dòng [error] phía trên.")

    _create_selections_csv_if_missing()
    video_count = len(list(VIDEOS_DIR.glob("scene-*_CAM_FRONT.mp4")))
    print(
        f"\nSTEP 1 XONG: parts={parts} linked_samples={summary['sample_links']} "
        f"videos={video_count} skipped={len(scene_names) - len(pending_scenes)}")
    print(f"Xem video tại: {VIDEOS_DIR}")
    print(f"Ghi scene_name,t0_sec vào: {SELECTIONS_CSV}")
    print("Sau khi ghi xong CSV, chạy: python3 scripts/review_nuscenes_simple.py step2")


def _write_actor_review_index(page_paths: list[Path]) -> Path:
    links = "\n".join(
        f'<li><a href="{html.escape(path.name)}">{html.escape(path.stem)}</a></li>'
        for path in page_paths
    )
    document = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><title>Actor review</title>
<style>body {{ font-family: sans-serif; margin: 32px; line-height: 1.5; }}</style>
</head><body><h1>Actor review</h1>
<p>Mở từng scene, xem đủ 8 ảnh và chọn tối đa một candidate trong bảng.</p>
<ol>{links}</ol></body></html>
"""
    index_path = ACTOR_REVIEW_DIR / "index.html"
    index_path.write_text(document, encoding="utf-8")
    return index_path


def _write_actor_review_csv(candidates_by_scene: dict[str, list[str]], *, resume: bool = False) -> None:
    fields = ["scene_name", "candidate_label", "interaction_within_3s", "notes"]
    original = b""
    existing = set()
    if ACTOR_REVIEW_CSV.exists():
        if not resume:
            raise ValueError(f"{ACTOR_REVIEW_CSV} đã tồn tại; không ghi đè lựa chọn human.")
        original = ACTOR_REVIEW_CSV.read_bytes()
        reader = csv.DictReader(io.StringIO(original.decode("utf-8-sig"), newline=""))
        if reader.fieldnames != fields:
            raise ValueError(f"actor_review.csv phải giữ đúng 4 cột: {','.join(fields)}")
        for row in reader:
            name = row["scene_name"]
            if name in existing or name not in candidates_by_scene or None in row or any(v is None for v in row.values()):
                raise ValueError(f"actor_review.csv có dòng thiếu, trùng hoặc scene ngoài index: {name}")
            existing.add(name)
    additions = io.StringIO(newline="")
    if not original:
        csv.writer(additions, lineterminator="\n").writerow(fields)
    for scene_name, candidates in candidates_by_scene.items():
        if scene_name in existing:
            continue
        candidate = candidates[0] if len(candidates) == 1 else "cand_"
        row = io.StringIO(newline="")
        csv.writer(row, lineterminator="\n").writerow([scene_name, candidate, "yes"])
        additions.write(row.getvalue().removesuffix("\n") + ',""\n')
    if additions.getvalue():
        separator = b"\n" if original and not original.endswith((b"\n", b"\r")) else b""
        with tempfile.NamedTemporaryFile(dir=ACTOR_REVIEW_CSV.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(original + separator + additions.getvalue().encode("utf-8"))
        temporary.replace(ACTOR_REVIEW_CSV)


def export_no_candidate_overlays(overlay_dir: Path, out_dir: Path, *, resume: bool = False) -> int:
    """Export empty-admission scenes with every visible candidate labelled."""
    if os.path.lexists(out_dir) and not resume:
        raise ValueError(f"{out_dir} đã tồn tại; không ghi đè bản xem lại.")
    index = json.loads((overlay_dir / "index.json").read_text(encoding="utf-8"))
    selected = []
    for row in index["valid"]:
        name = str(row["scene_name"])
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"scene_name không hợp lệ: {name}")
        scene_dir = overlay_dir / name
        manifest = json.loads((scene_dir / "overlay_manifest.json").read_text(encoding="utf-8"))
        if manifest["admissible_stable_candidates"]:
            continue
        for frame in range(8):
            path = scene_dir / f"{frame:03d}.jpg"
            if not path.is_file():
                raise ValueError(f"Thiếu ảnh overlay: {path}")
        selected.append({**row, "manifest": str(
            (out_dir / name / "overlay_manifest.json").resolve())})
    from scripts.nuscenes_review_queue import file_sha256, _write_json_atomic
    selected_names = {row["scene_name"] for row in selected}
    current_names = {row["scene_name"] for row in index["valid"]}
    removed_records = {row["scene_name"]: row for row in index.get("removed", [])}
    replaced = set()
    obsolete = []

    def validate_existing(name):
        current = overlay_dir / name
        destination = out_dir / name
        old_hash = file_sha256(destination / "overlay_manifest.json")
        frame_hashes = None
        if name not in current_names and name in removed_records:
            record = removed_records[name]
            digest = record.get("removed_manifest_sha256", record.get("archived_manifest_sha256"))
            if old_hash != digest:
                raise ValueError(f"Manifest không khớp scene đã bỏ: {destination}")
            frame_hashes = record.get("frame_sha256")
            replaced.add(name)
        else:
            manifest = json.loads((current / "overlay_manifest.json").read_text(encoding="utf-8"))
            if old_hash != file_sha256(current / "overlay_manifest.json"):
                if old_hash not in manifest.get("superseded_overlay_manifest_sha256s", []):
                    raise ValueError(f"Bản xem lại không khớp manifest nguồn: {destination}")
                frame_hashes = manifest.get("superseded_frame_sha256", {}).get(old_hash)
                replaced.add(name)
            else:
                frame_hashes = {f"{i:03d}.jpg": file_sha256(current / f"{i:03d}.jpg") for i in range(8)}
        marker_path = destination / "review_render.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8")) if marker_path.is_file() else None
        if marker and (marker.get("label_policy") != "all_candidates" or marker.get("source_manifest_sha256") != old_hash):
            raise ValueError(f"Bản xem lại không khớp manifest nguồn: {marker_path}")
        for i in range(8):
            filename = f"{i:03d}.jpg"
            target = destination / filename
            expected = marker.get("frame_sha256", {}).get(filename) if marker else (frame_hashes or {}).get(filename)
            if (marker is not None or frame_hashes is not None) and expected is None:
                raise ValueError(f"Thiếu checksum ảnh xem lại: {target}")
            # A replacement may carry only a manifest hash. Replacements are disposable
            # inspection copies, and are regenerated from the current source.
            if not target.is_file() or target.stat().st_size == 0 or (expected is not None and file_sha256(target) != expected):
                raise ValueError(f"Bản xem lại khác hoặc thiếu file gốc: {target}; không ghi đè.")
        return marker

    if out_dir.exists():
        for path in out_dir.iterdir():
            if path.is_dir() and path.name not in selected_names:
                if path.name not in current_names and path.name not in removed_records:
                    raise ValueError(f"Scene ngoài index trong folder xem lại: {path}")
                validate_existing(path.name)
                if path.name not in replaced:
                    raise ValueError(f"Scene không có bản thay T0 hợp lệ: {path}")
                obsolete.append(path)
    pending = []
    manifests = {}
    relabel = set()
    for row in selected:
        name = row["scene_name"]
        source_manifest = overlay_dir / name / "overlay_manifest.json"
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        manifests[name] = manifest
        # A manifest without this field has rendered all candidates.
        needs_labels = bool(manifest.get("candidates")) and manifest.get("rendered_candidate_labels") == []
        if not (out_dir / name).exists():
            pending.append(row)
            if needs_labels:
                relabel.add(name)
            continue
        marker = validate_existing(name)
        if name in replaced or (needs_labels and marker is None):
            pending.append(row)
            if needs_labels:
                relabel.add(name)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".no_candidates_", dir=out_dir.parent) as temporary:
        staged = Path(temporary) / "overlays"
        staged.mkdir()
        for row in pending:
            name = row["scene_name"]
            existing = out_dir / name
            shutil.copytree(existing if existing.exists() and name not in replaced else overlay_dir / name, staged / name)
            if name in relabel:
                from simulation.review_overlay import render_saved_candidate_labels
                render_saved_candidate_labels(manifests[name], overlay_dir / name, staged / name)
                _write_json_atomic(staged / name / "review_render.json", {
                    "label_policy": "all_candidates",
                    "rendered_candidate_labels": list(manifests[name]["candidates"]),
                    "source_manifest_sha256": file_sha256(overlay_dir / name / "overlay_manifest.json"),
                    "frame_sha256": {f"{i:03d}.jpg": file_sha256(staged / name / f"{i:03d}.jpg")
                                     for i in range(8)},
                })
        export_index = {
            "source_overlay_dir": str(overlay_dir.resolve()),
            "purpose": "inspection_only_no_admissible_candidates",
            "valid": selected, "rejected": [],
        }
        if out_dir.exists():
            moved = []
            installed = []
            try:
                for row in pending:
                    destination = out_dir / row["scene_name"]
                    backup = Path(temporary) / row["scene_name"]
                    if destination.exists():
                        destination.rename(backup)
                        moved.append((destination, backup))
                    (staged / row["scene_name"]).rename(destination)
                    installed.append(destination)
                for path in obsolete:
                    backup = Path(temporary) / path.name
                    path.rename(backup)
                    moved.append((path, backup))
                if not (out_dir / "index.json").is_file() or json.loads((out_dir / "index.json").read_text()) != export_index:
                    _write_json_atomic(out_dir / "index.json", export_index)
            except OSError:
                for path in reversed(installed):
                    path.rename(staged / path.name)
                for path, backup in reversed(moved):
                    backup.rename(path)
                raise
        else:
            _write_json_atomic(staged / "index.json", export_index)
            staged.rename(out_dir)
    return len(selected)


def step2() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.nuscenes_review_queue import (
        _read_minimal_selections,
        make_overlay_round,
        plan_overlay_resume,
        validate_t0,
        write_actor_review_page,
    )

    if not SELECTIONS_CSV.is_file():
        raise ValueError(f"Không có {SELECTIONS_CSV}; hãy chạy step1 trước.")
    rows = _read_minimal_selections(SELECTIONS_CSV)
    if not rows and not (OVERLAYS_DIR / "index.json").is_file():
        raise ValueError(f"{SELECTIONS_CSV} chưa có scene nào.")
    _, pending = plan_overlay_resume(SELECTIONS_CSV, OVERLAYS_DIR,
                                    allow_empty_t0_changes=True, allow_empty_removals=True)
    if pending:
        _require_external_tools()
    for row in pending:
        sidecar = VIDEOS_DIR / f"{row['scene_name']}_CAM_FRONT.json"
        if not sidecar.is_file():
            raise ValueError(f"Không có video/JSON cho {row['scene_name']}.")
        try:
            validate_t0(sidecar, float(row["t0_sec"]))
        except ValueError as exc:
            raise ValueError(f"{row['scene_name']} T0 không hợp lệ: {exc}") from exc

    summary = make_overlay_round(
        DATAROOT, VIDEOS_DIR, SELECTIONS_CSV, OVERLAYS_DIR, resume=True,
        allow_empty_t0_changes=True, allow_empty_removals=True)
    index = json.loads((OVERLAYS_DIR / "index.json").read_text(encoding="utf-8"))
    empty_count = export_no_candidate_overlays(OVERLAYS_DIR, NO_CANDIDATES_DIR, resume=True)
    print(f"Scene không có candidate: {empty_count} — {NO_CANDIDATES_DIR}")
    scene_names = [
        str(row["scene_name"])
        for row in index["valid"]
        if row.get("admissible_stable_candidates")
    ]
    if not scene_names:
        print(f"STEP 2 XONG: new={summary['created']} rebuilt={summary['rebuilt']} removed={summary['removed']} skipped={summary['skipped']}; không có actor admissible.")
        return

    ACTOR_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    pages = []
    for scene_name in scene_names:
        page = ACTOR_REVIEW_DIR / f"{scene_name}.html"
        if not page.is_file():
            page = write_actor_review_page(OVERLAYS_DIR, scene_name, ACTOR_REVIEW_DIR)
        pages.append(page)
    index_path = _write_actor_review_index(pages)
    _write_actor_review_csv({
        str(row["scene_name"]): list(row["admissible_stable_candidates"])
        for row in index["valid"] if row.get("admissible_stable_candidates")
    }, resume=True)
    webbrowser.open(index_path.resolve().as_uri())

    print(
        f"\nSTEP 2 XONG: new={summary['created']} rebuilt={summary['rebuilt']} removed={summary['removed']} skipped={summary['skipped']} overlays={summary['valid']} "
        f"actor_scenes={len(scene_names)}")
    print(f"Xem 8 ảnh và candidate tại: {index_path}")
    print(f"Ghi lựa chọn actor vào: {ACTOR_REVIEW_CSV}")
    print("Sau khi ghi xong CSV, chạy: python3 scripts/review_nuscenes_simple.py step3")


def actor_decisions_from_csv(
    csv_path: Path, overlay_dir: Path, expected_scenes: list[str]
) -> list[dict]:
    from scripts.nuscenes_review_queue import file_sha256

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_fields = [
            "scene_name", "candidate_label", "interaction_within_3s", "notes"]
        if reader.fieldnames != expected_fields:
            raise ValueError(
                "actor_review.csv phải giữ đúng 4 cột: " + ",".join(expected_fields))
        rows = list(reader)

    by_scene: dict[str, dict] = {}
    for row in rows:
        scene_name = str(row.get("scene_name", "")).strip()
        if not scene_name or scene_name in by_scene:
            raise ValueError(f"scene_name thiếu hoặc trùng trong actor_review.csv: {scene_name}")
        by_scene[scene_name] = row
    if set(by_scene) != set(expected_scenes):
        raise ValueError("Không thêm hoặc xóa scene trong actor_review.csv.")

    decisions: list[dict] = []
    for scene_name in expected_scenes:
        row = by_scene[scene_name]
        candidate = str(row.get("candidate_label", "")).strip()
        if candidate == "cand_":
            raise ValueError(f"{scene_name}: hãy điền đủ mã cand_XXX hoặc để trống cả 3 ô để skip.")
        interaction = str(row.get("interaction_within_3s", "")).strip().lower()
        notes = str(row.get("notes", "")).strip()
        manifest_path = overlay_dir / scene_name / "overlay_manifest.json"
        manifest_hash = file_sha256(manifest_path)
        if not candidate:
            if interaction or notes:
                raise ValueError(
                    f"{scene_name}: để trống cả 3 ô nếu muốn skip scene.")
            decisions.append({
                "scene_name": scene_name,
                "status": "skipped",
                "overlay_manifest_sha256": manifest_hash,
            })
            continue
        if interaction not in {"yes", "y", "true"}:
            raise ValueError(f"{scene_name}: interaction_within_3s phải là yes.")
        if not notes:
            raise ValueError(f"{scene_name}: cần ghi lý do ngắn trong notes.")
        decisions.append({
            "scene_name": scene_name,
            "status": "approved",
            "candidate_label": candidate,
            "interaction_within_3s": True,
            "notes": notes,
            "overlay_manifest_sha256": manifest_hash,
        })
    return decisions


def step3() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.nuscenes_review_queue import (
        ACTOR_REVIEW_STAGE,
        SCHEMA_VERSION,
        merge_actor_contributions,
    )

    if not ACTOR_REVIEW_CSV.is_file() or not (OVERLAYS_DIR / "index.json").is_file():
        raise ValueError("Thiếu output step2. Hãy chạy step2 trước.")
    index = json.loads((OVERLAYS_DIR / "index.json").read_text(encoding="utf-8"))
    expected_scenes = [
        str(row["scene_name"])
        for row in index["valid"]
        if row.get("admissible_stable_candidates")
    ]
    decisions = actor_decisions_from_csv(
        ACTOR_REVIEW_CSV, OVERLAYS_DIR, expected_scenes)
    contribution = {
        "schema_version": SCHEMA_VERSION,
        "stage": ACTOR_REVIEW_STAGE,
        "reviewer": "external_reviewer",
        "decisions": decisions,
    }
    ACTOR_DECISIONS_JSON.write_text(
        json.dumps(contribution, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    approved = merge_actor_contributions(
        OVERLAYS_DIR,
        SELECTIONS_CSV,
        [ACTOR_DECISIONS_JSON],
        APPROVED_CSV,
        APPROVED_MANIFEST,
    )
    if not approved:
        raise ValueError("Không actor nào được approve; chưa có package để copy.")

    print(f"\nSTEP 3 XONG: approved={len(approved)}")
    print(f"CSV cuối: {APPROVED_CSV}")
    print(f"JSON cuối: {APPROVED_MANIFEST}")
    print(f"Copy toàn bộ folder này về máy OWNER: {RESULTS_DIR}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("step1", "step2", "step3", "diagnose", "export-no-candidates"))
    parser.add_argument("--scene", help="Chi tiết một scene; chỉ dùng với diagnose.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        if args.scene and args.step != "diagnose":
            raise ValueError("--scene chỉ dùng với diagnose.")
        if args.step == "export-no-candidates":
            count = export_no_candidate_overlays(OVERLAYS_DIR, NO_CANDIDATES_DIR, resume=True)
            print(f"Scene không có candidate: {count} — {NO_CANDIDATES_DIR}")
            return 0
        if args.step == "diagnose":
            sys.path.insert(0, str(PROJECT_ROOT))
            from scripts.nuscenes_review_diagnostics import diagnose_round
            diagnose_round(OVERLAYS_DIR, args.scene)
            return 0
        _bootstrap_or_reexec(args.step)
        {"step1": step1, "step2": step2, "step3": step3}[args.step]()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
