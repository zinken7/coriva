"""Explain saved overlay admission decisions without rerunning the pipeline."""

from collections import Counter
import json
from pathlib import Path


REASON_TEXT = {
    "outside_image": "tâm annotation nằm ngoài vùng ảnh cho phép",
    "behind_camera": "tâm annotation nằm phía sau camera",
    "missing_annotation": "thiếu annotation của actor tại frame này",
    "missing_camera_keyframe": "thiếu keyframe camera",
    "actor_missing_from_t0_root": "actor không có trong observed state tại T0",
    "lead_actor_not_ahead_of_ego": "xe dẫn đầu không nằm phía trước ego",
    "adjacent_actor_not_ahead_of_ego": "xe bên cạnh không nằm phía trước ego",
    "lead_actor_not_moving_forward_fast_enough": "vận tốc dọc của xe dẫn đầu dưới ngưỡng phanh",
    "adjacent_actor_not_moving_forward_fast_enough": "vận tốc dọc của xe bên cạnh dưới ngưỡng cut-in",
    "adjacent_actor_inside_ego_lane_center_corridor": "xe đã ở trong hành lang giữa làn ego",
    "vehicle_role_not_intervention_eligible": "vai trò của xe chưa được hỗ trợ cho can thiệp",
    "unsupported_nuscenes_intervention_actor_type": "loại actor chưa được hỗ trợ cho can thiệp",
    "road_entry_actor_has_wrong_role": "vai trò không phù hợp với can thiệp đi vào đường",
    "road_entry_actor_not_ahead_of_ego": "actor không nằm phía trước ego",
    "road_entry_actor_already_in_ego_lane_corridor": "actor đã chạm hoặc nằm trong hành lang làn ego",
    "road_entry_actor_moving_away_from_ego_lane": "actor đang đi ra xa làn ego quá ngưỡng cho phép",
    "road_entry_actor_cannot_reach_ego_corridor_within_horizon": "profile can thiệp không đưa actor tới hành lang ego trong 3 giây",
    "canonical_intervention_has_no_material_target_effect": "thay đổi vị trí và tốc độ đều dưới ngưỡng hiệu ứng",
    "critical_actor_does_not_interact_with_ego_within_3s": "mô phỏng không có va chạm và khoảng cách mép nhỏ nhất vượt ngưỡng trong 3 giây",
}


def _explain(reason: str) -> str:
    explanation = REASON_TEXT.get(reason.split(":", 1)[0])
    return f"{reason} — {explanation}" if explanation else reason


def _failed_stage(label: str, manifest: dict) -> str:
    # Use recorded admission lists as authority; do not re-score saved data.
    for key, stage in (
        ("stable_candidates", "visibility"),
        ("role_state_eligible_stable_candidates", "role_state"),
        ("admissible_stable_candidates", "effect"),
    ):
        if label not in manifest[key]:
            return stage
    return "admissible"


def format_scene_diagnostics(scene_name: str, manifest: dict, *, details: bool = False) -> str:
    candidates = manifest["candidates"]
    stages = Counter(_failed_stage(label, manifest) for label in candidates)
    selection = manifest.get("selection", {})
    lines = [
        f"{scene_name}: candidates={len(candidates)}, "
        f"stable={len(manifest['stable_candidates'])}, "
        f"role_state={len(manifest['role_state_eligible_stable_candidates'])}, "
        f"admissible={len(manifest['admissible_stable_candidates'])}",
        f"  T0 requested={selection.get('requested_t0_sec', '?')}s; "
        f"resolved={selection.get('resolved_t0_sec', '?')}s",
        "  Loại ở bước đầu tiên: " + ", ".join(
            f"{stage}={stages[stage]}" for stage in ("visibility", "role_state", "effect")),
    ]
    reasons = Counter()
    for label, candidate in candidates.items():
        stage = _failed_stage(label, manifest)
        frames = candidate.get("frames", [])
        failed_frames = [f"{i:03d}:{frame.get('reason') or 'unknown'}"
                         for i, frame in enumerate(frames) if not frame.get("visible")]
        role = candidate.get("role_state_applicability") or {}
        effect = candidate.get("paired_canonical_twin_effect_gate") or {}
        if stage == "visibility":
            codes = sorted({frame.get("reason") or "unknown_visibility_reason"
                            for frame in frames if not frame.get("visible")})
        elif stage == "role_state":
            codes = role.get("reasons") or ["role_state_reason_not_recorded"]
        elif stage == "effect":
            codes = effect.get("reasons") or ["effect_reason_not_recorded"]
        else:
            codes = []
        reasons.update(sorted(set(codes)))
        if not details:
            continue
        lines.append(f"  {label} ({candidate.get('entity_type', '?')}, "
                     f"{candidate.get('role_type', '?')}): {stage}; "
                     f"visible={sum(bool(f.get('visible')) for f in frames)}/8")
        if failed_frames:
            lines.append("    Frames không đạt: " + ", ".join(failed_frames))
        # Include all recorded failures, even when an earlier gate already failed.
        for reason in dict.fromkeys(codes + role.get("reasons", []) + effect.get("reasons", [])):
            lines.append("    " + _explain(reason))
        evidence = {
            "t0_state": candidate.get("t0_state"),
            "role_thresholds": role.get("thresholds"),
            "derived_state": role.get("derived_state"),
            "intervention": candidate.get("recommended_intervention"),
            "effect": {key: value for key, value in effect.items()
                       if key.startswith(("max_target_", "min_treated_"))
                       or key in ("thresholds", "treated_collision_depth")},
        }
        lines.append("    Số liệu đã lưu: " + json.dumps(evidence, ensure_ascii=False))
    if not candidates:
        lines.append("  Không có annotation candidate tại T0. Manifest không ghi lý do "
                     "cho từng actor bị loại trước khi gán nhãn (loại đối tượng, bán kính 60 m).")
    if reasons:
        lines.append("  Lý do tại bước loại đầu tiên (số actor; một actor có thể có nhiều lý do):")
        lines.extend(f"    {count} × {_explain(reason)}" for reason, count in reasons.most_common())
    return "\n".join(lines)


def diagnose_round(overlay_dir: Path, scene_name: str | None = None) -> None:
    index_path = overlay_dir / "index.json"
    if not index_path.is_file():
        raise ValueError(f"Chưa có {index_path}; cần output step2 để chẩn đoán.")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    rows = index["valid"] + index["rejected"]
    if scene_name and not any(row["scene_name"] == scene_name for row in rows):
        raise ValueError(f"{scene_name} không có trong {index_path}")
    print("Đọc manifest của round hiện tại; không render lại hoặc sửa lựa chọn.")
    print("visible dùng tâm annotation chiếu lên ảnh, không kiểm tra che khuất; "
          "effect là kết quả mô phỏng canonical twin.")
    empty = 0
    for row in index["valid"]:
        name = row["scene_name"]
        if scene_name and name != scene_name:
            continue
        manifest = json.loads((overlay_dir / name / "overlay_manifest.json").read_text(encoding="utf-8"))
        has_candidates = bool(manifest["admissible_stable_candidates"])
        empty += not has_candidates
        if not scene_name and has_candidates:
            continue
        print(format_scene_diagnostics(name, manifest, details=bool(scene_name)))
    for row in index["rejected"]:
        if row["scene_name"] == scene_name:
            print(f"REJECT {row['scene_name']}: {row.get('error', 'unknown')}")
    if not scene_name:
        print(f"Tổng: {len(index['valid'])} scene có overlay; {empty} không có actor admissible; "
              f"{len(index['rejected'])} T0 bị loại.")
