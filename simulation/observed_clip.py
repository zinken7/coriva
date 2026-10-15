"""
Build the observed video clip a VLM watches for a nuScenes-derived scenario.

Scenario frame export uses real consecutive CAM_FRONT keyframes ending at the
T=0 sample. Full-scene review clips can still use ``collect_cam_frames``. This
is genuine real-world video — no CARLA, no rendering — so the nuScenes track
can run the full benchmark without a GPU.
"""

import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

from simulation.protocol import OBSERVED_FRAME_COUNT


def _cam_keyframe_by_sample(tables: Dict[str, Dict[str, Any]], channel: str) -> Dict[str, str]:
    """Map sample_token -> CAM keyframe image filename for the given channel."""
    prefix = f"samples/{channel}/"
    out: Dict[str, str] = {}
    for sd in tables["sample_data"].values():
        if sd.get("is_key_frame") and sd["filename"].startswith(prefix):
            out.setdefault(sd["sample_token"], sd["filename"])
    return out


def write_scenario_frames(tables: Dict[str, Dict[str, Any]], sample_token: str,
                          dataroot: str, scen_dir: str, num_frames: int = 8,
                          channel: str = "CAM_FRONT",
                          cam_by_sample: Optional[Dict[str, str]] = None) -> List[str]:
    """Copy exactly ``num_frames`` consecutive keyframes ending at T0.

    The output is chronological and contains no uniform subsampling: for the
    8-frame CORIVA protocol it is T=-3.5, -3.0, ..., -0.5, 0.0 seconds at the
    nominal nuScenes 2 Hz annotation rate.  A short/missing history is an
    invalid candidate rather than a silently shorter clip.
    """
    selected = collect_consecutive_cam_frames(
        tables, sample_token, num_frames, channel, cam_by_sample=cam_by_sample)
    missing = [fn for fn in selected if not os.path.isfile(os.path.join(dataroot, fn))]
    if missing:
        raise FileNotFoundError(
            f"missing {channel} source image for sample {sample_token}: "
            f"{os.path.join(dataroot, missing[0])}")
    out_dir = os.path.join(scen_dir, "frames")
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    for i, fn in enumerate(selected):
        dst = os.path.join(out_dir, f"{i:03d}.jpg")
        shutil.copy(os.path.join(dataroot, fn), dst)
        written.append(dst)
    return written


def write_model_frames(tables: Dict[str, Dict[str, Any]], sample_token: str,
                       dataroot: str, scen_dir: str, root: Dict[str, Any],
                       num_frames: int = OBSERVED_FRAME_COUNT,
                       channel: str = "CAM_FRONT",
                       cam_by_sample: Optional[Dict[str, str]] = None,
                       visibility_policy: str = "center-v1") -> Dict[str, List[str]]:
    """Write raw audit frames and deterministic ID-overlaid model frames.

    center-v1 draws a fixed marker around the projected centre;
    bbox-area-v2 draws the clipped projected box. Visibility and identity
    come from metadata, with the same instance token used by JSON/query.
    """
    from PIL import Image, ImageDraw, ImageFont
    from simulation.nuscenes_visibility import policy_actor_visibility

    selected = collect_consecutive_cam_frames(
        tables, sample_token, num_frames, channel, cam_by_sample=cam_by_sample)
    tokens = [sample_token]
    tok = sample_token
    for _ in range(num_frames - 1):
        tok = tables["sample"][tok].get("prev")
        if not tok:
            raise ValueError("history window ended before all overlay frames")
        tokens.append(tok)
    tokens.reverse()
    raw_dir = os.path.join(scen_dir, "raw_frames")
    out_dir = os.path.join(scen_dir, "frames")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except (OSError, IOError):
        font = ImageFont.load_default()
    written_raw, written_overlay = [], []
    for index, (frame_token, source) in enumerate(zip(tokens, selected)):
        src_path = os.path.join(dataroot, source)
        image = Image.open(src_path).convert("RGB")
        raw_path = os.path.join(raw_dir, f"{index:03d}.jpg")
        shutil.copyfile(src_path, raw_path)
        draw = ImageDraw.Draw(image)
        for actor in root.get("actors", []):
            instance = actor.get("source_instance_token")
            if not instance:
                continue
            vis = policy_actor_visibility(tables, frame_token, instance, channel,
                                          policy=visibility_policy)
            if not vis.get("visible"):
                continue
            u, v = int(round(vis["u"])), int(round(vis["v"]))
            label = str(actor["id"])
            box = (vis["bbox_clipped"] if visibility_policy == "bbox-area-v2"
                   else (u - 18, v - 18, u + 18, v + 18))
            draw.rectangle(box, outline=(255, 32, 32), width=3)
            label_xy = (u + 20, v - 16)
            if visibility_policy == "bbox-area-v2":
                text_box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
                label_xy = (max(1, min(box[0] + 2, image.width - (text_box[2] - text_box[0]) - 2)),
                            max(1, min(box[1] + 2, image.height - (text_box[3] - text_box[1]) - 2)))
            draw.text(label_xy, label, fill=(255, 32, 32), font=font,
                      stroke_width=1, stroke_fill=(255, 255, 255))
        overlay_path = os.path.join(out_dir, f"{index:03d}.jpg")
        image.save(overlay_path, quality=95)
        written_raw.append(raw_path)
        written_overlay.append(overlay_path)
    return {"raw_frames": written_raw, "frames": written_overlay}


def collect_consecutive_cam_frames(tables: Dict[str, Dict[str, Any]], sample_token: str,
                                   num_frames: int = OBSERVED_FRAME_COUNT,
                                   channel: str = "CAM_FRONT",
                                   cam_by_sample: Optional[Dict[str, str]] = None) -> List[str]:
    """Return exactly N consecutive keyframe filenames ending at ``sample_token``.

    Traversing ``sample.prev`` makes the temporal contract explicit and avoids
    accidentally selecting sparse frames from the scene prefix.  All samples
    must remain in the same scene and have a keyframe for ``channel``.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    samples = tables["sample"]
    if sample_token not in samples:
        raise ValueError(f"unknown T0 sample_token: {sample_token}")
    scene_token = samples[sample_token]["scene_token"]
    tokens = [sample_token]
    tok = sample_token
    for _ in range(num_frames - 1):
        tok = samples[tok].get("prev")
        if not tok:
            raise ValueError(
                f"sample {sample_token} has fewer than {num_frames} consecutive history samples")
        if samples[tok]["scene_token"] != scene_token:
            raise ValueError(f"history for sample {sample_token} crosses a scene boundary")
        tokens.append(tok)
    tokens.reverse()

    cam = cam_by_sample if cam_by_sample is not None else _cam_keyframe_by_sample(tables, channel)
    missing_tokens = [tok for tok in tokens if tok not in cam]
    if missing_tokens:
        raise ValueError(
            f"sample {sample_token} history is missing {channel} keyframe(s): "
            + ", ".join(missing_tokens))
    return [cam[tok] for tok in tokens]


def collect_cam_frames(tables: Dict[str, Dict[str, Any]], sample_token: str,
                       channel: str = "CAM_FRONT") -> List[str]:
    """Ordered CAM frame filenames from the scene's first sample up to sample_token."""
    cam = _cam_keyframe_by_sample(tables, channel)
    scene = tables["scene"][tables["sample"][sample_token]["scene_token"]]
    frames: List[str] = []
    tok = scene["first_sample_token"]
    while tok:
        if tok in cam:
            frames.append(cam[tok])
        if tok == sample_token:
            break
        tok = tables["sample"][tok]["next"]
    return frames


def build_observed_clip(dataroot: str, tables: Dict[str, Dict[str, Any]], sample_token: str,
                        out_path: str, fps: int = 2, channel: str = "CAM_FRONT") -> str:
    """Encode the collected CAM frames into an mp4 at out_path (requires ffmpeg)."""
    frames = collect_cam_frames(tables, sample_token, channel)
    if not frames:
        raise ValueError(f"no {channel} keyframes up to sample {sample_token}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Sequentially-named symlinks so ffmpeg can use the image2 pattern reader.
    tmp = tempfile.mkdtemp(prefix="coriva_clip_")
    try:
        for i, fn in enumerate(frames):
            src = os.path.abspath(os.path.join(dataroot, fn))
            dst = os.path.join(tmp, f"{i:05d}.jpg")
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copyfile(src, dst)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
             "-i", os.path.join(tmp, "%05d.jpg"),
             "-vf", "format=yuv420p", "-c:v", "libx264", out_path],
            check=True,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out_path
