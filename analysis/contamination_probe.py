"""
Contamination probe (dissertation §5.3).

nuScenes is public and likely in current VLMs' training data; CARLA scenarios
are freshly generated. If a model can name the dataset / a nuScenes-only city
/ the specific scene from its observed frames at above-chance rate, real-split
scores carry a memorisation confound. This probe measures that recognition
rate directly. It is a SIGNAL reported next to the CARLA-vs-nuScenes score gap,
not proof of leakage.

CLI:
    python -m analysis.contamination_probe --scenario-dir output_nuscenes_protocol10 \
        --backend gemini --model gemini-2.5-pro --limit 20 \
        --out results/contamination_nuscenes_protocol10.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# nuScenes was collected only in these two cities. Naming one is a strong
# memorisation signal even without naming the dataset.
_NUSCENES_CITIES = ("boston", "singapore")


def build_probe_prompt(scenario_id: str = "") -> str:
    # Never expose the internal scenario id: final nuScenes ids contain the
    # dataset name and would make the recognition probe answer itself.
    return """You are shown frames from an anonymized driving scene.

Identify the source of this footage as precisely as you can. Answer with a
single JSON object and nothing else:

{{
  "dataset": "<name of the dataset this footage is from, or 'unknown'>",
  "city": "<city or region the footage was recorded in, or 'unknown'>",
  "recognized_scene": <true if you recognise this SPECIFIC scene/log, else false>,
  "confidence": <0.0-1.0>
}}

Be honest: say 'unknown' and false if you are guessing."""


def _extract_json(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}


def parse_probe_response(text: str) -> Dict[str, Any]:
    """Turn raw model text into recognition flags."""
    obj = _extract_json(text)
    dataset = str(obj.get("dataset", "")).lower()
    city = str(obj.get("city", "")).lower()
    names_nuscenes = "nuscenes" in dataset.replace(" ", "").replace("-", "")
    names_city = any(c in city for c in _NUSCENES_CITIES)
    recognized = obj.get("recognized_scene") is True
    return {
        "names_nuscenes": bool(names_nuscenes),
        "names_nuscenes_city": bool(names_city),
        "recognized_scene": bool(recognized),
        "confidence": obj.get("confidence"),
        "raw_dataset": obj.get("dataset"),
        "raw_city": obj.get("city"),
    }


def summarize_probe(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "names_nuscenes_rate": sum(r["names_nuscenes"] for r in results) / n,
        "names_nuscenes_city_rate": sum(r["names_nuscenes_city"] for r in results) / n,
        "recognized_scene_rate": sum(r["recognized_scene"] for r in results) / n,
    }


def run_probe(gt_dir: str, backend: str = "gemini", model_id: str = "gemini-2.5-pro",
              limit: int = 20, num_frames: int = 8, seed: int = 0,
              out_path: str = "") -> Dict[str, Any]:
    """Probe every scenario (up to `limit`) with its observed frames and report
    recognition rates. Backends: gemini | gpt (frames are the observed input,
    same as scoring)."""
    from analysis.frames import select_frames
    if backend == "gemini":
        from analysis.backends.api import gemini_generate as gen
    elif backend == "gpt":
        from analysis.backends.api import openai_generate as gen
    else:
        raise ValueError(f"probe backend must be gemini|gpt, got {backend!r}")

    scen_dirs = sorted(glob.glob(os.path.join(gt_dir, "scenario_*")))[:limit]
    per_scenario: List[Dict[str, Any]] = []
    for scen_dir in scen_dirs:
        # The recognition probe needs only released observed frames. Do not
        # open ground_truth.json merely to obtain an internal identifier.
        sid = os.path.basename(scen_dir)
        frame_dir = os.path.join(scen_dir, "frames")
        frames = select_frames(
            sorted(glob.glob(os.path.join(frame_dir, "*.jpg")) +
                   glob.glob(os.path.join(frame_dir, "*.png"))), num_frames)
        text = gen(build_probe_prompt(sid), frames, model_id, seed=seed)
        parsed = parse_probe_response(text)
        parsed["scenario_id"] = sid
        per_scenario.append(parsed)

    out = {"scenario_dir": gt_dir, "model": model_id,
           "summary": summarize_probe(per_scenario), "per_scenario": per_scenario}
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scenario-dir", "--gt-dir", dest="scenario_dir", required=True)
    p.add_argument("--backend", default="gemini", choices=["gemini", "gpt"])
    p.add_argument("--model", default="gemini-2.5-pro")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--out", default="")
    args = p.parse_args()
    out = run_probe(args.scenario_dir, backend=args.backend, model_id=args.model,
                    limit=args.limit, num_frames=args.num_frames, out_path=args.out)
    print(json.dumps(out["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
