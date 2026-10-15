"""
Contamination probe (TODO.md A3 / dissertation §5.3 contamination signal).

nuScenes is public and likely in VLM training corpora; CARLA scenarios are
newly generated. If a model can identify a nuScenes scene from its frames at
above-chance rate, the real-split scores carry a memorisation confound. The
probe shows a model the observed frames, asks it to name the dataset / city /
whether it recognises the specific scene, and reports the recognition rate.
It is a SIGNAL, not proof — reported alongside the CARLA-vs-nuScenes gap.

Transport is mocked here (as for the API backend tests); the real smoke is a
separate cheap run.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis import contamination_probe as cp  # noqa: E402


def test_probe_prompt_asks_for_dataset_city_scene_recognition():
    prompt = cp.build_probe_prompt("nuscenes_0001")
    low = prompt.lower()
    assert "dataset" in low and "city" in low
    assert "recognise" in low or "recognize" in low
    # must request a machine-readable answer
    assert "json" in low
    assert "nuscenes_0001" not in low


@pytest.mark.parametrize("text,expect_ns,expect_recog", [
    ('{"dataset": "nuScenes", "city": "Boston", "recognized_scene": true, "confidence": 0.9}', True, True),
    ('{"dataset": "unknown", "city": "unknown", "recognized_scene": false, "confidence": 0.1}', False, False),
    ('{"dataset": "some driving dataset", "city": "Singapore", "recognized_scene": false, "confidence": 0.4}', False, False),
    ('the answer is {"dataset":"NUSCENES","city":"singapore","recognized_scene":true,"confidence":0.8} ok', True, True),
])
def test_parse_probe_response(text, expect_ns, expect_recog):
    out = cp.parse_probe_response(text)
    assert out["names_nuscenes"] is expect_ns
    assert out["recognized_scene"] is expect_recog


def test_parse_probe_response_city_only_counts_as_nuscenes_signal():
    # names a nuScenes-only city but not the dataset → still a memorisation signal
    out = cp.parse_probe_response('{"dataset":"?","city":"boston","recognized_scene":false}')
    assert out["names_nuscenes_city"] is True
    assert out["names_nuscenes"] is False


def test_parse_probe_response_survives_garbage():
    out = cp.parse_probe_response("I cannot determine the source.")
    assert out["names_nuscenes"] is False and out["recognized_scene"] is False


def test_summarize_probe_rates():
    results = [
        {"names_nuscenes": True, "names_nuscenes_city": True, "recognized_scene": True},
        {"names_nuscenes": False, "names_nuscenes_city": True, "recognized_scene": False},
        {"names_nuscenes": False, "names_nuscenes_city": False, "recognized_scene": False},
        {"names_nuscenes": True, "names_nuscenes_city": True, "recognized_scene": False},
    ]
    s = cp.summarize_probe(results)
    assert s["n"] == 4
    assert s["names_nuscenes_rate"] == pytest.approx(0.5)
    assert s["names_nuscenes_city_rate"] == pytest.approx(0.75)
    assert s["recognized_scene_rate"] == pytest.approx(0.25)


@pytest.mark.skip(reason="needs pilot fixture data that is not part of this release")
def test_run_probe_end_to_end_mocked(tmp_path, monkeypatch):
    from analysis.backends import api
    monkeypatch.setattr(api, "_post_json", lambda *a, **k: {
        "candidates": [{"content": {"parts": [{"text":
            '{"dataset":"nuScenes","city":"Boston","recognized_scene":true,"confidence":0.9}'}]}}],
        "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20},
    })
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    out = cp.run_probe("pilot30", backend="gemini", model_id="gemini-2.5-pro",
                       limit=3, num_frames=8, out_path=str(tmp_path / "probe.json"))
    assert out["summary"]["n"] == 3
    assert out["summary"]["names_nuscenes_rate"] == pytest.approx(1.0)
    saved = json.load(open(tmp_path / "probe.json"))
    assert saved["summary"]["n"] == 3 and len(saved["per_scenario"]) == 3
