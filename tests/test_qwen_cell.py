"""Single-cell diagnostic reaches the real parser/scorers without GPU/network."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import test_qwen_cell as diagnostic
from tests.test_evaluation_campaign import make_protocol_dataset


def test_single_cell_offload_is_recorded_and_scores_real_pipeline(tmp_path, monkeypatch):
    from analysis.backends import local_hf
    from analysis.backends.result import result_from_raw
    from analysis.run_model import _oracle_text, _validate_model_config
    source = make_protocol_dataset(tmp_path, split="nuscenes", n=1)
    gt = json.loads((source / "scenario_1/ground_truth.json").read_text())
    monkeypatch.setattr(diagnostic, "repository_state",
                        lambda: {"code_commit": "f" * 40, "worktree_clean": True})
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))
    model = SimpleNamespace(dtype="bf16", parameters=lambda: iter([SimpleNamespace(is_cuda=True)]))
    monkeypatch.setattr(local_hf, "_load", lambda *a: (model, None))
    calls = []

    def generate(prompt, frames, model_id, **kwargs):
        calls.append((prompt, frames, kwargs))
        return result_from_raw(_oracle_text(gt))

    monkeypatch.setattr(local_hf, "generate_result", generate)
    args = SimpleNamespace(dataset_root=source, out_dir=tmp_path / "diagnostic", split="nuscenes",
                           condition="video_history", scenario="scenario_1", offload_cache=True,
                           configuration="qwen3-vl-8b-instruct", scenarios=None)
    diagnostic.run(args)
    assert len(calls) == 1 and len(calls[0][1]) == 8
    assert calls[0][2]["cache_implementation"] == "offloaded"
    frozen = json.loads((args.out_dir / "frozen.json").read_text())
    cfg = frozen["model"]
    assert cfg["cache_implementation"] == "offloaded"
    cell = args.out_dir / "cells/nuscenes/qwen3-vl-8b-instruct_video_history"
    manifest = json.loads((cell / "scenario_1/run_manifest.json").read_text())
    assert manifest["cache_implementation"] == "offloaded"
    assert json.loads((cell / "completeness.json").read_text())["complete"]
    assert json.loads((cell / "report.json").read_text())["summary"]["L4_pair_valid_rate"] == 1
    assert (cell / "collected.json").is_file() and (args.out_dir / "tables.csv").is_file()
    # Do not silently reuse a diagnostic cache strategy for a campaign.
    with pytest.raises(ValueError, match="diagnostic option"):
        _validate_model_config(cfg, "local", cfg["model_id"], 0.0)
    # The helper must never replace existing raw responses or reports.
    with pytest.raises(FileExistsError):
        diagnostic.run(args)
    assert len(calls) == 1


def test_single_cell_refuses_dirty_checkout_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "repository_state", lambda: {"worktree_clean": False})
    args = SimpleNamespace(out_dir=tmp_path / "diagnostic")
    with pytest.raises(ValueError, match="clean checkout"):
        diagnostic.run(args)
    assert not args.out_dir.exists()
