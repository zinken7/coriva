"""Run one diagnostic cell, preserving eight frames, BF16 and scorer settings.

Defaults to the nuScenes video_history cell that exhausted a 24 GB RTX 3090.
KV cache offloading uses host RAM; this is diagnostic evidence, not admission.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.evaluation_campaign import (
    CONDITIONS, dataset_inventory, sha, write_json, write_campaign_identity, score_cell)
from analysis.evaluation_report import collect_cell, mean_stat, export_tables
from analysis.model_panel import load_model_panel, resolve_configurations
from analysis.smoke_admission import repository_state
from analysis.run_model import run_model


def run(args):
    state = repository_state()
    if not state["worktree_clean"]:
        raise ValueError(f"Commit/push the changes and pull a clean checkout first: {state}")
    source = args.dataset_root.resolve()
    if args.scenarios is not None:
        if args.scenarios < 1:
            raise ValueError("--scenarios must be a positive integer")
        chosen = sorted(p for p in source.glob("scenario_*") if p.is_dir())[:args.scenarios]
        if len(chosen) != args.scenarios:
            raise ValueError(f"{source} holds {len(chosen)} scenarios, asked for {args.scenarios}")
    else:
        scenario = source / args.scenario
        if Path(args.scenario).name != args.scenario or not args.scenario.startswith("scenario_"):
            raise ValueError("--scenario must be a scenario_* directory name")
        if not scenario.is_dir():
            raise FileNotFoundError(scenario)
        chosen = [scenario]
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    subset = out / "inputs" / args.split
    subset.mkdir(parents=True)
    for scenario in chosen:
        (subset / scenario.name).symlink_to(scenario, target_is_directory=True)
    rows = dataset_inventory(subset, len(chosen), args.split)
    panel = Path(__file__).resolve().parents[1] / "analysis/model_panel.yaml"
    model_cfg = resolve_configurations(load_model_panel(str(panel)), [args.configuration])[0]
    model_cfg["cache_implementation"] = "offloaded" if args.offload_cache else None
    cfg = {"campaign_dir": str(out), "splits": {args.split: {"path": str(subset)}}}
    frozen = {"purpose": "single-cell diagnostic, not campaign admission", "repository": state,
              "model": model_cfg, "datasets": {args.split: rows}}
    write_json(out / "frozen.json", frozen)

    import torch
    from analysis.backends.local_hf import _load
    model, _ = _load(model_cfg["adapter"], model_cfg["model_id"], model_cfg["revision"])
    if model.dtype != torch.bfloat16 or not all(p.is_cuda for p in model.parameters()):
        raise RuntimeError("This diagnostic requires BF16 model weights entirely on GPU")
    cell = dict(split=args.split, backend="local", condition=args.condition,
                configuration=model_cfg["name"],
                out_dir=str(out / "cells" / args.split / f"{model_cfg['name']}_{args.condition}"))
    write_campaign_identity(cfg, cell)
    print(f"RUN {args.split}/{len(chosen)} scenario(s)/{args.condition}; "
          f"cache={model_cfg['cache_implementation'] or 'default'}", flush=True)
    stats = run_model(str(subset), cell["out_dir"], backend="local", condition=args.condition,
                      model_id=model_cfg["model_id"], model_config=model_cfg,
                      temperature=0.0, seed=0, provisional_smoke=True)
    write_json(Path(cell["out_dir"]) / "run_stats.json", stats)
    score_cell(cfg, cell, frozen)
    data = collect_cell(cell, rows, split_path=str(subset), frozen_sha256=sha(out / "frozen.json"))
    write_json(Path(cell["out_dir"]) / "collected.json", data)
    records = [dict(split=args.split, configuration=model_cfg["name"], condition=args.condition,
                    comparison="scores", metric=metric, **mean_stat(list(values.values())))
               for metric, values in data["metrics"].items()]
    export_tables(out, records)
    print(f"DONE: {out} (reports written; inspect branch/pair validity, not just completion)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory; never overwrites results")
    parser.add_argument("--configuration", default="qwen3-vl-8b-instruct",
                        help="Panel configuration name; the panel validates it")
    parser.add_argument("--split", choices=("carla", "nuscenes"), default="nuscenes")
    parser.add_argument("--condition", choices=CONDITIONS, default="video_history")
    parser.add_argument("--scenario", default="scenario_1")
    parser.add_argument("--scenarios", type=int,
                        help="Run the first N scenarios of the split instead of one. "
                             "Still a diagnostic: the output is never campaign evidence.")
    parser.add_argument("--offload-cache", action="store_true", help="Move KV cache to host RAM")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
