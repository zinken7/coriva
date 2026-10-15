"""Prefetch exact model-panel revisions for fully offline GPU jobs.

Run this on an internet-enabled login/transfer node.  Compute jobs continue to
address checkpoints by pinned Hugging Face ID + revision and resolve them from
the shared cache; no machine-specific model-path list is needed.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.model_panel import load_model_panel


def panel_snapshots(panel_path: str,
                    checkpoint_keys: Optional[List[str]] = None
                    ) -> List[Dict[str, Any]]:
    panel = load_model_panel(panel_path)
    wanted = set(checkpoint_keys or [])
    known = {item["key"] for item in panel["checkpoints"]}
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(f"unknown panel checkpoints: {unknown}")
    snapshots = []
    for checkpoint in panel["checkpoints"]:
        if checkpoint["source"] != "huggingface":
            continue
        if wanted and checkpoint["key"] not in wanted:
            continue
        snapshots.append({
            "key": checkpoint["key"],
            "repo_id": checkpoint["model_id"],
            "revision": checkpoint["revision"],
            "gated": bool(checkpoint.get("gated")),
        })
    return snapshots


def download_snapshots(snapshots: List[Dict[str, Any]],
                       cache_dir: Optional[str] = None) -> List[str]:
    from huggingface_hub import snapshot_download

    paths = []
    for snapshot in snapshots:
        kwargs: Dict[str, Any] = {
            "repo_id": snapshot["repo_id"],
            "revision": snapshot["revision"],
        }
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if snapshot.get("gated"):
            kwargs["token"] = True
        path = snapshot_download(**kwargs)
        print(f"downloaded {snapshot['key']}@{snapshot['revision']} -> {path}")
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--panel", default="analysis/model_panel.yaml")
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="panel checkpoint key; repeat to download a subset")
    parser.add_argument("--cache-dir", default=None,
                        help="hub cache containing models--ORG--NAME (default: HF_HUB_CACHE or HF_HOME/hub)")
    parser.add_argument("--list", action="store_true",
                        help="print exact snapshots without downloading")
    args = parser.parse_args()
    snapshots = panel_snapshots(args.panel, args.checkpoint)
    if args.list:
        for item in snapshots:
            print(f"{item['key']} {item['repo_id']} {item['revision']}"
                  f" gated={str(item['gated']).lower()}")
        return 0
    download_snapshots(snapshots, args.cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
