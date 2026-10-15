"""Pinned offline-weight prefetch configuration."""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.prefetch_weights import download_snapshots, panel_snapshots


def test_panel_prefetch_resolves_four_unique_open_revisions():
    snapshots = panel_snapshots("analysis/model_panel.yaml")
    assert len(snapshots) == 4
    assert len({item["repo_id"] for item in snapshots}) == 4
    assert all(len(item["revision"]) == 40 for item in snapshots)
    cosmos = next(item for item in snapshots if item["key"] == "cosmos-reason2-8b")
    assert cosmos["gated"] is True


def test_panel_prefetch_rejects_unknown_checkpoint():
    with pytest.raises(ValueError, match="unknown panel checkpoints"):
        panel_snapshots("analysis/model_panel.yaml", ["not-a-checkpoint"])


def test_download_uses_revision_cache_and_token_only_for_gated(monkeypatch):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return "/cache/" + kwargs["repo_id"].replace("/", "--")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_download))
    snapshots = [
        {"key": "open", "repo_id": "a/open", "revision": "a" * 40,
         "gated": False},
        {"key": "gated", "repo_id": "b/gated", "revision": "b" * 40,
         "gated": True},
    ]
    paths = download_snapshots(snapshots, "/shared/hf")
    assert len(paths) == 2
    assert calls[0] == {"repo_id": "a/open", "revision": "a" * 40,
                        "cache_dir": "/shared/hf"}
    assert calls[1]["token"] is True
