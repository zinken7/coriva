"""Tests for the pure helper in the nuScenes downloader (no network)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_resolve_mini_uses_known_url():
    from scripts.download_nuscenes import resolve_download
    url, fname = resolve_download("mini")
    assert url.endswith("v1.0-mini.tgz")
    assert fname == "v1.0-mini.tgz"


def test_resolve_gated_split_without_url_raises():
    from scripts.download_nuscenes import resolve_download
    with pytest.raises(ValueError):
        resolve_download("trainval")


def test_resolve_url_override_derives_filename():
    from scripts.download_nuscenes import resolve_download
    url, fname = resolve_download(
        "trainval", "https://acct.example/dl/v1.0-trainval01_blobs.tgz?token=abc")
    assert url.startswith("https://acct.example")
    assert fname == "v1.0-trainval01_blobs.tgz"
