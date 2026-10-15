"""Portable font contract for labelled nuScenes review overlays."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_overlay_font_is_bundled_with_the_project():
    from simulation.review_overlay import overlay_font_path

    font = Path(overlay_font_path()).resolve()
    project_root = Path(__file__).resolve().parents[1]

    assert font.name == "NotoSans-Regular.ttf"
    assert project_root in font.parents
    assert font.is_file()
    assert font.stat().st_size > 100_000


def test_bundled_overlay_font_renders_with_imagemagick(tmp_path):
    if not __import__("shutil").which("magick"):
        pytest.skip("ImageMagick is not installed")

    import subprocess

    from PIL import Image
    from simulation.review_overlay import overlay_font_path

    source = tmp_path / "source.jpg"
    rendered = tmp_path / "rendered.jpg"
    Image.new("RGB", (320, 180), "black").save(source)

    subprocess.run([
        "magick", str(source),
        "-font", overlay_font_path(),
        "-fill", "red",
        "-pointsize", "26",
        "-annotate", "+20+40", "cand_001",
        str(rendered),
    ], check=True)

    assert rendered.is_file()
    assert rendered.stat().st_size > 0
