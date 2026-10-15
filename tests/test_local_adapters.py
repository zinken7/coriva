"""Step 3 local-adapter contracts without loading GPU dependencies."""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.backends import local_hf
from analysis.backends.result import (GenerationResult, result_from_raw,
                                      split_thinking_trace)


def test_adapter_registry_uses_checkpoint_documented_classes():
    assert local_hf.ADAPTERS["qwen3_vl"].model_class == \
        "Qwen3VLForConditionalGeneration"
    assert local_hf.ADAPTERS["cosmos_reason2"].model_class == \
        "Qwen3VLForConditionalGeneration"
    assert local_hf.ADAPTERS["qwen35"].model_class == \
        "Qwen3_5ForConditionalGeneration"
    assert local_hf.ADAPTERS["molmo2"].model_class == \
        "AutoModelForImageTextToText"
    assert local_hf.ADAPTERS["molmo2"].trust_remote_code is True


def test_message_builders_preserve_image_order_without_video_resampling():
    images = ["frame-0", "frame-1", "frame-2"]
    qwen = local_hf.build_messages("prompt", images, "qwen3_vl")[0]["content"]
    molmo = local_hf.build_messages("prompt", images, "molmo2")[0]["content"]
    assert [item.get("image") for item in qwen if item["type"] == "image"] == images
    assert [item.get("image") for item in molmo if item["type"] == "image"] == images
    assert qwen[-1] == {"type": "text", "text": "prompt"}
    assert molmo[0] == {"type": "text", "text": "prompt"}
    assert not any(item["type"] == "video" for item in qwen + molmo)


def test_qwen35_requires_explicit_direct_mode_only_for_its_adapter():
    assert local_hf.chat_template_kwargs("qwen35", False) == \
        {"enable_thinking": False}
    for value in (None, True):
        with pytest.raises(ValueError, match="direct mode"):
            local_hf.chat_template_kwargs("qwen35", value)
    with pytest.raises(ValueError, match="does not accept"):
        local_hf.chat_template_kwargs("molmo2", False)


def test_thinking_split_never_sends_unclosed_trace_to_scorer():
    trace, final = split_thinking_trace("<think>reason</think>\n{\"ok\": true}")
    assert trace == "<think>reason</think>"
    assert final == '{"ok": true}'
    trace, final = split_thinking_trace(
        "  <think>reason</think>\n{\"ok\": true}")
    assert trace == "  <think>reason</think>"
    assert final == '{"ok": true}'
    trace, final = split_thinking_trace("<think>unfinished")
    assert trace == "<think>unfinished" and final == ""
    plain = result_from_raw('{"ok": true}', separate_thinking=True)
    assert plain.thinking_trace is None and plain.final_text == '{"ok": true}'
    embedded = result_from_raw(
        '{"literal": "<think> is data"}', separate_thinking=True)
    assert embedded.thinking_trace is None
    assert embedded.final_text == '{"literal": "<think> is data"}'


class _FakeInputs(dict):
    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, device):
        self["moved_to"] = device
        return self


class _FakeProcessor:
    def __init__(self, raw_text):
        self.raw_text = raw_text
        self.template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        return _FakeInputs(input_ids=[[11, 12]])

    def batch_decode(self, rows, **kwargs):
        self.decoded_rows = rows
        return [self.raw_text]


class _FakeModel:
    device = "cuda:0"
    config = SimpleNamespace(_commit_hash="pinned-sha")

    def generate(self, **kwargs):
        self.generation_kwargs = kwargs
        return [[11, 12, 21, 22, 23]]


class _FakeCuda:
    def is_available(self):
        return True

    def reset_peak_memory_stats(self):
        self.reset = True

    def max_memory_allocated(self):
        return 123456


class _InferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_qwen35_generation_threads_mode_revision_cap_and_separates_trace(monkeypatch):
    raw = ('<think>private reasoning</think>\n'
           '{"predicted_branches": [], "predicted_counterfactuals": []}')
    processor = _FakeProcessor(raw)
    model = _FakeModel()
    fake_torch = SimpleNamespace(
        cuda=_FakeCuda(), inference_mode=lambda: _InferenceMode(),
        manual_seed=lambda seed: None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(local_hf, "_load", lambda *args: (model, processor))
    monkeypatch.setattr(local_hf, "_open_images", lambda paths: list(paths))

    result = local_hf.generate_result(
        "prompt", ["000.jpg", "001.jpg"], "Qwen/Qwen3.5-9B",
        adapter="qwen35", revision="a" * 40, enable_thinking=False,
        max_new_tokens=32768, temperature=0.0)

    assert isinstance(result, GenerationResult)
    assert result.thinking_trace == "<think>private reasoning</think>"
    assert result.final_text.startswith('{"predicted_branches"')
    assert processor.template_kwargs["enable_thinking"] is False
    assert model.generation_kwargs["max_new_tokens"] == 32768
    assert model.generation_kwargs["do_sample"] is False
    assert result.metadata["input_tokens"] == 2
    assert result.metadata["output_tokens"] == 3
    assert result.metadata["peak_cuda_memory_bytes"] == 123456
    assert result.metadata["ordered_image_count"] == 2


@pytest.mark.parametrize("cache", [None, "offloaded"])
def test_qwen_cache_option_preserves_images_and_decoding(monkeypatch, cache):
    model, processor = _FakeModel(), _FakeProcessor('{"predicted_branches": []}')
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=_FakeCuda(), inference_mode=lambda: _InferenceMode()))
    monkeypatch.setattr(local_hf, "_load", lambda *args: (model, processor))
    monkeypatch.setattr(local_hf, "_open_images", lambda paths: list(paths))
    images = [f"{i:03d}.png" for i in range(8)]
    result = local_hf.generate_result("prompt", images, "model", adapter="qwen3_vl",
                                     revision="a" * 40, cache_implementation=cache)
    assert model.generation_kwargs.get("cache_implementation") == cache
    assert ("cache_implementation" in model.generation_kwargs) == (cache is not None)
    assert model.generation_kwargs["max_new_tokens"] == 65536   # the panel cap
    assert model.generation_kwargs["do_sample"] is False
    assert result.metadata["cache_implementation"] == cache
    assert result.metadata["ordered_image_count"] == 8
    assert [x["image"] for x in processor.messages[0]["content"] if x["type"] == "image"] == images


@pytest.mark.parametrize("adapter", ["qwen3_vl", "cosmos_reason2", "molmo2"])
def test_every_local_adapter_separates_an_explicit_leading_trace(
        monkeypatch, adapter):
    raw = '<think>reasoning</think>\n{"predicted_branches": []}'
    processor = _FakeProcessor(raw)
    model = _FakeModel()
    fake_torch = SimpleNamespace(
        cuda=_FakeCuda(), inference_mode=lambda: _InferenceMode(),
        manual_seed=lambda seed: None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(local_hf, "_load", lambda *args: (model, processor))
    monkeypatch.setattr(local_hf, "_open_images", lambda paths: list(paths))

    result = local_hf.generate_result(
        "prompt", ["000.jpg"], "org/model", adapter=adapter,
        revision="a" * 40, max_new_tokens=32768, temperature=0.0)

    assert result.thinking_trace == "<think>reasoning</think>"
    assert result.final_text == '{"predicted_branches": []}'


@pytest.mark.parametrize("adapter,class_name,trust", [
    ("qwen3_vl", "Qwen3VLForConditionalGeneration", False),
    ("cosmos_reason2", "Qwen3VLForConditionalGeneration", False),
    ("qwen35", "Qwen3_5ForConditionalGeneration", False),
    ("molmo2", "AutoModelForImageTextToText", True),
])
def test_loader_passes_exact_revision_to_model_and_processor(
        monkeypatch, adapter, class_name, trust):
    calls = []

    class Factory:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append((cls.__name__, model_id, kwargs))
            return object()

    factories = {name: type(name, (Factory,), {}) for name in {
        "Qwen3VLForConditionalGeneration", "Qwen3_5ForConditionalGeneration",
        "AutoModelForImageTextToText", "AutoProcessor"}}
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(**factories))
    # The Molmo2 runtime shims (commits f49f0c6..7383f39) patch real
    # Transformers internals and walk model.modules(); they are covered by
    # their own tests. This test checks only what the loader passes to
    # from_pretrained, so the shims are no-ops here.
    for shim in ("_register_legacy_default_rope", "_register_legacy_optional_attributes",
                 "_register_legacy_mask_kwargs", "_shim_cache_position", "_reinit_remote_rotary_buffers"):
        monkeypatch.setattr(local_hf, shim, lambda *a, **k: None)
    local_hf._CACHE.clear()
    revision = "b" * 40
    local_hf._load(adapter, "org/model", revision)
    assert calls[0][0] == class_name
    assert calls[0][2]["revision"] == revision
    assert calls[0][2]["dtype"] == "bfloat16"
    assert calls[1][0] == "AutoProcessor"
    assert calls[1][2]["revision"] == revision
    assert (calls[0][2].get("trust_remote_code") is True) == trust
    assert (calls[1][2].get("trust_remote_code") is True) == trust
    local_hf._CACHE.clear()
