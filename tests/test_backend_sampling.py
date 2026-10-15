"""Generation controls for real-model runs.

The benchmark is greedy. Positive-temperature sampling remains a generic
backend capability for experiments outside the benchmark protocol.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.rq3a import make_synthetic_gts


class TestGenKwargs:
    def test_greedy_by_default(self):
        from analysis.backends.local_hf import gen_kwargs
        assert gen_kwargs() == {"max_new_tokens": 65536, "do_sample": False}

    def test_sampling_enabled_when_temperature_positive(self):
        from analysis.backends.local_hf import gen_kwargs
        kw = gen_kwargs(temperature=0.7, max_new_tokens=512)
        assert kw == {"max_new_tokens": 512, "do_sample": True,
                      "temperature": 0.7}

    def test_zero_temperature_stays_greedy(self):
        from analysis.backends.local_hf import gen_kwargs
        assert gen_kwargs(temperature=0.0)["do_sample"] is False


class TestRunnerThreadsGenOpts:
    def test_backend_receives_seed_and_temperature(self, tmp_path):
        from analysis.run_model import run_model, BACKENDS
        gt_dir = tmp_path / "gt"
        gt = make_synthetic_gts(1)[0]
        d = gt_dir / "scenario_1"
        d.mkdir(parents=True)
        with open(d / "ground_truth.json", "w") as f:
            json.dump(gt, f)

        seen = {}

        def spy(prompt, frame_paths, model_id, opts):
            seen.update(opts)
            return json.dumps({"predicted_branches": [],
                               "predicted_counterfactuals": []})

        BACKENDS["spy"] = spy
        try:
            run_model(str(gt_dir), str(tmp_path / "r"), backend="spy",
                      seed=7, temperature=0.7)
        finally:
            del BACKENDS["spy"]
        assert seen["seed"] == 7
        assert seen["temperature"] == 0.7

    def test_greedy_run_does_not_forward_run_id_as_sampling_seed(self, tmp_path):
        from analysis.run_model import run_model, BACKENDS
        gt_dir = tmp_path / "gt"
        gt = make_synthetic_gts(1)[0]
        d = gt_dir / "scenario_1"
        d.mkdir(parents=True)
        with open(d / "ground_truth.json", "w") as f:
            json.dump(gt, f)

        seen = {}

        def spy(prompt, frame_paths, model_id, opts):
            seen.update(opts)
            return json.dumps({"predicted_branches": [],
                               "predicted_counterfactuals": []})

        BACKENDS["spy"] = spy
        try:
            run_model(str(gt_dir), str(tmp_path / "r"), backend="spy",
                      seed=0, temperature=0.0)
        finally:
            del BACKENDS["spy"]
        assert seen == {"seed": None, "temperature": 0.0}
        manifest = json.load(open(tmp_path / "r" / "scenario_1" /
                                  "run_manifest.json"))
        assert manifest["seed"] == 0
        assert manifest["generation_seed"] is None
        assert manifest["temperature"] == 0.0
        assert manifest["decoding"] == "greedy"


def test_legacy_default_rope_matches_transformers4_formula():
    """Molmo2 remote code needs ROPE_INIT_FUNCTIONS['default'] (removed in the
    pinned Transformers 5.16). The shim reproduces the 4.x default exactly."""
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from analysis.backends.local_hf import _default_rope_parameters

    cfg = SimpleNamespace(rope_theta=1_000_000.0, head_dim=128, hidden_size=3584,
                          num_attention_heads=28)
    inv_freq, scaling = _default_rope_parameters(cfg, device="cpu", seq_len=4096)
    expected = 1.0 / (1_000_000.0 ** (torch.arange(0, 128, 2, dtype=torch.int64).float() / 128))
    assert scaling == 1.0
    assert inv_freq.shape == (64,)
    assert torch.allclose(inv_freq, expected)
    # Without head_dim the rotary dim falls back to hidden_size / heads.
    cfg2 = SimpleNamespace(rope_theta=10_000.0, hidden_size=4096, num_attention_heads=32)
    assert _default_rope_parameters(cfg2)[0].shape == (64,)


def test_register_legacy_default_rope_is_idempotent_and_in_place():
    transformers = pytest.importorskip("transformers")
    from transformers import modeling_rope_utils
    from analysis.backends.local_hf import _register_legacy_default_rope, _default_rope_parameters

    table = modeling_rope_utils.ROPE_INIT_FUNCTIONS
    had = "default" in table
    before = table.get("default")
    _register_legacy_default_rope()
    _register_legacy_default_rope()
    assert modeling_rope_utils.ROPE_INIT_FUNCTIONS is table          # same dict object
    assert "default" in table
    if had:
        assert table["default"] is before                             # never overrides an in-tree entry
    else:
        assert table["default"] is _default_rope_parameters
        del table["default"]


def test_legacy_optional_attributes_shim_pops_and_assigns():
    """Molmo2Processor passes 4.x-style optional_attributes to
    ProcessorMixin.__init__; the pinned 5.x mixin rejects them."""
    pytest.importorskip("transformers")
    from transformers.processing_utils import ProcessorMixin
    from analysis.backends.local_hf import _register_legacy_optional_attributes

    _register_legacy_optional_attributes()
    _register_legacy_optional_attributes()          # idempotent
    assert getattr(ProcessorMixin.__init__, "_coriva_legacy_optional_attributes", False)

    seen = {}

    class Legacy(ProcessorMixin):
        attributes = ["tokenizer"]
        optional_attributes = ["chat_template", "image_use_col_tokens", "time_mode"]
        tokenizer_class = "AutoTokenizer"

        def __init__(self, tokenizer=None, image_use_col_tokens=True, **kwargs):
            # Mirror Molmo2: forward the option to the mixin as a kwarg.
            ProcessorMixin.__init__(self, tokenizer, image_use_col_tokens=image_use_col_tokens, **kwargs)
            seen["done"] = True

    class Tok:  # minimal stand-in; the mixin only stores it
        init_kwargs = {}

    try:
        proc = Legacy(Tok(), image_use_col_tokens=False)
    except TypeError as exc:
        if "Unexpected keyword argument" in str(exc):
            raise                                   # the regression this shim removes
        pytest.skip(f"mixin rejected the stand-in tokenizer, not the option: {exc}")
    assert seen["done"] and proc.image_use_col_tokens is False
    assert proc.time_mode is None                    # unspecified optional -> None, as in 4.x


def test_cache_position_shim_fills_prefill_and_decode_positions():
    """Molmo2's prepare_inputs_for_generation indexes cache_position[0];
    Transformers 5 passes None. The shim recreates the base method's value."""
    torch = pytest.importorskip("torch")
    from analysis.backends.local_hf import _shim_cache_position

    calls = []

    class Remote:
        def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                          cache_position=None, **kwargs):
            calls.append(cache_position)
            return {"input_ids": input_ids}

    class Cache:
        def __init__(self, n): self.n = n
        def get_seq_length(self): return self.n

    model = Remote()
    model.prepare_inputs_for_generation = model.prepare_inputs_for_generation  # bound, as on a real module
    _shim_cache_position(model)
    ids = torch.zeros((1, 5), dtype=torch.long)
    model.prepare_inputs_for_generation(ids, past_key_values=None, is_first_iteration=True)
    model.prepare_inputs_for_generation(ids, past_key_values=Cache(5), next_sequence_length=1)
    model.prepare_inputs_for_generation(ids, past_key_values=Cache(9), cache_position=torch.tensor([9]))
    assert calls[0].tolist() == [0, 1, 2, 3, 4]        # prefill: starts at 0 -> pixel inputs attached
    assert calls[1].tolist() == [5]                    # decode: offset by cache length
    assert calls[2].tolist() == [9]                    # caller-supplied value untouched

    class NoCachePosition:
        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {}
    plain = NoCachePosition(); before = plain.prepare_inputs_for_generation
    _shim_cache_position(plain)
    assert plain.prepare_inputs_for_generation == before                # untouched


def test_legacy_mask_kwargs_renames_input_embeds_and_drops_cache_position():
    from analysis.backends.local_hf import _legacy_mask_kwargs

    seen = {}

    def create_causal_mask(config, inputs_embeds, attention_mask, past_key_values, position_ids=None):
        seen.update(config=config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                    past_key_values=past_key_values, position_ids=position_ids)
        return "mask"

    wrapped = _legacy_mask_kwargs(create_causal_mask)
    assert _legacy_mask_kwargs(wrapped) is wrapped                      # idempotent
    # Molmo2's 4.5x-style call
    assert wrapped(config="cfg", input_embeds="emb", attention_mask=None, cache_position="cp",
                   past_key_values="cache", position_ids="pos") == "mask"
    assert seen == dict(config="cfg", inputs_embeds="emb", attention_mask=None,
                        past_key_values="cache", position_ids="pos")
    # New-style callers are untouched
    assert wrapped("cfg", "emb2", None, None) == "mask" and seen["inputs_embeds"] == "emb2"


def test_register_legacy_mask_kwargs_patches_module_namespaces():
    import types
    pytest.importorskip("transformers")
    from transformers import masking_utils
    from analysis.backends.local_hf import _register_legacy_mask_kwargs, _LEGACY_MASK_FUNCTIONS

    remote = types.ModuleType("fake_remote_molmo2")
    remote.create_causal_mask = masking_utils.create_causal_mask
    _register_legacy_mask_kwargs(remote)
    for name in _LEGACY_MASK_FUNCTIONS:
        assert getattr(getattr(masking_utils, name), "_coriva_legacy_mask_kwargs", False)
    assert getattr(remote.create_causal_mask, "_coriva_legacy_mask_kwargs", False)


def test_reinit_remote_rotary_buffers_recomputes_zeroed_inv_freq():
    """Meta-device loading leaves remote rotary buffers uninitialised; the
    adapter recomputes them with the module's own rope_init_fn."""
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from analysis.backends.local_hf import _default_rope_parameters, _reinit_remote_rotary_buffers

    class Rotary(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(rope_theta=10_000.0, head_dim=8)
            self.rope_init_fn = _default_rope_parameters
            self.register_buffer("inv_freq", torch.zeros(4), persistent=False)   # as materialised from meta
            self.original_inv_freq = self.inv_freq
            self.attention_scaling = 0.0

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary = Rotary()
            self.linear = torch.nn.Linear(2, 2)                                  # no rope: untouched

    model = Model()
    assert _reinit_remote_rotary_buffers(model) == 1
    assert model.rotary.inv_freq[0].item() == 1.0
    assert torch.allclose(model.rotary.inv_freq, _default_rope_parameters(model.rotary.config)[0])
    assert model.rotary.original_inv_freq is model.rotary.inv_freq
    assert model.rotary.attention_scaling == 1.0
    assert "inv_freq" in dict(model.rotary.named_buffers())                     # still a buffer, still non-persistent
    assert "rotary.inv_freq" not in model.state_dict()
