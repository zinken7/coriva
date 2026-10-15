"""Pinned Hugging Face adapters for the four CORIVA open checkpoints.

Heavy imports remain lazy.  All adapters receive the same ordered images and
text; only the checkpoint's documented loader/chat-template mechanics differ.
The benchmark never converts the eight released images to an MP4.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from analysis.backends.result import GenerationResult, result_from_raw
from analysis.model_panel import MAX_OUTPUT_TOKENS, TRANSFORMERS_RUNTIME_REVISION

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
TRANSFORMERS_REVISION = TRANSFORMERS_RUNTIME_REVISION


@dataclass(frozen=True)
class AdapterDefinition:
    model_class: str
    trust_remote_code: bool = False
    requires_direct_mode: bool = False
    text_first: bool = False


ADAPTERS: Dict[str, AdapterDefinition] = {
    "qwen3_vl": AdapterDefinition("Qwen3VLForConditionalGeneration"),
    "cosmos_reason2": AdapterDefinition("Qwen3VLForConditionalGeneration"),
    "qwen35": AdapterDefinition(
        "Qwen3_5ForConditionalGeneration", requires_direct_mode=True),
    "molmo2": AdapterDefinition(
        "AutoModelForImageTextToText", trust_remote_code=True, text_first=True),
}

_CACHE: Dict[Tuple[str, str, Optional[str]], Tuple[Any, Any]] = {}


def adapter_definition(adapter: str) -> AdapterDefinition:
    try:
        return ADAPTERS[adapter]
    except KeyError as exc:
        raise ValueError(
            f"unsupported local adapter {adapter!r}; expected {sorted(ADAPTERS)}") from exc


def gen_kwargs(seed: Optional[int] = None, temperature: Optional[float] = None,
               max_new_tokens: int = MAX_OUTPUT_TOKENS) -> Dict[str, Any]:
    """Generation kwargs: one greedy run for the protocol."""
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    kw: Dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if temperature and temperature > 0:
        kw["do_sample"] = True
        kw["temperature"] = float(temperature)
    else:
        kw["do_sample"] = False
    return kw


def _default_rope_parameters(config: Any, device: Any = None,
                             seq_len: Optional[int] = None,
                             **_: Any) -> Tuple[Any, float]:
    """Plain RoPE inverse frequencies for remote code written against
    Transformers 4.x, where ``ROPE_INIT_FUNCTIONS["default"]`` existed.

    The pinned Transformers 5.16 removed that entry (each in-tree model now
    computes the default on its own class), so Molmo2's ``modeling_molmo2.py``
    fails at ``ROPE_INIT_FUNCTIONS[self.rope_type]`` with ``KeyError:
    'default'``. Mirrors the removed
    ``_compute_default_rope_parameters``: ``theta ** (-2i/dim)`` over the
    rotary dimension, attention scaling 1.0. ``seq_len`` is accepted and
    ignored, as in the original, because ``dynamic_rope_update`` passes it."""
    import torch

    base = getattr(config, "rope_theta", None)
    if base is None:
        base = (getattr(config, "rope_parameters", None) or {}).get("rope_theta")
    if base is None:
        raise ValueError("default RoPE needs rope_theta on the config")
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64)
                               .to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0


def _register_legacy_default_rope() -> None:
    """Restore ``ROPE_INIT_FUNCTIONS["default"]`` for remote-code adapters.

    Mutates the dict in place so modules that already imported the name (the
    remote ``modeling_molmo2`` does ``from transformers.modeling_rope_utils
    import ROPE_INIT_FUNCTIONS``) see it too. ``dynamic_rope_update`` only
    acts on ``dynamic``/``longrope`` types, so the ``default`` entry is used
    once, at construction."""
    from transformers import modeling_rope_utils

    modeling_rope_utils.ROPE_INIT_FUNCTIONS.setdefault(
        "default", _default_rope_parameters)


def _register_legacy_optional_attributes() -> None:
    """Accept the Transformers 4.x ``optional_attributes`` protocol in
    ``ProcessorMixin.__init__``.

    In 4.x a processor listed extra constructor options in a class attribute
    ``optional_attributes`` and the mixin popped them from kwargs and set them
    on the instance. The pinned 5.16 mixin raises ``TypeError: Unexpected
    keyword argument`` for any kwarg that is not a modality attribute, which
    is where Molmo2's remote ``Molmo2Processor.__init__`` (five such options,
    e.g. ``image_use_col_tokens``) fails. The wrapper
    pops those names before delegating and assigns them afterwards, leaving
    ``chat_template`` to the mixin. Idempotent; in-tree processors have no
    ``optional_attributes`` and are untouched."""
    import functools

    from transformers.processing_utils import ProcessorMixin

    original = ProcessorMixin.__init__
    if getattr(original, "_coriva_legacy_optional_attributes", False):
        return

    @functools.wraps(original)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        names = [name for name in (getattr(type(self), "optional_attributes", None) or ())
                 if name != "chat_template"]
        legacy = {name: kwargs.pop(name, None) for name in names}
        original(self, *args, **kwargs)
        for name, value in legacy.items():
            setattr(self, name, value)

    __init__._coriva_legacy_optional_attributes = True  # type: ignore[attr-defined]
    ProcessorMixin.__init__ = __init__  # type: ignore[method-assign]


_LEGACY_MASK_FUNCTIONS = ("create_causal_mask", "create_masks_for_generate")


def _legacy_mask_kwargs(function: Any) -> Any:
    """Accept the Transformers 4.5x calling convention for the mask builders.

    Remote Molmo2 calls ``create_causal_mask(config=..., input_embeds=...,
    cache_position=..., ...)``. The pinned 5.16 builders take
    ``inputs_embeds`` and have no ``cache_position`` (they derive positions
    from the cache), so the call failed with ``TypeError: unexpected keyword
    argument 'input_embeds'``. Renames the one and drops
    the other; every other argument passes through."""
    import functools

    if getattr(function, "_coriva_legacy_mask_kwargs", False):
        return function

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        kwargs.pop("cache_position", None)
        return function(*args, **kwargs)

    wrapper._coriva_legacy_mask_kwargs = True  # type: ignore[attr-defined]
    return wrapper


def _register_legacy_mask_kwargs(*modules: Any) -> None:
    """Install :func:`_legacy_mask_kwargs` on ``transformers.masking_utils``
    and on any remote module that already imported the names (``from
    transformers.masking_utils import create_causal_mask`` binds at import
    time, so the module namespace must be patched too)."""
    from transformers import masking_utils

    for module in (masking_utils, *modules):
        for name in _LEGACY_MASK_FUNCTIONS:
            function = getattr(module, name, None)
            if callable(function):
                setattr(module, name, _legacy_mask_kwargs(function))


def _reinit_remote_rotary_buffers(model: Any) -> int:
    """Recompute rotary ``inv_freq`` buffers of a remote-code model.

    Transformers 5 builds the model on the ``meta`` device and materialises
    it while loading weights. Buffers that are computed in ``__init__`` and
    not stored in the checkpoint (``inv_freq`` of every rotary embedding) come
    back as uninitialised memory; in-tree models recompute them in
    ``_init_weights``, remote Molmo2 does not, so its RoPE table was all
    zeros and long (image) prompts degenerated while short text prompts still
    read fine. Calls each module's own ``rope_init_fn``
    exactly as its ``__init__`` did and refreshes ``original_inv_freq`` and
    ``attention_scaling``. Returns the number of modules refreshed."""
    import torch

    refreshed = 0
    with torch.no_grad():
        for module in model.modules():
            init_fn = getattr(module, "rope_init_fn", None)
            buffer = getattr(module, "inv_freq", None)
            if not callable(init_fn) or not isinstance(buffer, torch.Tensor):
                continue
            inv_freq, attention_scaling = init_fn(module.config, buffer.device)
            module.inv_freq = inv_freq.to(device=buffer.device, dtype=buffer.dtype)
            module.original_inv_freq = module.inv_freq
            module.attention_scaling = attention_scaling
            refreshed += 1
    return refreshed


def _shim_cache_position(model: Any) -> None:
    """Supply ``cache_position`` to a remote ``prepare_inputs_for_generation``.

    Transformers 5 no longer passes ``cache_position`` into
    ``prepare_inputs_for_generation`` (the base method recreates it inside
    ``model_inputs`` for remote models that declare it on ``forward``).
    Molmo2's override tests ``cache_position[0] == 0`` on the argument itself
    to attach pixel inputs on the prefill step only, so it crashed with
    ``TypeError: 'NoneType' object is not subscriptable``.
    The wrapper computes the same value the base method does: an arange over
    this step's tokens offset by the cache's seen length. Only the first
    element matters to the override. No-op for models whose override does not
    take ``cache_position``."""
    import functools
    import inspect

    original = model.prepare_inputs_for_generation
    if "cache_position" not in inspect.signature(original).parameters:
        return

    @functools.wraps(original)
    def prepare_inputs_for_generation(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("cache_position") is None:
            import torch

            input_ids = args[0] if args else kwargs.get("input_ids")
            past = kwargs.get("past_key_values")
            past_seen = past.get_seq_length() if hasattr(past, "get_seq_length") else 0
            next_len = kwargs.get("next_sequence_length")
            seq_len = int(next_len) if next_len is not None else int(input_ids.shape[1])
            kwargs["cache_position"] = torch.arange(
                seq_len, device=input_ids.device) + past_seen
        return original(*args, **kwargs)

    model.prepare_inputs_for_generation = prepare_inputs_for_generation


def _load(adapter: str, model_id: str,
          revision: Optional[str]) -> Tuple[Any, Any]:
    """Load one pinned model/processor pair and cache it for the process."""
    definition = adapter_definition(adapter)
    cache_key = (adapter, model_id, revision)
    if cache_key not in _CACHE:
        import transformers

        if definition.trust_remote_code:
            _register_legacy_default_rope()
            _register_legacy_optional_attributes()
            _register_legacy_mask_kwargs()

        try:
            model_class = getattr(transformers, definition.model_class)
        except AttributeError as exc:
            raise RuntimeError(
                f"the pinned Transformers runtime lacks {definition.model_class}; "
                f"install requirements-models.txt at revision {TRANSFORMERS_REVISION}") from exc
        processor_class = transformers.AutoProcessor
        common: Dict[str, Any] = {"revision": revision}
        if definition.trust_remote_code:
            # Remote code is constrained by the same immutable model revision.
            common["trust_remote_code"] = True
        model = model_class.from_pretrained(
            model_id, dtype="bfloat16", device_map="auto", **common)
        if definition.trust_remote_code:
            import sys

            _shim_cache_position(model)
            _reinit_remote_rotary_buffers(model)
            # Remote modules bound the mask builders at import time; rebind
            # them (all Molmo2 classes live in one module).
            remote = sys.modules.get(type(model).__module__)
            if remote is not None:
                _register_legacy_mask_kwargs(remote)
        processor = processor_class.from_pretrained(model_id, **common)
        _CACHE[cache_key] = (model, processor)
    return _CACHE[cache_key]


def build_messages(prompt: str, images: Sequence[Any],
                   adapter: str) -> List[Dict[str, Any]]:
    """Build one turn while preserving the released image order."""
    definition = adapter_definition(adapter)
    image_items = [{"type": "image", "image": image} for image in images]
    text_item = {"type": "text", "text": prompt}
    content = ([text_item] + image_items if definition.text_first
               else image_items + [text_item])
    return [{"role": "user", "content": content}]


def chat_template_kwargs(adapter: str,
                         enable_thinking: Optional[bool]) -> Dict[str, Any]:
    """Return only documented template switches for the selected family."""
    definition = adapter_definition(adapter)
    if definition.requires_direct_mode:
        if enable_thinking is not False:
            raise ValueError("Qwen3.5 requires explicit direct mode (enable_thinking=False)")
        return {"enable_thinking": False}
    if enable_thinking is not None:
        raise ValueError(f"adapter {adapter} does not accept enable_thinking")
    return {}


def _open_images(frame_paths: Sequence[str]) -> List[Any]:
    from PIL import Image

    images: List[Any] = []
    for path in frame_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return images


def _move_inputs(inputs: Any, device: Any) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()}


def _input_ids(inputs: Any) -> Any:
    if hasattr(inputs, "input_ids"):
        return inputs.input_ids
    return inputs["input_ids"]


def _decode(processor: Any, token_rows: Any) -> str:
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(
            token_rows, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
    return processor.tokenizer.batch_decode(
        token_rows, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0]


def _token_count(row: Any) -> int:
    if hasattr(row, "shape"):
        return int(row.shape[-1])
    return len(row)


def _peak_cuda_bytes(torch_module: Any) -> Optional[int]:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return None
    return int(cuda.max_memory_allocated())


def _distribution_vcs_commit(distribution: str) -> Optional[str]:
    """Read the immutable VCS commit recorded by a direct-url installation."""
    try:
        direct_url = importlib.metadata.distribution(distribution).read_text(
            "direct_url.json")
        record = json.loads(direct_url or "{}")
        value = (record.get("vcs_info") or {}).get("commit_id")
        return value if isinstance(value, str) and value else None
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError,
            AttributeError, TypeError):
        return None


def _distribution_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def generate_result(prompt: str, frame_paths: List[str], model_id: str,
                    *, adapter: str, revision: Optional[str],
                    inference_mode: Optional[str] = None,
                    enable_thinking: Optional[bool] = None,
                    max_new_tokens: int = MAX_OUTPUT_TOKENS,
                    seed: Optional[int] = None,
                    temperature: Optional[float] = None,
                    cache_implementation: Optional[str] = None) -> GenerationResult:
    """Run a pinned local configuration and return auditable output metadata."""
    import torch

    if cache_implementation not in (None, "offloaded"):
        raise ValueError("diagnostic cache_implementation must be None or 'offloaded'")
    adapter_definition(adapter)
    model, processor = _load(adapter, model_id or DEFAULT_MODEL, revision)
    images = _open_images(frame_paths)
    messages = build_messages(prompt, images, adapter)
    template_args = chat_template_kwargs(adapter, enable_thinking)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", **template_args)
    inputs = _move_inputs(inputs, model.device)
    input_ids = _input_ids(inputs)
    kwargs = gen_kwargs(seed, temperature, max_new_tokens)
    if cache_implementation is not None:
        kwargs["cache_implementation"] = cache_implementation
    if kwargs["do_sample"] and seed is not None:
        torch.manual_seed(seed)
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(**inputs, **kwargs)
    elapsed = time.monotonic() - started
    trimmed = [out[_token_count(inp):]
               for inp, out in zip(input_ids, generated)]
    raw_text = _decode(processor, trimmed)
    metadata = {
        "adapter": adapter,
        "requested_model_id": model_id or DEFAULT_MODEL,
        "requested_revision": revision,
        "model_commit_hash": getattr(getattr(model, "config", None),
                                     "_commit_hash", None),
        "inference_mode": inference_mode or (
            "direct" if adapter == "qwen35" else "default"),
        "input_tokens": _token_count(input_ids[0]),
        "output_tokens": _token_count(trimmed[0]),
        "max_output_tokens": max_new_tokens,
        "cache_implementation": cache_implementation,
        "latency_seconds": elapsed,
        "peak_cuda_memory_bytes": _peak_cuda_bytes(torch),
        "ordered_image_count": len(frame_paths),
        "transformers_version": _distribution_version("transformers"),
        "transformers_commit_hash": _distribution_vcs_commit("transformers"),
        "torch_version": getattr(torch, "__version__", None),
        "cuda_runtime_version": getattr(getattr(torch, "version", None),
                                        "cuda", None),
        "cuda_device_name": (
            cuda.get_device_name(0)
            if cuda is not None and cuda.is_available() and
            hasattr(cuda, "get_device_name") else None),
        "model_dtype": str(getattr(model, "dtype", None)),
        "python_version": platform.python_version(),
    }
    return result_from_raw(raw_text, separate_thinking=True, metadata=metadata)


def generate(prompt: str, frame_paths: List[str], model_id: str,
             max_new_tokens: int = MAX_OUTPUT_TOKENS,
             seed: Optional[int] = None,
             temperature: Optional[float] = None,
             adapter: str = "qwen3_vl",
             revision: Optional[str] = None,
             inference_mode: Optional[str] = None,
             enable_thinking: Optional[bool] = None) -> str:
    """Compatibility wrapper returning only scorer-visible final text."""
    return generate_result(
        prompt, frame_paths, model_id, adapter=adapter, revision=revision,
        inference_mode=inference_mode,
        enable_thinking=enable_thinking, max_new_tokens=max_new_tokens,
        seed=seed, temperature=temperature).final_text
