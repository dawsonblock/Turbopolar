"""Validation helpers for mlx_lm Llama integration."""

from typing import Any, Tuple


SUPPORTED_MODULE_NAME = "mlx_lm.models.llama.Attention"


def _module_fqn(obj: Any) -> str:
    cls = type(obj)
    module = getattr(cls, "__module__", "")
    name = getattr(cls, "__qualname__", cls.__name__)
    return f"{module}.{name}"


def validate_llama_model(model: Any) -> Tuple[int, int, int]:
    """Validate that ``model`` is a supported mlx_lm Llama instance.

    Returns:
        (num_q_heads, num_kv_heads, head_dim)

    Raises:
        ValueError: if the model architecture is not supported.
    """
    if model is None:
        raise ValueError("model is None")

    layers = None
    if hasattr(model, "layers"):
        layers = model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers

    if layers is None:
        raise ValueError(
            "Model has no layers attribute; not a supported Llama implementation."
        )

    if len(layers) == 0:
        raise ValueError("Model has no layers.")

    first_layer = layers[0]
    attention = getattr(
        first_layer, "attention", getattr(first_layer, "self_attn", None)
    )
    if attention is None:
        raise ValueError(
            "Layer has no attention/self_attn attribute; not a supported Llama implementation."
        )

    fqn = _module_fqn(attention)
    if fqn != SUPPORTED_MODULE_NAME:
        raise ValueError(
            f"TurboPolar adapter only supports {SUPPORTED_MODULE_NAME}, got {fqn}"
        )

    n_heads = getattr(attention, "n_heads", None)
    n_kv_heads = getattr(attention, "n_kv_heads", None)
    if n_heads is None or n_kv_heads is None:
        raise ValueError("Attention module missing n_heads or n_kv_heads.")

    if n_heads % n_kv_heads != 0:
        raise ValueError(
            f"GQA ratio does not divide: n_heads={n_heads}, n_kv_heads={n_kv_heads}"
        )

    # Infer head_dim from q_proj weight shape: [hidden_size, head_dim] per head?
    # In mlx_lm Llama, q_proj.weight shape is (n_heads * head_dim, hidden_size).
    # We can compute head_dim = q_proj.weight.shape[0] // n_heads.
    q_proj = getattr(attention, "q_proj", None)
    if q_proj is None or not hasattr(q_proj, "weight"):
        raise ValueError("Attention module missing q_proj weight.")

    hidden_size = q_proj.weight.shape[0]
    if hidden_size % n_heads != 0:
        raise ValueError(
            f"hidden_size {hidden_size} not divisible by n_heads {n_heads}"
        )

    head_dim = hidden_size // n_heads
    if head_dim != 128:
        raise ValueError(
            f"TurboPolar adapter only supports head_dim=128, got {head_dim}"
        )

    return int(n_heads), int(n_kv_heads), int(head_dim)


def validate_attention_module(attn: Any, layer_index: int):
    """Validate a single attention module within a Llama model."""
    fqn = _module_fqn(attn)
    if fqn != SUPPORTED_MODULE_NAME:
        raise ValueError(
            f"Layer {layer_index}: expected {SUPPORTED_MODULE_NAME}, got {fqn}"
        )

    required = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "rope",
        "n_heads",
        "n_kv_heads",
        "scale",
    )
    for attr in required:
        if not hasattr(attn, attr):
            raise ValueError(f"Layer {layer_index}: attention module missing {attr}")

    if attn.n_heads % attn.n_kv_heads != 0:
        raise ValueError(
            f"Layer {layer_index}: GQA ratio does not divide: "
            f"n_heads={attn.n_heads}, n_kv_heads={attn.n_kv_heads}"
        )
