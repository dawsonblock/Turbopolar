"""Instance-level adapter that installs TurboPolar into an mlx_lm Llama model."""

from typing import Any, Dict, List, Optional

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.integrations.mlx_lm.attention import TurboPolarLlamaAttention
from rfsn_v11.integrations.mlx_lm.cache import make_turbo_caches
from rfsn_v11.integrations.mlx_lm.support import (
    validate_attention_module,
    validate_llama_model,
)


class TurboPolarLlamaAdapter:
    """Install and remove TurboPolar attention wrappers from a Llama model instance.

    The adapter only touches the supplied model instance. Other model instances in
    the same process are unaffected. Installation is atomic: if any layer fails
    validation, all previously wrapped layers are restored.
    """

    def __init__(
        self,
        turbo_config: Optional[TurboPolarConfig] = None,
    ):
        if turbo_config is None:
            turbo_config = TurboPolarConfig()
        self.turbo_config = turbo_config
        self._installed = False
        self._model: Optional[Any] = None
        self._original_attentions: Dict[int, Any] = {}
        self._wrapped_layer_count = 0

    @property
    def wrapped_layer_count(self) -> int:
        return self._wrapped_layer_count

    @property
    def is_installed(self) -> bool:
        return self._installed

    def install(self, model: Any) -> None:
        """Wrap every Llama attention module in ``model`` for TurboPolar decode.

        Raises:
            ValueError: if the model is unsupported.
            RuntimeError: if the adapter is already installed on a model.
        """
        if self._installed:
            raise RuntimeError(
                "TurboPolarLlamaAdapter is already installed; call uninstall() first."
            )

        num_q_heads, num_kv_heads, head_dim = validate_llama_model(model)

        # Validate model configuration matches the adapter's turbo_config.
        if self.turbo_config.num_q_heads != num_q_heads:
            raise ValueError(
                f"Model has {num_q_heads} query heads but config expects "
                f"{self.turbo_config.num_q_heads}"
            )
        if self.turbo_config.num_kv_heads != num_kv_heads:
            raise ValueError(
                f"Model has {num_kv_heads} KV heads but config expects "
                f"{self.turbo_config.num_kv_heads}"
            )
        if self.turbo_config.head_dim != head_dim:
            raise ValueError(
                f"Model has head_dim={head_dim} but config expects "
                f"{self.turbo_config.head_dim}"
            )

        layers = model.layers if hasattr(model, "layers") else model.model.layers
        if len(layers) == 0:
            raise ValueError("Model has no layers.")

        original_attentions: Dict[int, Any] = {}
        try:
            for i, layer in enumerate(layers):
                if i in self._original_attentions:
                    continue  # already installed
                attention = getattr(
                    layer, "attention", getattr(layer, "self_attn", None)
                )
                if attention is None:
                    raise ValueError(f"Layer {i}: no attention/self_attn attribute.")
                validate_attention_module(attention, i)
                wrapped = TurboPolarLlamaAttention(
                    original_attention=attention,
                    turbo_config=self.turbo_config,
                    layer_index=i,
                )
                if hasattr(layer, "attention"):
                    layer.attention = wrapped
                else:
                    layer.self_attn = wrapped
                original_attentions[i] = attention
        except Exception:
            # Roll back any partial installation before re-raising.
            for idx, orig in original_attentions.items():
                layer = layers[idx]
                if hasattr(layer, "attention"):
                    layer.attention = orig
                else:
                    layer.self_attn = orig
            raise

        self._model = model
        self._original_attentions = original_attentions
        self._wrapped_layer_count = len(original_attentions)
        self._installed = True

    def uninstall(self) -> None:
        """Restore the original attention modules."""
        if not self._installed or self._model is None:
            self._installed = False
            return

        layers = (
            self._model.layers
            if hasattr(self._model, "layers")
            else self._model.model.layers
        )
        for i, orig in self._original_attentions.items():
            layer = layers[i]
            if hasattr(layer, "attention"):
                layer.attention = orig
            else:
                layer.self_attn = orig

        self._original_attentions = {}
        self._wrapped_layer_count = 0
        self._model = None
        self._installed = False

    def make_caches(self, num_layers: int) -> List[Any]:
        """Create a TurboPolarFastCache for each layer.

        The config is derived from the adapter's turbo_config but with model head
        counts injected if they differ.
        """
        return make_turbo_caches(
            num_layers=num_layers,
            num_q_heads=self.turbo_config.num_q_heads,
            num_kv_heads=self.turbo_config.num_kv_heads,
            head_dim=self.turbo_config.head_dim,
            use_qjl=self.turbo_config.use_qjl,
            execution_mode=self.turbo_config.execution_mode,
        )
