"""Backwards-compatible re-exports for benchmark scripts.

Runtime components have moved to rfsn_v11.integrations.mlx_lm.*.
Import from there instead of from this benchmark module.
"""

import warnings

from rfsn_v11.integrations.mlx_lm.cache import (  # noqa: F401
    TurboPolarFastCache,
    make_turbo_caches,
)

warnings.warn(
    "benchmarks.turbopolar_fast_attention is deprecated; "
    "import from rfsn_v11.integrations.mlx_lm.cache instead.",
    DeprecationWarning,
    stacklevel=2,
)
