"""Execution-mode policy for TurboPolar Metal kernels.

Three explicit modes:
- REFERENCE: MLX/Python correctness reference only.
- METAL_STRICT: Every required operation must execute through Metal.
                Any unavailable kernel, compilation error, dispatch error,
                or fallback is fatal.
- DEVELOPMENT_AUTO: Optional local diagnostic mode. May fall back, but
                    can never generate promotion evidence.
"""

from enum import Enum


class ExecutionMode(str, Enum):
    REFERENCE = "reference"
    METAL_STRICT = "metal_strict"
    DEVELOPMENT_AUTO = "development_auto"


class MetalExecutionRequiredError(RuntimeError):
    """Raised when a Metal kernel is required but unavailable in strict mode."""


class MetalKernelInitializationError(RuntimeError):
    """Raised when Metal kernel initialization fails."""


class MetalKernelDispatchError(RuntimeError):
    """Raised when a Metal kernel dispatch fails in strict mode."""
