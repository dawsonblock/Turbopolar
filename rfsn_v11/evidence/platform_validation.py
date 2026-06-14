"""Platform validation for native Apple Silicon benchmarks."""

import platform
from typing import List, Optional

from rfsn_v11.evidence.provenance import ProvenanceEvidence


def validate_apple_silicon_platform(provenance: ProvenanceEvidence) -> List[str]:
    """Validate that benchmark was run on Apple Silicon platform.

    Args:
        provenance: Provenance evidence to validate

    Returns:
        List of validation error messages. Empty list means validation passed.
    """
    errors = []
    
    # Check macOS version
    if not provenance.macos_version:
        errors.append("macOS version is required for Apple Silicon validation")
    
    # Check chip model
    if not provenance.chip_model:
        errors.append("Chip model is required for Apple Silicon validation")
    
    # Validate chip model is Apple Silicon
    apple_silicon_chips = [
        "M1", "M1 Pro", "M1 Max", "M1 Ultra",
        "M2", "M2 Pro", "M2 Max", "M2 Ultra",
        "M3", "M3 Pro", "M3 Max", "M3 Ultra",
    ]
    
    if provenance.chip_model and provenance.chip_model not in apple_silicon_chips:
        errors.append(
            f"Chip model '{provenance.chip_model}' is not recognized as Apple Silicon. "
            f"Expected one of: {apple_silicon_chips}"
        )
    
    # Check Metal kernel source hash is present
    if not provenance.metal_kernel_source_hash:
        errors.append("Metal kernel source hash is required for Apple Silicon validation")
    
    # Check kernel binding hash is present
    if not provenance.kernel_binding_hash:
        errors.append("Kernel binding hash is required for Apple Silicon validation")
    
    # Validate hash lengths
    if provenance.metal_kernel_source_hash and len(provenance.metal_kernel_source_hash) != 64:
        errors.append(
            f"Metal kernel source hash must be 64 characters, "
            f"got {len(provenance.metal_kernel_source_hash)}"
        )
    
    if provenance.kernel_binding_hash and len(provenance.kernel_binding_hash) != 64:
        errors.append(
            f"Kernel binding hash must be 64 characters, "
            f"got {len(provenance.kernel_binding_hash)}"
        )
    
    return errors


def validate_current_platform_is_apple_silicon() -> List[str]:
    """Validate that current execution platform is Apple Silicon.

    Returns:
        List of validation error messages. Empty list means validation passed.
    """
    errors = []
    
    # Check OS
    if platform.system() != "Darwin":
        errors.append(f"Platform must be macOS (Darwin), got {platform.system()}")
    
    # Check architecture
    if platform.machine() not in ["arm64", "arm64e"]:
        errors.append(f"Architecture must be arm64, got {platform.machine()}")
    
    return errors


def validate_metal_execution_mode(provenance: ProvenanceEvidence) -> List[str]:
    """Validate that Metal execution mode is properly configured.

    Args:
        provenance: Provenance evidence to validate

    Returns:
        List of validation error messages. Empty list means validation passed.
    """
    errors = []
    
    if not provenance.execution_mode:
        errors.append("Execution mode is required")
    
    valid_modes = ["metal_strict", "development_auto", "reference"]
    if provenance.execution_mode and provenance.execution_mode not in valid_modes:
        errors.append(
            f"Execution mode '{provenance.execution_mode}' is invalid. "
            f"Expected one of: {valid_modes}"
        )
    
    # For Apple Silicon, metal_strict is preferred
    if (provenance.execution_mode != "metal_strict" and
        provenance.chip_model and "M" in provenance.chip_model):
        errors.append(
            f"For Apple Silicon benchmarks, execution mode should be 'metal_strict', "
            f"got '{provenance.execution_mode}'"
        )
    
    return errors