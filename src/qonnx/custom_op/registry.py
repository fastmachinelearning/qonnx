# Copyright (c) 2020 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import importlib
import inspect
import warnings
from threading import RLock
from typing import Dict, List, Optional, Tuple, Type
from onnx import NodeProto
from qonnx.custom_op.base import CustomOp

# Nested registry for O(1) lookups: domain -> op_type -> version -> CustomOp class
# Uses "since version" semantics: version N covers opset N until a higher version exists
_OP_REGISTRY: Dict[str, Dict[str, Dict[int, Type[CustomOp]]]] = {}

_REGISTRY_LOCK = RLock()

# Maps ONNX domain names to Python module paths (used for imports only)
_DOMAIN_ALIASES: Dict[str, str] = {
    "onnx.brevitas": "qonnx.custom_op.general",
}


def add_domain_alias(domain: str, module_path: str) -> None:
    """Map a domain name to a different module path.

    Args:
        domain: The ONNX domain name (e.g., "finn.custom_op.fpgadataflow")
        module_path: The Python module path to use instead (e.g., "finn_custom_ops.fpgadataflow")
    """
    with _REGISTRY_LOCK:
        _DOMAIN_ALIASES[domain] = module_path


def resolve_domain(domain: str) -> str:
    """Resolve a domain to its actual module path, handling aliases.

    Args:
        domain: The ONNX domain name

    Returns:
        Resolved module path
    """
    return _DOMAIN_ALIASES.get(domain, domain)


def _get_op_type_for_class(cls: Type[CustomOp]) -> str:
    """Extract the op_type from a CustomOp class name, stripping _vN suffix if present.

    Args:
        cls: CustomOp class

    Returns:
        op_type string (e.g., "IntQuant_v2" -> "IntQuant")
    """
    name = cls.__name__
    # Strip _vN suffix if present
    if "_v" in name:
        parts = name.split("_v")
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]  # IntQuant_v2 -> IntQuant
    return name


def _get_op_version_for_class(cls: Type[CustomOp]) -> int:
    """Extract version from a CustomOp class name.

    Args:
        cls: CustomOp class

    Returns:
        Opset version (defaults to 1 if no _vN suffix present)
    """
    name = cls.__name__
    if "_v" in name:
        parts = name.rsplit("_v", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
    return 1


def _discover_from_custom_op_dict(module, op_type: str, domain: str) -> Dict[int, Type[CustomOp]]:
    """Extract CustomOp versions from legacy custom_op dict (backward compatibility).

    Supports the old registration pattern:
        custom_op = dict()
        custom_op["IntQuant"] = IntQuant
        custom_op["IntQuant_v2"] = IntQuant_v2

    Args:
        module: The imported module to check
        op_type: The specific op type to discover
        domain: The domain name (for warnings)

    Returns:
        Dict mapping version -> CustomOp class
    """
    versions = {}

    if not (hasattr(module, "custom_op") and isinstance(module.custom_op, dict)):
        return versions

    # Iterate all dict entries, filter by op_type
    for key, obj in module.custom_op.items():
        # Check if this dict key matches the requested op_type
        base_name = key.split("_v")[0] if "_v" in key else key
        if base_name != op_type:
            continue

        if not (inspect.isclass(obj) and issubclass(obj, CustomOp) and obj is not CustomOp):
            continue

        try:
            version = _get_op_version_for_class(obj)
        except ValueError as e:
            warnings.warn(str(e))
            continue

        if version in versions:
            warnings.warn(
                f"Multiple classes found for {domain}.{op_type} version {version}: "
                f"{versions[version].__name__} and {obj.__name__}. Using {obj.__name__}."
            )
        versions[version] = obj

    return versions


def _discover_custom_op_versions(domain: str, op_type: str) -> Dict[int, Type[CustomOp]]:
    """Discover all versions of a SPECIFIC custom op without loading entire domain.

    Uses __all__ when available for efficient filtering, otherwise falls back to
    full module inspection. Only loads classes matching the requested op_type.

    Args:
        domain: The ONNX domain name
        op_type: The specific op type to discover

    Returns:
        Dict mapping version -> CustomOp class
    """
    module_path = resolve_domain(domain)
    versions = {}

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        return versions

    # Fast path: use __all__ to find only matching classes
    if hasattr(module, "__all__"):
        # Filter __all__ to find all versions of THIS op_type
        # e.g., op_type="IntQuant" matches ["IntQuant", "IntQuant_v2", "IntQuant_v4"]
        candidates = []
        for name in module.__all__:
            # Strip _vN suffix to check if it matches
            base_name = name.split("_v")[0] if "_v" in name else name
            if base_name == op_type:
                candidates.append(name)

        # Import ONLY the matching classes (lazy loading)
        for name in candidates:
            try:
                obj = getattr(module, name)
            except AttributeError:
                continue

            if not (inspect.isclass(obj) and issubclass(obj, CustomOp) and obj is not CustomOp):
                continue

            try:
                version = _get_op_version_for_class(obj)
            except ValueError as e:
                warnings.warn(str(e))
                continue

            if version in versions:
                warnings.warn(
                    f"Multiple classes found for {domain}.{op_type} version {version}: "
                    f"{versions[version].__name__} and {obj.__name__}. Using {obj.__name__}."
                )
            versions[version] = obj

        # Backward compatibility: if __all__ didn't have the op, try custom_op dict
        if not versions:
            versions = _discover_from_custom_op_dict(module, op_type, domain)

    else:
        # No __all__ - try legacy dict first (O(1) check, cheaper than full scan)
        versions = _discover_from_custom_op_dict(module, op_type, domain)

        # Still nothing? Fallback to full module scan (for external modules)
        if not versions:
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if not issubclass(obj, CustomOp) or obj is CustomOp:
                    continue

                class_op_type = _get_op_type_for_class(obj)
                if class_op_type != op_type:
                    continue

                try:
                    version = _get_op_version_for_class(obj)
                except ValueError as e:
                    warnings.warn(str(e))
                    continue

                if version in versions:
                    warnings.warn(
                        f"Multiple classes found for {domain}.{op_type} version {version}: "
                        f"{versions[version].__name__} and {obj.__name__}. Using {obj.__name__}."
                    )
                versions[version] = obj

    return versions


def _resolve_version(
    available_versions: Dict[int, Type[CustomOp]], requested_version: Optional[int]
) -> Tuple[int, Type[CustomOp]]:
    """Resolve which version to use given available and requested versions.

    Uses "since version" semantics: highest version <= requested is selected.

    Resolution strategy:
    1. If requested is None, use highest available version
    2. Try exact match
    3. Use highest version <= requested
    4. Raise KeyError if no suitable version

    Args:
        available_versions: Dict of available versions -> CustomOp classes
        requested_version: Requested opset version, or None for highest

    Returns:
        Tuple of (resolved_version, CustomOp_class)

    Raises:
        KeyError: If no suitable version found
    """
    if not available_versions:
        raise KeyError("No versions available")

    # Strategy 1: If no specific version requested, use highest
    if requested_version is None:
        highest = max(available_versions.keys())
        return highest, available_versions[highest]

    # Strategy 2: Try exact match
    if requested_version in available_versions:
        return requested_version, available_versions[requested_version]

    # Strategy 3: Use highest version <= requested (since version semantics)
    suitable = [v for v in available_versions.keys() if v <= requested_version]
    if suitable:
        selected = max(suitable)
        return selected, available_versions[selected]

    # Strategy 4: No suitable version found
    available_list = sorted(available_versions.keys())
    raise KeyError(
        f"No suitable version found. Requested: {requested_version}, "
        f"Available: {available_list}. Lowest available version is {available_list[0]}."
    )


def add_op_to_domain(domain: str, op_class: Type[CustomOp]) -> None:
    """Register a custom op directly to a domain at runtime.

    The op_type and version are automatically derived from the class name.
    Useful for testing and experimentation. For production, define CustomOps
    in the appropriate module file.

    Args:
        domain: ONNX domain name (e.g., "qonnx.custom_op.general")
        op_class: CustomOp subclass (version inferred from name)

    Example:
        add_op_to_domain("qonnx.custom_op.general", MyTestOp)      # v1
        add_op_to_domain("qonnx.custom_op.general", MyTestOp_v2)  # v2
    """
    if not issubclass(op_class, CustomOp):
        raise ValueError(f"{op_class} must be a subclass of CustomOp")

    op_type = _get_op_type_for_class(op_class)
    op_version = _get_op_version_for_class(op_class)

    with _REGISTRY_LOCK:
        # Ensure nested dict structure exists
        if domain not in _OP_REGISTRY:
            _OP_REGISTRY[domain] = {}
        if op_type not in _OP_REGISTRY[domain]:
            _OP_REGISTRY[domain][op_type] = {}

        _OP_REGISTRY[domain][op_type][op_version] = op_class


def getCustomOp(node: NodeProto, onnx_opset_version: int | None = None) -> CustomOp:
    """Get a custom op instance for an ONNX node.

    Uses "since version" semantics: selects highest version <= requested opset.
    Lazy loads only the requested op_type using __all__ for efficiency.

    Args:
        node: ONNX node with domain and op_type attributes
        onnx_opset_version: Opset version from model's opset_import, or None for highest

    Returns:
        CustomOp instance for the node

    Raises:
        KeyError: If op_type not found in domain or no suitable version available
    """
    op_type = node.op_type
    domain = node.domain

    with _REGISTRY_LOCK:
        # O(1) nested dict lookup to check cache
        if domain in _OP_REGISTRY and op_type in _OP_REGISTRY[domain]:
            cached_versions = _OP_REGISTRY[domain][op_type]
        else:
            # Cache miss: discover THIS op only (lazy, uses __all__ for speed)
            cached_versions = _discover_custom_op_versions(domain, op_type)

            if not cached_versions:
                module_path = resolve_domain(domain)
                raise KeyError(
                    f"Op '{op_type}' not found in domain '{domain}' (module: {module_path}). "
                    f"Ensure it's defined in the module with proper naming (OpName or OpName_vN)."
                )

            # Cache it in nested structure
            if domain not in _OP_REGISTRY:
                _OP_REGISTRY[domain] = {}
            _OP_REGISTRY[domain][op_type] = cached_versions

        # Resolve which version to use
        resolved_version, op_class = _resolve_version(cached_versions, onnx_opset_version)

        # Instantiate and return
        return op_class(node, onnx_opset_version=resolved_version)


def get_supported_versions(domain: str, op_type: str) -> List[int]:
    """Get list of supported opset versions for a custom op.

    Returns all "since versions" where the operator was introduced or changed.

    Args:
        domain: ONNX domain name
        op_type: Operation type name

    Returns:
        Sorted list of opset versions

    Raises:
        KeyError: If op not found
    """
    with _REGISTRY_LOCK:
        # O(1) check if cached
        if domain in _OP_REGISTRY and op_type in _OP_REGISTRY[domain]:
            return sorted(_OP_REGISTRY[domain][op_type].keys())

        # Not cached: discover this op
        versions_dict = _discover_custom_op_versions(domain, op_type)

        if not versions_dict:
            raise KeyError(f"Op '{op_type}' not found in domain '{domain}'")

        # Cache discovered versions
        if domain not in _OP_REGISTRY:
            _OP_REGISTRY[domain] = {}
        _OP_REGISTRY[domain][op_type] = versions_dict

        return sorted(versions_dict.keys())


def is_custom_op(domain: str, op_type: Optional[str] = None) -> bool:
    """Check if a custom op exists or if a domain has any custom ops.

    Args:
        domain: The ONNX domain name
        op_type: Optional operation type name. If None, checks if domain has any ops.

    Returns:
        True if the specific op exists (when op_type given) or
        if any ops exist for the domain (when op_type=None), False otherwise
    """
    # Empty domain means standard ONNX op
    if not domain:
        return False

    with _REGISTRY_LOCK:
        if op_type is not None:
            # Check for specific op - O(1) with nested dict
            if domain in _OP_REGISTRY and op_type in _OP_REGISTRY[domain]:
                return True
            # Try to discover
            versions = _discover_custom_op_versions(domain, op_type)
            return len(versions) > 0
        else:
            # Check if domain has any registered ops
            if domain in _OP_REGISTRY and _OP_REGISTRY[domain]:
                return True
            # Try to import the domain module as fallback
            module_path = resolve_domain(domain)
            try:
                importlib.import_module(module_path)
                return True
            except (ModuleNotFoundError, ValueError):
                return False


def hasCustomOp(domain: str, op_type: str) -> bool:
    """Deprecated: Use is_custom_op instead.

    Check if a custom op exists.

    Args:
        domain: The ONNX domain name
        op_type: The operation type name

    Returns:
        True if the op exists, False otherwise
    """
    warnings.warn(
        "hasCustomOp is deprecated and will be removed in QONNX v1.0. " "Use is_custom_op instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return is_custom_op(domain, op_type)


def get_ops_in_domain(domain: str) -> List[Tuple[str, Type[CustomOp]]]:
    """Get all CustomOp classes available in a domain.

    Note: Returns unique op_types. If multiple versions exist, returns the highest version.
    This function eagerly loads all ops in the domain.

    Args:
        domain: ONNX domain name (e.g., "qonnx.custom_op.general")

    Returns:
        List of (op_type, op_class) tuples

    Example:
        ::

            ops = get_ops_in_domain("qonnx.custom_op.general")
            for op_name, op_class in ops:
                print(f"{op_name}: {op_class}")

    """
    module_path = resolve_domain(domain)
    ops_dict = {}

    with _REGISTRY_LOCK:
        # Strategy 1: Get cached ops (fast path) - use highest version
        if domain in _OP_REGISTRY:
            for op_type, versions in _OP_REGISTRY[domain].items():
                if versions:
                    highest_version = max(versions.keys())
                    ops_dict[op_type] = versions[highest_version]

        # Strategy 2: Discover from module (for uncached ops)
        # This uses full scan since we want ALL ops
        try:
            module = importlib.import_module(module_path)

            # Use __all__ if available for efficiency
            if hasattr(module, "__all__"):
                candidates = [(name, getattr(module, name, None)) for name in module.__all__]
                candidates = [(n, obj) for n, obj in candidates if obj is not None]
            else:
                candidates = inspect.getmembers(module, inspect.isclass)

            for name, obj in candidates:
                if not (inspect.isclass(obj) and issubclass(obj, CustomOp) and obj is not CustomOp):
                    continue

                op_type = _get_op_type_for_class(obj)
                try:
                    version = _get_op_version_for_class(obj)
                except ValueError:
                    continue

                # Keep highest version only
                if op_type not in ops_dict:
                    ops_dict[op_type] = obj
                else:
                    # Check if this version is higher
                    existing_version = _get_op_version_for_class(ops_dict[op_type])
                    if version > existing_version:
                        ops_dict[op_type] = obj

        except ModuleNotFoundError:
            pass  # Domain doesn't exist as module, return cached ops only

    return list(ops_dict.items())
