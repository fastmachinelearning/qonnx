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
import warnings
from importlib import metadata

from qonnx.util.basic import get_preferred_onnx_opset

# global registry mapping (domain, op_type) -> CustomOp subclass
CUSTOM_OP_REGISTRY = {}

# global registry for custom op metadata
_OP_METADATA = {}

# global registry mapping domains to their module paths
# Structure: DOMAIN_REGISTRY[domain] = module_path (or None if module_path == domain)
DOMAIN_REGISTRY = {}

# Track which domains have been loaded
_LOADED_DOMAINS = set()

# Track domain dependencies discovered through inheritance
_DOMAIN_DEPENDENCIES = {}  # domain -> set of dependency domains




def _ensure_domain_loaded(domain):
    """Ensure a domain and its dependencies are loaded."""
    if domain in _LOADED_DOMAINS:
        return
    
    # Mark as loaded first to prevent infinite recursion
    _LOADED_DOMAINS.add(domain)
    
    # First load any known dependencies
    if domain in _DOMAIN_DEPENDENCIES:
        for dep_domain in _DOMAIN_DEPENDENCIES[domain]:
            if dep_domain != domain:  # Avoid self-dependencies
                _ensure_domain_loaded(dep_domain)
    
    # Try to import the domain module
    if domain in DOMAIN_REGISTRY:
        module_path = DOMAIN_REGISTRY[domain] or domain
        try:
            importlib.import_module(module_path)
        except ImportError as e:
            # Remove from loaded if import failed
            _LOADED_DOMAINS.discard(domain)
            # Continue without raising - domain might still work
    elif domain.startswith(("finn.", "qonnx.")):
        # Try importing even if not in registry
        try:
            importlib.import_module(domain)
        except ImportError:
            # Remove from loaded if import failed
            _LOADED_DOMAINS.discard(domain)


def _register_op_with_dependencies(domain, op_type, cls, metadata=None):
    """Register an op and track its inheritance dependencies."""
    # Register the op
    CUSTOM_OP_REGISTRY[(domain, op_type)] = cls
    if metadata is not None:
        _OP_METADATA[(domain, op_type)] = metadata
    
    # Detect dependencies from inheritance
    for base in cls.__bases__:
        # Skip abstract base classes and non-custom ops
        if base.__name__ in ('CustomOp', 'ABC', 'object', 'HWCustomOp', 'HLSBackend', 'RTLBackend'):
            continue
            
        # Check if base class is a registered custom op
        for (reg_domain, reg_op), reg_cls in CUSTOM_OP_REGISTRY.items():
            if reg_cls == base:
                # Found a dependency - track it
                if domain not in _DOMAIN_DEPENDENCIES:
                    _DOMAIN_DEPENDENCIES[domain] = set()
                _DOMAIN_DEPENDENCIES[domain].add(reg_domain)
                
                # Immediately ensure the dependency is loaded
                _ensure_domain_loaded(reg_domain)
                break
    
    return cls


def is_custom_op_domain(domain):
    """Check if domain is registered for custom ops."""
    # Check if domain is directly registered or starts with a registered domain
    return domain in DOMAIN_REGISTRY or any(domain.startswith(d) for d in DOMAIN_REGISTRY)


def hasCustomOp(domain, op_type):
    """Check if a custom op exists without creating an instance.
    
    Args:
        domain: The domain of the custom op
        op_type: The op_type of the custom op
        
    Returns:
        bool: True if the op is registered, False otherwise
    """
    # Ensure domain is loaded first
    _ensure_domain_loaded(domain)
    return (domain, op_type) in CUSTOM_OP_REGISTRY


def get_ops_in_domain(domain):
    """Get all registered ops in a domain.
    
    Args:
        domain: The domain to query
        
    Returns:
        List[Tuple[str, Type[CustomOp]]]: List of (op_type, class) tuples
    """
    return [(op_type, cls) for (d, op_type), cls in CUSTOM_OP_REGISTRY.items() 
            if d == domain]


def get_op_metadata(domain, op_type):
    """Get metadata for a registered custom op.
    
    Args:
        domain: The domain of the custom op
        op_type: The op_type of the custom op
        
    Returns:
        dict: The metadata dict if available, None otherwise
    """
    return _OP_METADATA.get((domain, op_type))


def register_domain(domain, module_path=None):
    """Register a domain with its associated module path.
    
    This function registers the domain and its module path, allowing classes
    defined in any direct child module of this path to use @register_custom_op.
    Subfolders/subpackages must be registered separately.
    
    Args:
        domain: The domain to register (e.g., "finn.custom_op.fpgadataflow")
        module_path: The Python module path. If None, uses the domain as the path.
    """
    DOMAIN_REGISTRY[domain] = module_path


# Keep register_domain_path as deprecated alias for backward compatibility
def register_domain_path(module_path, domain):
    """Deprecated: Use register_domain instead."""
    return register_domain(domain, module_path)



def register_custom_op(domain=None, op_type=None, *, metadata=None):
    """Register a custom op with flexible domain and op_type specification.
    
    Can be used in three ways:
    1. @register_custom_op("domain", "OpType") - Explicit domain and op_type
    2. @register_custom_op("domain") - Explicit domain, class name as op_type
    3. @register_custom_op - Automatic domain inference, class name as op_type
    
    Args:
        domain: The domain for the custom op (optional)
        op_type: The op_type for the custom op (optional)
        metadata: Optional dict of metadata about the op (backend, version, etc.)
        
    Returns:
        Decorated class or decorator function
    """
    # Determine which mode we're in based on arguments
    if domain is not None and isinstance(domain, str):
        # Mode 1 or 2: Explicit domain provided
        if op_type is not None and isinstance(op_type, str):
            # Mode 1: Both domain and op_type provided
            def decorator(cls):
                return _register_op_with_dependencies(domain, op_type, cls, metadata)
            return decorator
        else:
            # Mode 2: Only domain provided, use class name as op_type
            def decorator(cls):
                final_op_type = cls.__name__
                return _register_op_with_dependencies(domain, final_op_type, cls, metadata)
            return decorator
    else:
        # Mode 3: No domain provided, or called without arguments
        # Handle the case where it's used as @register_custom_op (no parentheses)
        if domain is not None and not isinstance(domain, str):
            # This means domain is actually the class (decorator without parentheses)
            cls = domain
            module = cls.__module__
            
            # Find domain from registered domains
            inferred_domain = None
            for registered_domain, module_path in DOMAIN_REGISTRY.items():
                if module_path is None:
                    module_path = registered_domain
                if module.startswith(module_path + "."):
                    remainder = module[len(module_path) + 1:]
                    if "." not in remainder:
                        inferred_domain = registered_domain
                        break
                elif module == module_path:
                    inferred_domain = registered_domain
                    break
            
            if inferred_domain is None:
                raise ValueError(
                    f"Module '{module}' is not in a registered domain path. "
                    f"Either:\n"
                    f"1. Use @register_custom_op('domain', 'OpType')\n"
                    f"2. Use @register_custom_op('domain') to use class name as op_type\n"
                    f"3. Register the domain: register_domain('your.domain', '{'.'.join(module.split('.')[:-1])}')"
                )
            
            final_op_type = cls.__name__
            return _register_op_with_dependencies(inferred_domain, final_op_type, cls, metadata)
        else:
            # Decorator called with parentheses but no domain
            def decorator(cls):
                module = cls.__module__
                
                # Find domain from registered domains
                inferred_domain = None
                for registered_domain, module_path in DOMAIN_REGISTRY.items():
                    if module_path is None:
                        module_path = registered_domain
                    if module.startswith(module_path + "."):
                        remainder = module[len(module_path) + 1:]
                        if "." not in remainder:
                            inferred_domain = registered_domain
                            break
                    elif module == module_path:
                        inferred_domain = registered_domain
                        break
                
                if inferred_domain is None:
                    raise ValueError(
                        f"Module '{module}' is not in a registered domain path. "
                        f"Either:\n"
                        f"1. Use @register_custom_op('domain', 'OpType')\n"
                        f"2. Use @register_custom_op('domain') to use class name as op_type\n"
                        f"3. Register the domain: register_domain('your.domain', '{'.'.join(module.split('.')[:-1])}')"
                    )
                
                # Use provided op_type or default to class name
                final_op_type = op_type or cls.__name__
                return _register_op_with_dependencies(inferred_domain, final_op_type, cls, metadata)
            return decorator


def getCustomOp(node, onnx_opset_version=get_preferred_onnx_opset(), brevitas_exception=True):
    """Return a QONNX CustomOp instance for the given ONNX node."""
    op_type = node.op_type
    domain = node.domain
    
    if brevitas_exception:
        # transparently resolve Brevitas domain ops to qonnx ones
        domain = domain.replace("onnx.brevitas", "qonnx.custom_op.general")
    
    # Ensure the domain is loaded (will load dependencies automatically)
    _ensure_domain_loaded(domain)
    
    key = (domain, op_type)
    cls = CUSTOM_OP_REGISTRY.get(key)
    if cls is not None:
        return cls(node, onnx_opset_version=onnx_opset_version)
    
    # If not found and domain starts with finn, try explicit import as fallback
    # This handles cases where domain isn't registered but module exists
    if domain.startswith("finn.custom_op") and domain not in _LOADED_DOMAINS:
        try:
            importlib.import_module(domain)
            _LOADED_DOMAINS.add(domain)
            # Check again after import
            cls = CUSTOM_OP_REGISTRY.get(key)
            if cls is not None:
                return cls(node, onnx_opset_version=onnx_opset_version)
        except ImportError:
            pass
    
    available_domains = sorted(DOMAIN_REGISTRY.keys())
    raise Exception(
        f"Op '{op_type}' not found in domain '{domain}'. "
        f"Available domains: {available_domains}"
    )
