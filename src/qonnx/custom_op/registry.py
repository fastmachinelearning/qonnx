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


def register_op(domain, op_type, metadata=None):
    """Decorator for registering CustomOp classes.
    
    Args:
        domain: The domain for the custom op
        op_type: The op_type for the custom op
        metadata: Optional dict of metadata about the op (backend, version, etc.)
    """

    def decorator(cls):
        CUSTOM_OP_REGISTRY[(domain, op_type)] = cls
        if metadata is not None:
            _OP_METADATA[(domain, op_type)] = metadata
        return cls

    return decorator


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


def register_custom_op(cls=None, *, op_type=None):
    """Register a custom op, inferring domain from parent module path.
    
    Can be used as @register_custom_op or @register_custom_op(op_type="CustomName").
    Domain is inferred from registered module paths. Op type defaults to class name.
    
    Args:
        cls: The class to register (when used without parentheses)
        op_type: Optional custom op_type (defaults to class name)
        
    Returns:
        Decorated class or decorator function
    """
    def decorator(cls):
        # Get module path
        module = cls.__module__
        
        # Check if module is a direct child of any registered domain's module path
        domain = None
        for registered_domain, module_path in DOMAIN_REGISTRY.items():
            # Use domain as module path if not specified
            if module_path is None:
                module_path = registered_domain
            # Check if module is direct child of registered path
            if module.startswith(module_path + "."):
                # Ensure it's a direct child, not nested deeper
                remainder = module[len(module_path) + 1:]
                if "." not in remainder:  # No more dots = direct child
                    domain = registered_domain
                    break
            # Also check exact match (for __init__.py files)
            elif module == module_path:
                domain = registered_domain
                break
        
        if domain is None:
            raise ValueError(
                f"Module '{module}' is not in a registered domain path. "
                f"Either:\n"
                f"1. Use @register_op(domain='...', op_type='{cls.__name__}')\n"
                f"2. Register the domain: register_domain('your.domain', '{'.'.join(module.split('.')[:-1])}')"
            )
        
        # Use class name as op_type if not specified
        final_op_type = op_type or cls.__name__
        
        # Register using the standard mechanism
        return register_op(domain=domain, op_type=final_op_type)(cls)
    
    # Handle both @register_custom_op and @register_custom_op()
    if cls is None:
        return decorator
    return decorator(cls)


def getCustomOp(node, onnx_opset_version=get_preferred_onnx_opset(), brevitas_exception=True):
    """Return a QONNX CustomOp instance for the given ONNX node."""
    op_type = node.op_type
    domain = node.domain
    
    if brevitas_exception:
        # transparently resolve Brevitas domain ops to qonnx ones
        domain = domain.replace("onnx.brevitas", "qonnx.custom_op.general")
    
    key = (domain, op_type)
    cls = CUSTOM_OP_REGISTRY.get(key)
    if cls is not None:
        return cls(node, onnx_opset_version=onnx_opset_version)
    
    # Check if we need to import the module to trigger registration
    if domain.startswith("finn.custom_op"):
        try:
            importlib.import_module(domain)
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
