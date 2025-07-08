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

# global registry for custom op domains
_CUSTOM_DOMAINS = set()

# global registry for custom op metadata
_OP_METADATA = {}


def register_custom_domain(domain):
    """Register a domain as containing custom ops."""
    _CUSTOM_DOMAINS.add(domain)


def is_custom_op_domain(domain):
    """Check if domain is registered for custom ops."""
    return any(domain.startswith(d) for d in _CUSTOM_DOMAINS)


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
        # Auto-register the domain when an op is registered
        register_custom_domain(domain)
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


def getCustomOp(node, onnx_opset_version=get_preferred_onnx_opset(), brevitas_exception=True):
    """Return a QONNX CustomOp instance for the given ONNX node, if it exists."""

    op_type = node.op_type
    domain = node.domain
    if brevitas_exception:
        # transparently resolve Brevitas domain ops to qonnx ones
        domain = domain.replace("onnx.brevitas", "qonnx.custom_op.general")

    key = (domain, op_type)
    cls = CUSTOM_OP_REGISTRY.get(key)
    if cls is not None:
        return cls(node, onnx_opset_version=onnx_opset_version)

    try:
        opset_module = importlib.import_module(domain)
    except ModuleNotFoundError:
        raise Exception(f"Could not load custom opset {domain}, check your PYTHONPATH")

    # op may have registered itself on import
    cls = CUSTOM_OP_REGISTRY.get(key)
    if cls is not None:
        return cls(node, onnx_opset_version=onnx_opset_version)

    # fallback to legacy custom_op dictionary
    if hasattr(opset_module, "custom_op") and isinstance(opset_module.custom_op, dict):
        try:
            inst_wrapper = opset_module.custom_op[op_type]
            return inst_wrapper(node, onnx_opset_version=onnx_opset_version)
        except KeyError:
            pass

    raise Exception(f"Op {op_type} not found in custom opset {domain}")
