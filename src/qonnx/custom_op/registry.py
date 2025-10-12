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
from threading import RLock
from typing import Dict, Tuple, Type

from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import get_preferred_onnx_opset

# Registry keyed by original ONNX domain: (domain, op_type) -> CustomOp class
_OP_REGISTRY: Dict[Tuple[str, str], Type[CustomOp]] = {}

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


def _discover_custom_op(domain: str, op_type: str) -> bool:
    """Discover and register a single custom op.

    Args:
        domain: The ONNX domain name
        op_type: The specific op type to discover

    Returns:
        True if op was found and registered, False otherwise
    """
    module_path = resolve_domain(domain)

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        return False

    # Try namespace lookup
    op_class = getattr(module, op_type, None)
    if inspect.isclass(op_class) and issubclass(op_class, CustomOp):
        _OP_REGISTRY[(domain, op_type)] = op_class
        return True

    # Try legacy dict
    custom_op_dict = getattr(module, 'custom_op', None)
    if isinstance(custom_op_dict, dict):
        op_class = custom_op_dict.get(op_type)
        if inspect.isclass(op_class) and issubclass(op_class, CustomOp):
            _OP_REGISTRY[(domain, op_type)] = op_class
            return True

    return False


def getCustomOp(node, onnx_opset_version=get_preferred_onnx_opset()):
    """Get a custom op instance for an ONNX node.

    Args:
        node: ONNX node with domain and op_type attributes
        onnx_opset_version: ONNX opset version to use

    Returns:
        CustomOp instance for the node

    Raises:
        KeyError: If op_type not found in domain
    """
    op_type = node.op_type
    domain = node.domain
    key = (domain, op_type)

    with _REGISTRY_LOCK:
        if key in _OP_REGISTRY:
            return _OP_REGISTRY[key](node, onnx_opset_version=onnx_opset_version)

        if _discover_custom_op(domain, op_type):
            return _OP_REGISTRY[key](node, onnx_opset_version=onnx_opset_version)

        module_path = resolve_domain(domain)
        raise KeyError(
            f"Op '{op_type}' not found in domain '{domain}' (module: {module_path}). "
            f"Ensure it's exported in the module namespace or in the custom_op dict."
        )


def hasCustomOp(domain: str, op_type: str) -> bool:
    """Check if a custom op exists.

    Args:
        domain: The ONNX domain name
        op_type: The operation type name

    Returns:
        True if the op exists, False otherwise
    """
    key = (domain, op_type)

    with _REGISTRY_LOCK:
        if key in _OP_REGISTRY:
            return True
        return _discover_custom_op(domain, op_type)