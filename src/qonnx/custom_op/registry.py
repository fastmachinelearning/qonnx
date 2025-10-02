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
from typing import Dict

from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import get_preferred_onnx_opset

# Domain to module path mapping (only when different)
DOMAIN_MODULES: Dict[str, str] = {
    "onnx.brevitas": "qonnx.custom_op.general",  # Built-in compatibility
}


def add_domain_alias(domain: str, module_path: str) -> None:
    """Map a domain name to a different module path.
    
    Args:
        domain: The ONNX domain name
        module_path: The Python module path to use instead
        
    Example:
        add_domain_alias("finn.custom_op.fpgadataflow", "finn_custom_ops.fpgadataflow")
    """
    DOMAIN_MODULES[domain] = module_path


def add_op_to_domain(domain: str, op_type: str, op_class: type) -> None:
    """Add a custom op directly to a domain's module namespace.
    
    This function dynamically adds custom ops to module namespaces at runtime.
    Useful for test cases or dynamic op registration.
    
    Args:
        domain: The ONNX domain name (e.g., "qonnx.custom_op.general")
        op_type: The operation type name (e.g., "MyCustomOp")
        op_class: The CustomOp subclass to add
        
    Example:
        add_op_to_domain("qonnx.custom_op.general", "TestOp", TestOp)
    """
    if not inspect.isclass(op_class) or not issubclass(op_class, CustomOp):
        raise ValueError(f"{op_class} must be a subclass of CustomOp")
    
    # Get the actual module path
    module_path = DOMAIN_MODULES.get(domain, domain)
    
    try:
        # Import the module and add the op to its namespace
        module = importlib.import_module(module_path)
        setattr(module, op_type, op_class)
    except ModuleNotFoundError:
        raise ValueError(f"Could not find module for domain '{domain}' (tried: {module_path})")


def getCustomOp(node, onnx_opset_version=get_preferred_onnx_opset()):
    """Get a custom op instance for an ONNX node.
    
    Lookup order:
    1. Direct attribute lookup in module namespace
    2. Legacy custom_op dictionary (backward compatibility)
    """
    op_type = node.op_type
    domain = node.domain
    
    # Get module path (handles brevitas via DOMAIN_MODULES mapping)
    module_path = DOMAIN_MODULES.get(domain, domain)
    
    try:
        # Import the domain module
        module = importlib.import_module(module_path)
        
        # Strategy 1: Direct namespace lookup (preferred)
        if hasattr(module, op_type):
            obj = getattr(module, op_type)
            if inspect.isclass(obj) and issubclass(obj, CustomOp):
                return obj(node, onnx_opset_version=onnx_opset_version)
        
        # Strategy 2: Legacy custom_op dict (backward compatibility)
        if hasattr(module, 'custom_op') and isinstance(module.custom_op, dict):
            if op_type in module.custom_op:
                cls = module.custom_op[op_type]
                return cls(node, onnx_opset_version=onnx_opset_version)
        
        # Not found - provide clear error
        raise KeyError(
            f"Op '{op_type}' not found in domain '{domain}' (module: {module_path}). "
            f"Register it using add_op_to_domain() or ensure it's exported in the module."
        )
        
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Could not load module '{module_path}' for domain '{domain}'. "
            f"Ensure the module is installed and on your PYTHONPATH."
        ) from e


# Legacy functions for backward compatibility
def hasCustomOp(domain, op_type):
    """Check if a custom op exists in the domain's module namespace."""
    module_path = DOMAIN_MODULES.get(domain, domain)
    
    try:
        module = importlib.import_module(module_path)
        
        # Check namespace first
        if hasattr(module, op_type):
            obj = getattr(module, op_type)
            if inspect.isclass(obj) and issubclass(obj, CustomOp):
                return True
        
        # Check legacy dict
        if hasattr(module, 'custom_op') and isinstance(module.custom_op, dict):
            return op_type in module.custom_op
            
        return False
    except ModuleNotFoundError:
        return False


def get_ops_in_domain(domain):
    """Get all ops in a domain by inspecting the module namespace."""
    ops = []
    module_path = DOMAIN_MODULES.get(domain, domain)
    
    try:
        module = importlib.import_module(module_path)
        
        # Check module namespace
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, CustomOp) and 
                obj is not CustomOp and
                not name.startswith('_')):
                ops.append((name, obj))
        
        # Also check legacy dict if present
        if hasattr(module, 'custom_op') and isinstance(module.custom_op, dict):
            for name, cls in module.custom_op.items():
                if not any(op[0] == name for op in ops):
                    ops.append((name, cls))
        
        return ops
    except ModuleNotFoundError:
        return []
    except Exception as e:
        # Log the error but return empty list for backward compatibility
        import warnings
        warnings.warn(f"Error inspecting domain '{domain}': {e}")
        return []