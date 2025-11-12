# Copyright (c) 2025 Advanced Micro Devices, Inc.
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
# * Neither the name of QONNX nor the names of its
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

import json
import os
import pytest
import tempfile

import onnx
import onnx.helper as helper
import numpy as np
from onnxscript import script, FLOAT, BOOL
from onnxscript import opset13 as op
from onnxscript.values import Opset

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model
from qonnx.util.config import extract_model_config_to_json, extract_model_config

"""
This test module uses ONNX Script for cleaner, more Pythonic graph definitions.

ONNX Script benefits:
- Decorator-based syntax (@script()) for defining graphs as Python functions
- Type annotations (FLOAT[...], BOOL) for clear tensor shapes
- **Python if/else statements automatically convert to ONNX If nodes!**
- Nested if statements create nested subgraphs automatically
- Much cleaner than verbose helper.make_node() and helper.make_graph() calls
- Standard operators via opset13 (e.g., op.Identity, op.Add)

Key feature: Python control flow → ONNX control flow
  if condition:
      result = op.Add(x, y)
  else:
      result = op.Mul(x, y)
      
  This Python code automatically generates an ONNX If node with proper then_branch 
  and else_branch subgraphs containing the Add and Mul operations!

Limitations:
- Custom ops (like Im2Col) must still use traditional helper functions
- Operations in if/else must be inlined (not function calls) for proper subgraph generation
- Need default_opset=op when using if statements
- We use a hybrid approach: ONNX Script for graphs with standard ops, helpers for custom ops
"""

# this is a pretend opset so that we can create
# qonnx custom ops with onnxscript
qops = Opset("qonnx.custom_op.general", 1)

# Helper functions for verifying configs


def verify_config_basic_structure(config):
    """Helper to verify basic config structure."""
    assert isinstance(config, dict), "Config should be a dictionary"


def verify_node_attributes(config, node_name, expected_attrs):
    """Helper to verify node attributes in config.
    
    Args:
        config: The extracted config dictionary
        node_name: Name of the node to check (can include hierarchy prefix)
        expected_attrs: Dict of attribute_name -> expected_value
    """
    assert node_name in config
    
    # Check that all config attributes match expected_attrs
    for attr in config[node_name]:
        assert attr in expected_attrs, f"Unexpected attribute '{attr}' found in config for node '{node_name}'"
    
    for attr_name, expected_value in expected_attrs.items():
        assert config[node_name][attr_name] == expected_value


def extract_config_to_temp_json(model, attr_names):
    """Helper to extract config to a temporary JSON file and return the config dict.
    
    Automatically cleans up the temp file after reading.
    
    Returns:
        tuple: (config_dict, cleanup_function)
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_filename = f.name
    
    extract_model_config_to_json(model, json_filename, attr_names)
    
    with open(json_filename, 'r') as f:
        config = json.load(f)
    
    def cleanup():
        if os.path.exists(json_filename):
            os.remove(json_filename)
    
    return config, cleanup


def make_simple_model_with_im2col():
    """Create a simple model with Im2Col nodes that have configurable attributes.
    
    Uses ONNX Script with qops custom opset for direct Im2Col creation.
    """
    @script()
    def simple_graph(inp: FLOAT[1, 14, 14, 3]) -> FLOAT[1, 7, 7, 27]:
        # Custom Im2Col operation using qops opset
        out = qops.Im2Col(
            inp,
            stride=[2, 2],
            kernel_size=[3, 3],
            input_shape="(1, 14, 14, 3)",
            pad_amount=[0, 0, 0, 0]
        )
        return out
    
    # Convert to ONNX model
    model_proto = simple_graph.to_model_proto()
    model = ModelWrapper(model_proto)
    
    # Name the node
    model.graph.node[0].name = "Im2Col_0"
    
    return model


def make_nested_subgraph_model():
    """Create a model with nested subgraphs (subgraph within a subgraph).
    
    Uses ONNX Script with Python if statements and qops for custom operations.
    Demonstrates three levels of hierarchy:
    - Main graph with MainIm2Col_0 and MainIfNode_0
    - Mid-level subgraph with MidIm2Col_0 and MidIfNode_0  
    - Deep subgraph with DeepIm2Col_0
    """
    # Define main graph with nested if statements
    @script(default_opset=op)
    def main_graph_fn(main_inp: FLOAT[1, 28, 28, 1], main_condition: BOOL) -> FLOAT[1, 4, 4, 144]:
        """Main graph with nested if statements - creates 3 levels of hierarchy!"""
        main_intermediate = qops.Im2Col(
            main_inp,
            stride=[2, 2],
            kernel_size=[3, 3],
            input_shape="(1, 28, 28, 1)",
            pad_amount=[1, 1, 1, 1]
        )
        
        # Outer Python if statement → ONNX If node (MainIfNode_0)
        if main_condition:
            # Mid-level: MidIm2Col_0 operation
            mid_intermediate = qops.Im2Col(
                main_intermediate,
                stride=[1, 1],
                kernel_size=[5, 5],
                input_shape="(1, 14, 14, 3)",
                pad_amount=[2, 2, 2, 2]
            )
            
            # Inner Python if statement → nested ONNX If node (MidIfNode_0)
            if main_condition:  # Using main_condition as mid_condition
                # Deepest level: DeepIm2Col_0 operation
                main_out = qops.Im2Col(
                    mid_intermediate,
                    stride=[2, 2],
                    kernel_size=[3, 3],
                    input_shape="(1, 8, 8, 16)",
                    pad_amount=[0, 0, 0, 0]
                )
            else:
                main_out = qops.Im2Col(
                    mid_intermediate,
                    stride=[2, 2],
                    kernel_size=[3, 3],
                    input_shape="(1, 8, 8, 16)",
                    pad_amount=[0, 0, 0, 0]
                )
        else:
            # Mid-level: MidIm2Col_0 operation (same as then branch)
            mid_intermediate = qops.Im2Col(
                main_intermediate,
                stride=[1, 1],
                kernel_size=[5, 5],
                input_shape="(1, 14, 14, 3)",
                pad_amount=[2, 2, 2, 2]
            )
            
            # Inner Python if statement → nested ONNX If node (MidIfNode_0)
            if main_condition:  # Using main_condition as mid_condition
                # Deepest level: DeepIm2Col_0 operation
                main_out = qops.Im2Col(
                    mid_intermediate,
                    stride=[2, 2],
                    kernel_size=[3, 3],
                    input_shape="(1, 8, 8, 16)",
                    pad_amount=[0, 0, 0, 0]
                )
            else:
                main_out = qops.Im2Col(
                    mid_intermediate,
                    stride=[2, 2],
                    kernel_size=[3, 3],
                    input_shape="(1, 8, 8, 16)",
                    pad_amount=[0, 0, 0, 0]
                )
        
        return main_out
    
    # Convert ONNX Script function to model
    model_proto = main_graph_fn.to_model_proto()
    model = ModelWrapper(model_proto)
    
    # Name the nodes in main graph
    model.graph.node[0].name = "MainIm2Col_0"
    model.graph.node[1].name = "MainIfNode_0"
    
    # Add main condition initializer
    main_condition_init = helper.make_tensor("main_condition", onnx.TensorProto.BOOL, [], [True])
    model.graph.initializer.append(main_condition_init)
    
    # Name nodes in mid-level subgraph (then_branch)
    main_if_node = model.graph.node[1]
    mid_subgraph = main_if_node.attribute[0].g
    mid_subgraph.node[0].name = "MidIm2Col_0"
    mid_subgraph.node[1].name = "MidIfNode_0"
    
    # Add mid condition initializer
    mid_condition_init = helper.make_tensor("mid_condition", onnx.TensorProto.BOOL, [], [True])
    mid_subgraph.initializer.append(mid_condition_init)
    
    # Name nodes in deep subgraph (then_branch of mid If node)
    mid_if_node = mid_subgraph.node[1]
    deep_subgraph = mid_if_node.attribute[0].g
    deep_subgraph.node[0].name = "DeepIm2Col_0"
    
    return model


def test_extract_model_config_simple():
    """Test extracting config from a simple model without subgraphs."""
    model = make_simple_model_with_im2col()
    config = extract_model_config(model, None, ["input_shape", "kernel_size", "stride"])
    
    verify_config_basic_structure(config)
    verify_node_attributes(config, "Im2Col_0", {
        "input_shape": '(1, 14, 14, 3)',
        "kernel_size": [3, 3],
        "stride": [2, 2]
    })


def test_extract_model_config_with_subgraphs():
    """Test extracting config from a model with subgraphs (using nested model, testing first level)."""
    model = make_nested_subgraph_model()
    config = extract_model_config(model, None, ["kernel_size", "stride", "pad_amount"])
    
    verify_config_basic_structure(config)
    
    # Verify main graph node
    verify_node_attributes(config, "MainIm2Col_0", {
        "kernel_size": [3, 3],
        "stride": [2, 2],
        "pad_amount": [1, 1, 1, 1]
    })
    
    # Verify first-level subgraph node (mid-level)
    verify_node_attributes(config, "MainIfNode_0_MidIm2Col_0", {
        "kernel_size": [5, 5],
        "stride": [1, 1],
        "pad_amount": [2, 2, 2, 2]
    })
    
    # Verify no aliasing between hierarchy levels
    assert "MainIm2Col_0" in config  # Top-level node
    assert "MainIfNode_0_MidIm2Col_0" in config  # First-level subgraph node
    assert "MainIfNode_0_MidIfNode_0_DeepIm2Col_0" in config  # Nested subgraph node
    
    # Verify unprefixed names don't exist (they should have hierarchy prefix)
    assert "MidIm2Col_0" not in config
    assert "DeepIm2Col_0" not in config


def test_extract_model_config_nested_subgraphs():
    """Test extracting config from a model with nested subgraphs."""
    model = make_nested_subgraph_model()
    config = extract_model_config(model, None, ["kernel_size", "stride"])
    
    verify_config_basic_structure(config)
    
    # Verify nodes from all nesting levels with proper hierarchy prefixes
    verify_node_attributes(config, "MainIm2Col_0", {"kernel_size": [3, 3], "stride": [2, 2]})
    verify_node_attributes(config, "MainIfNode_0_MidIm2Col_0", {"kernel_size": [5, 5], "stride": [1, 1]})
    verify_node_attributes(config, "MainIfNode_0_MidIfNode_0_DeepIm2Col_0", {"kernel_size": [3, 3], "stride": [2, 2]})
    
    # Verify all nodes present with hierarchy-encoded names
    assert "MainIm2Col_0" in config
    assert "MainIfNode_0_MidIm2Col_0" in config
    assert "MainIfNode_0_MidIfNode_0_DeepIm2Col_0" in config


def test_extract_model_config_edge_cases():
    """Test edge cases: empty attribute list and nonexistent attributes."""
    model = make_simple_model_with_im2col()
    
    # Edge case 1: Empty attribute list - no attributes requested
    config = extract_model_config(model, None, [])
    verify_config_basic_structure(config)
    assert "Im2Col_0" not in config, "No nodes should be in config when no attributes are requested"
    
    # Edge case 2: Nonexistent attribute - attribute doesn't exist on any nodes
    config = extract_model_config(model, None, ["nonexistent_attr"])
    verify_config_basic_structure(config)
    assert "Im2Col_0" not in config, "Node should not appear if it has no matching attributes"

@pytest.mark.parametrize("model_name,model_factory", [
    ("simple", make_simple_model_with_im2col),
    ("nested", make_nested_subgraph_model),
])
def test_roundtrip_export_import(model_name, model_factory):
    """Test export/import round-trip for models with and without subgraphs.
    
    Parameterized test covering:
    - simple: Model without subgraphs
    - nested: Model with nested subgraphs (tests multi-level hierarchy)
    """
    from qonnx.transformation.general import ApplyConfig
    
    # Helper to collect all Im2Col nodes from model and subgraphs recursively
    def collect_im2col_attrs(model_wrapper, collected_attrs=None):
        if collected_attrs is None:
            collected_attrs = {}
        
        for node in model_wrapper.graph.node:
            if node.op_type == "Im2Col":
                inst = getCustomOp(node)
                collected_attrs[node.name] = {
                    "kernel_size": inst.get_nodeattr("kernel_size"),
                    "stride": inst.get_nodeattr("stride"),
                    "pad_amount": inst.get_nodeattr("pad_amount")
                }
            
            # Recursively check subgraphs
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    subgraph = model_wrapper.make_subgraph_modelwrapper(attr.g)
                    collect_im2col_attrs(subgraph, collected_attrs)
        
        return collected_attrs
    
    # Create first model and collect original attributes
    model1 = model_factory()
    original_attrs = collect_im2col_attrs(model1)
    
    # Export config from first model
    config, cleanup = extract_config_to_temp_json(model1, ["kernel_size", "stride", "pad_amount"])
    
    try:
        # Create a second model and modify its attributes
        model2 = model_factory()
        
        # Modify all Im2Col nodes to different values
        def modify_all_nodes(model_wrapper):
            for node in model_wrapper.graph.node:
                if node.op_type == "Im2Col":
                    inst = getCustomOp(node)
                    inst.set_nodeattr("kernel_size", [11, 11])
                    inst.set_nodeattr("stride", [5, 5])
                    inst.set_nodeattr("pad_amount", [7, 7, 7, 7])
                
                # Recursively modify subgraphs
                for attr in node.attribute:
                    if attr.type == onnx.AttributeProto.GRAPH:
                        subgraph = model_wrapper.make_subgraph_modelwrapper(attr.g)
                        modify_all_nodes(subgraph)
        
        modify_all_nodes(model2)
        
        # Apply the original config to model2
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_with_defaults = config.copy()
            config_with_defaults["Defaults"] = {}
            json.dump(config_with_defaults, f, indent=2)
            config_json_file = f.name
        
        model2 = model2.transform(ApplyConfig(config_json_file))
        
        # Collect attributes from model2 after applying config
        restored_attrs = collect_im2col_attrs(model2)
        
        # Verify all nodes in model2 now match original_attrs from model1
        assert len(restored_attrs) == len(original_attrs), \
            f"Expected {len(original_attrs)} nodes, got {len(restored_attrs)}"
        
        for node_name in original_attrs:
            assert node_name in restored_attrs, f"Node {node_name} not found after applying config"
            assert restored_attrs[node_name]["kernel_size"] == original_attrs[node_name]["kernel_size"], \
                f"Node {node_name} kernel_size not restored"
            assert restored_attrs[node_name]["stride"] == original_attrs[node_name]["stride"], \
                f"Node {node_name} stride not restored"
            assert restored_attrs[node_name]["pad_amount"] == original_attrs[node_name]["pad_amount"], \
                f"Node {node_name} pad_amount not restored"
        
        # Cleanup
        if os.path.exists(config_json_file):
            os.remove(config_json_file)
    finally:
        cleanup()


def test_roundtrip_partial_config():
    """Test that ApplyConfig only modifies specified attributes, leaving others unchanged."""
    from qonnx.transformation.general import ApplyConfig
    
    # Create model
    model = make_simple_model_with_im2col()
    node = model.graph.node[0]
    inst = getCustomOp(node)
    
    # Store original values
    original_kernel = inst.get_nodeattr("kernel_size")
    original_stride = inst.get_nodeattr("stride")
    original_pad = inst.get_nodeattr("pad_amount")
    
    # Export only kernel_size and stride (not pad_amount)
    config, cleanup = extract_config_to_temp_json(model, ["kernel_size", "stride"])
    
    try:
        # Modify all attributes
        inst.set_nodeattr("kernel_size", [7, 7])
        inst.set_nodeattr("stride", [4, 4])
        inst.set_nodeattr("pad_amount", [9, 9, 9, 9])
        
        # Create config with Defaults
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_with_defaults = config.copy()
            config_with_defaults["Defaults"] = {}
            json.dump(config_with_defaults, f, indent=2)
            config_json_file = f.name
        
        # Apply config
        model = model.transform(ApplyConfig(config_json_file))
        
        # Verify kernel_size and stride are restored
        inst = getCustomOp(model.graph.node[0])
        assert inst.get_nodeattr("kernel_size") == original_kernel
        assert inst.get_nodeattr("stride") == original_stride
        
        # Verify pad_amount remains modified (not in config)
        assert inst.get_nodeattr("pad_amount") == [9, 9, 9, 9]
        
        # Cleanup
        if os.path.exists(config_json_file):
            os.remove(config_json_file)
    finally:
        cleanup()
