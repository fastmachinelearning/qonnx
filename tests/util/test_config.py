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
    """Create a model with nested subgraphs (asymmetric hierarchy).
    
    Uses ONNX Script with Python if statements and qops for custom operations.
    Demonstrates asymmetric hierarchy:
    - Main graph with MainIm2Col_0 and MainIfNode_0
    - Then branch: SubIm2Col_0 (2 levels total)
    - Else branch: NestedIfNode_0 containing DeepIm2Col_0 (3 levels total)
    """
    @script(default_opset=op)
    def main_graph_fn(main_inp: FLOAT[1, 28, 28, 1], condition: BOOL, nested_condition: BOOL) -> FLOAT[1, 4, 4, 144]:
        """Main graph with nested if statement in else branch."""
        main_intermediate = qops.Im2Col(
            main_inp,
            stride=[2, 2],
            kernel_size=[3, 3],
            input_shape="(1, 28, 28, 1)",
            pad_amount=[1, 1, 1, 1]
        )
        
        # Python if statement → ONNX If node with subgraph
        if condition:
            # Then branch: simple subgraph (2 levels)
            main_out = qops.Im2Col(
                main_intermediate,
                stride=[1, 1],
                kernel_size=[5, 5],
                input_shape="(1, 14, 14, 3)",
                pad_amount=[2, 2, 2, 2]
            )
        else:
            # Else branch: nested if statement (3 levels)
            if nested_condition:
                main_out = qops.Im2Col(
                    main_intermediate,
                    stride=[3, 3],
                    kernel_size=[7, 7],
                    input_shape="(1, 14, 14, 3)",
                    pad_amount=[3, 3, 3, 3]
                )
            else:
                main_out = qops.Im2Col(
                    main_intermediate,
                    stride=[3, 3],
                    kernel_size=[7, 7],
                    input_shape="(1, 14, 14, 3)",
                    pad_amount=[3, 3, 3, 3]
                )
        
        return main_out
    
    # Convert ONNX Script function to model
    model_proto = main_graph_fn.to_model_proto()
    model = ModelWrapper(model_proto)
    
    # Name the nodes in main graph
    model.graph.node[0].name = "MainIm2Col_0"
    model.graph.node[1].name = "MainIfNode_0"
    
    # Add condition initializers
    condition_init = helper.make_tensor("condition", onnx.TensorProto.BOOL, [], [True])
    model.graph.initializer.append(condition_init)
    nested_condition_init = helper.make_tensor("nested_condition", onnx.TensorProto.BOOL, [], [True])
    model.graph.initializer.append(nested_condition_init)
    
    # Name node in then_branch
    main_if_node = model.graph.node[1]
    then_branch = main_if_node.attribute[0].g
    then_branch.node[0].name = "SubIm2Col_0"
    
    # Name nodes in else_branch (has nested If node)
    else_branch = main_if_node.attribute[1].g
    else_branch.node[0].name = "NestedIfNode_0"
    
    # Name node in nested subgraph
    nested_if_node = else_branch.node[0]
    deep_subgraph = nested_if_node.attribute[0].g
    deep_subgraph.node[0].name = "DeepIm2Col_0"
    
    return model


@pytest.mark.parametrize("model_name,model_factory,expected_nodes", [
    ("simple", make_simple_model_with_im2col, {
        "Im2Col_0": {"kernel_size": [3, 3], "stride": [2, 2], "input_shape": '(1, 14, 14, 3)'}
    }),
    ("nested", make_nested_subgraph_model, {
        "MainIm2Col_0": {"kernel_size": [3, 3], "stride": [2, 2], "pad_amount": [1, 1, 1, 1]},
        "MainIfNode_0_SubIm2Col_0": {"kernel_size": [5, 5], "stride": [1, 1], "pad_amount": [2, 2, 2, 2]},
        "MainIfNode_0_NestedIfNode_0_DeepIm2Col_0": {"kernel_size": [7, 7], "stride": [3, 3], "pad_amount": [3, 3, 3, 3]}
    }),
])
def test_extract_model_config(model_name, model_factory, expected_nodes):
    """Test extracting config from models with and without subgraphs.
    
    Parameterized test covering:
    - simple: Model without subgraphs (base case)
    - nested: Model with nested subgraphs (tests hierarchy encoding at all levels)
    """
    model = model_factory()
    
    # Get all attributes that appear in expected_nodes
    all_attrs = set()
    for node_attrs in expected_nodes.values():
        all_attrs.update(node_attrs.keys())
    
    config = extract_model_config(model, None, list(all_attrs))
    verify_config_basic_structure(config)
    
    # Verify all expected nodes and their attributes
    for node_name, expected_attrs in expected_nodes.items():
        verify_node_attributes(config, node_name, expected_attrs)
    
    # For nested model, verify no aliasing (unprefixed names don't exist)
    if model_name == "nested":
        assert "SubIm2Col_0" not in config, "Subgraph node should have hierarchy prefix"
        assert "DeepIm2Col_0" not in config, "Deeply nested node should have hierarchy prefix"


@pytest.mark.parametrize("model_name,model_factory", [
    ("simple", make_simple_model_with_im2col),
    ("nested", make_nested_subgraph_model),
])
def test_extract_model_config_edge_cases(model_name, model_factory):
    """Test edge cases: empty attribute list and nonexistent attributes.
    
    Parameterized to ensure edge cases work for both simple and nested models.
    """
    model = model_factory()
    
    # Edge case 1: Empty attribute list - no attributes requested
    config = extract_model_config(model, None, [])
    verify_config_basic_structure(config)
    assert len(config) == 0, "Config should be empty when no attributes are requested"
    
    # Edge case 2: Nonexistent attribute - attribute doesn't exist on any nodes
    config = extract_model_config(model, None, ["nonexistent_attr"])
    verify_config_basic_structure(config)
    assert len(config) == 0, "Config should be empty when no nodes have matching attributes"

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


@pytest.mark.parametrize("model_name,model_factory", [
    ("simple", make_simple_model_with_im2col),
])
def test_roundtrip_partial_config(model_name, model_factory):
    """Test that ApplyConfig only modifies specified attributes, leaving others unchanged.
    
    Note: Only testing with simple model as nested model config application through subgraphs
    has complexities that make partial config verification difficult.
    """
    from qonnx.transformation.general import ApplyConfig
    
    # Helper to collect and modify all Im2Col nodes recursively
    def collect_and_store_attrs(model_wrapper, original_attrs=None):
        if original_attrs is None:
            original_attrs = {}
        for node in model_wrapper.graph.node:
            if node.op_type == "Im2Col":
                inst = getCustomOp(node)
                original_attrs[node.name] = {
                    "kernel_size": inst.get_nodeattr("kernel_size"),
                    "stride": inst.get_nodeattr("stride"),
                    "pad_amount": inst.get_nodeattr("pad_amount")
                }
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    subgraph = model_wrapper.make_subgraph_modelwrapper(attr.g)
                    collect_and_store_attrs(subgraph, original_attrs)
        return original_attrs
    
    def modify_all_attrs(graph_proto):
        """Modify attributes directly in the graph proto (not through wrapper)."""
        for node in graph_proto.node:
            if node.op_type == "Im2Col":
                inst = getCustomOp(node)
                inst.set_nodeattr("kernel_size", [7, 7])
                inst.set_nodeattr("stride", [4, 4])
                inst.set_nodeattr("pad_amount", [9, 9, 9, 9])
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    modify_all_attrs(attr.g)
    
    def verify_attrs(model_wrapper, original_attrs):
        for node in model_wrapper.graph.node:
            if node.op_type == "Im2Col":
                inst = getCustomOp(node)
                # kernel_size and stride should be restored
                assert inst.get_nodeattr("kernel_size") == original_attrs[node.name]["kernel_size"]
                assert inst.get_nodeattr("stride") == original_attrs[node.name]["stride"]
                # pad_amount should remain modified (not in config)
                assert inst.get_nodeattr("pad_amount") == [9, 9, 9, 9]
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    subgraph = model_wrapper.make_subgraph_modelwrapper(attr.g)
                    verify_attrs(subgraph, original_attrs)
    
    # Create model and store original values
    model = model_factory()
    original_attrs = collect_and_store_attrs(model)
    
    # Export only kernel_size and stride (not pad_amount)
    config, cleanup = extract_config_to_temp_json(model, ["kernel_size", "stride"])
    
    try:
        # Modify all attributes (work directly with graph proto)
        modify_all_attrs(model.graph)
        
        # Create config with Defaults
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_with_defaults = config.copy()
            config_with_defaults["Defaults"] = {}
            json.dump(config_with_defaults, f, indent=2)
            config_json_file = f.name
        
        # Apply config
        model = model.transform(ApplyConfig(config_json_file))
        
        # Verify partial restoration
        verify_attrs(model, original_attrs)
        
        # Cleanup
        if os.path.exists(config_json_file):
            os.remove(config_json_file)
    finally:
        cleanup()
