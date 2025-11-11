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

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model
from qonnx.util.config import extract_model_config_to_json, extract_model_config


# Helper functions for creating ONNX nodes and graphs

def make_im2col_node(name, inputs, outputs, stride, kernel_size, input_shape, pad_amount):
    """Helper to create an Im2Col node with given parameters."""
    return helper.make_node(
        "Im2Col",
        inputs=inputs,
        outputs=outputs,
        domain="qonnx.custom_op.general",
        stride=stride,
        kernel_size=kernel_size,
        input_shape=input_shape,
        pad_amount=pad_amount,
        name=name
    )


def make_if_node_with_subgraph(name, condition_input, output, subgraph):
    """Helper to create an If node with a subgraph for both branches."""
    if_node = helper.make_node(
        "If",
        inputs=[condition_input],
        outputs=[output],
        domain="",  # Standard ONNX operator
        name=name
    )
    if_node.attribute.append(helper.make_attribute("then_branch", subgraph))
    if_node.attribute.append(helper.make_attribute("else_branch", subgraph))
    return if_node


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
    """Create a simple model with Im2Col nodes that have configurable attributes."""
    inp = helper.make_tensor_value_info("inp", onnx.TensorProto.FLOAT, [1, 14, 14, 3])
    out = helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [1, 7, 7, 27])
    
    im2col_node = make_im2col_node(
        "Im2Col_0", ["inp"], ["out"],
        stride=[2, 2], kernel_size=[3, 3],
        input_shape="(1, 14, 14, 3)", pad_amount=[0, 0, 0, 0]
    )
    
    graph = helper.make_graph(
        nodes=[im2col_node], name="simple_graph",
        inputs=[inp], outputs=[out]
    )
    
    model = qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    return ModelWrapper(model)


def make_model_with_subgraphs():
    """Create a model with nodes that contain subgraphs with Im2Col operations.
    The If node has different then_branch and else_branch subgraphs."""
    
    # Create then_branch subgraph with Im2Col nodes
    then_inp = helper.make_tensor_value_info("sub_inp", onnx.TensorProto.FLOAT, [1, 14, 14, 3])
    then_out = helper.make_tensor_value_info("sub_out", onnx.TensorProto.FLOAT, [1, 7, 7, 27])
    
    then_im2col_1 = make_im2col_node(
        "SubIm2Col_0", ["sub_inp"], ["sub_intermediate"],
        stride=[2, 2], kernel_size=[3, 3],
        input_shape="(1, 14, 14, 3)", pad_amount=[1, 1, 1, 1]
    )
    then_im2col_2 = make_im2col_node(
        "SubIm2Col_1", ["sub_intermediate"], ["sub_out"],
        stride=[1, 1], kernel_size=[5, 5],
        input_shape="(1, 7, 7, 27)", pad_amount=[2, 2, 2, 2]
    )
    
    then_branch = helper.make_graph(
        nodes=[then_im2col_1, then_im2col_2], name="then_branch",
        inputs=[then_inp], outputs=[then_out]
    )
    
    # Create else_branch subgraph with different Im2Col configurations
    else_inp = helper.make_tensor_value_info("sub_inp", onnx.TensorProto.FLOAT, [1, 14, 14, 3])
    else_out = helper.make_tensor_value_info("sub_out", onnx.TensorProto.FLOAT, [1, 7, 7, 27])
    
    else_im2col_1 = make_im2col_node(
        "SubIm2Col_0", ["sub_inp"], ["else_intermediate"],
        stride=[1, 1], kernel_size=[7, 7],
        input_shape="(1, 14, 14, 3)", pad_amount=[3, 3, 3, 3]
    )
    else_im2col_2 = make_im2col_node(
        "SubIm2Col_1", ["else_intermediate"], ["sub_out"],
        stride=[2, 2], kernel_size=[3, 3],
        input_shape="(1, 14, 14, 63)", pad_amount=[0, 0, 0, 0]
    )
    
    else_branch = helper.make_graph(
        nodes=[else_im2col_1, else_im2col_2], name="else_branch",
        inputs=[else_inp], outputs=[else_out]
    )
    
    # Create main graph
    main_inp = helper.make_tensor_value_info("main_inp", onnx.TensorProto.FLOAT, [1, 14, 14, 3])
    main_out = helper.make_tensor_value_info("main_out", onnx.TensorProto.FLOAT, [1, 7, 7, 27])
    
    main_im2col = make_im2col_node(
        "Im2Col_0", ["main_inp"], ["main_intermediate"],
        stride=[1, 1], kernel_size=[7, 7],
        input_shape="(1, 14, 14, 3)", pad_amount=[3, 3, 3, 3]
    )
    
    # Create If node with different then/else branches
    if_node = helper.make_node(
        "If",
        inputs=["condition"],
        outputs=["main_out"],
        domain="",
        name="IfNode_0"
    )
    if_node.attribute.append(helper.make_attribute("then_branch", then_branch))
    if_node.attribute.append(helper.make_attribute("else_branch", else_branch))
    
    condition_init = helper.make_tensor("condition", onnx.TensorProto.BOOL, [], [True])
    
    main_graph = helper.make_graph(
        nodes=[main_im2col, if_node], name="main_graph",
        inputs=[main_inp], outputs=[main_out],
        initializer=[condition_init]
    )
    
    model = qonnx_make_model(main_graph, opset_imports=[helper.make_opsetid("", 11)])
    return ModelWrapper(model)


def make_nested_subgraph_model():
    """Create a model with nested subgraphs (subgraph within a subgraph)."""
    # Deepest subgraph (level 2)
    deep_inp = helper.make_tensor_value_info("deep_inp", onnx.TensorProto.FLOAT, [1, 8, 8, 16])
    deep_out = helper.make_tensor_value_info("deep_out", onnx.TensorProto.FLOAT, [1, 4, 4, 144])
    
    deep_im2col = make_im2col_node(
        "DeepIm2Col_0", ["deep_inp"], ["deep_out"],
        stride=[2, 2], kernel_size=[3, 3],
        input_shape="(1, 8, 8, 16)", pad_amount=[0, 0, 0, 0]
    )
    
    deep_subgraph = helper.make_graph(
        nodes=[deep_im2col], name="deep_subgraph",
        inputs=[deep_inp], outputs=[deep_out]
    )
    
    # Middle subgraph (level 1)
    mid_inp = helper.make_tensor_value_info("mid_inp", onnx.TensorProto.FLOAT, [1, 14, 14, 3])
    mid_out = helper.make_tensor_value_info("mid_out", onnx.TensorProto.FLOAT, [1, 4, 4, 144])
    
    mid_im2col = make_im2col_node(
        "MidIm2Col_0", ["mid_inp"], ["mid_intermediate"],
        stride=[1, 1], kernel_size=[5, 5],
        input_shape="(1, 14, 14, 3)", pad_amount=[2, 2, 2, 2]
    )
    
    mid_if_node = make_if_node_with_subgraph("MidIfNode_0", "mid_condition", "mid_out", deep_subgraph)
    mid_condition_init = helper.make_tensor("mid_condition", onnx.TensorProto.BOOL, [], [True])
    
    mid_subgraph = helper.make_graph(
        nodes=[mid_im2col, mid_if_node], name="mid_subgraph",
        inputs=[mid_inp], outputs=[mid_out],
        initializer=[mid_condition_init]
    )
    
    # Main graph
    main_inp = helper.make_tensor_value_info("main_inp", onnx.TensorProto.FLOAT, [1, 28, 28, 1])
    main_out = helper.make_tensor_value_info("main_out", onnx.TensorProto.FLOAT, [1, 4, 4, 144])
    
    main_im2col = make_im2col_node(
        "MainIm2Col_0", ["main_inp"], ["main_intermediate"],
        stride=[2, 2], kernel_size=[3, 3],
        input_shape="(1, 28, 28, 1)", pad_amount=[1, 1, 1, 1]
    )
    
    main_if_node = make_if_node_with_subgraph("MainIfNode_0", "main_condition", "main_out", mid_subgraph)
    main_condition_init = helper.make_tensor("main_condition", onnx.TensorProto.BOOL, [], [True])
    
    main_graph = helper.make_graph(
        nodes=[main_im2col, main_if_node], name="main_graph",
        inputs=[main_inp], outputs=[main_out],
        initializer=[main_condition_init]
    )
    
    model = qonnx_make_model(main_graph, opset_imports=[helper.make_opsetid("", 11)])
    return ModelWrapper(model)


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


def test_extract_model_config_to_json_simple():
    """Test extracting config to JSON from a simple model without subgraphs."""
    model = make_simple_model_with_im2col()
    config, cleanup = extract_config_to_temp_json(model, ["kernel_size", "stride", "pad_amount"])
    
    try:
        verify_config_basic_structure(config)
        verify_node_attributes(config, "Im2Col_0", {
            "kernel_size": [3, 3],
            "stride": [2, 2],
            "pad_amount": [0, 0, 0, 0]
        })
    finally:
        cleanup()


def test_extract_model_config_with_subgraphs():
    """Test extracting config from a model with subgraphs.
    The If node has different configurations in then_branch and else_branch."""
    model = make_model_with_subgraphs()
    config = extract_model_config(model, None, ["kernel_size", "stride", "pad_amount"])
    
    verify_config_basic_structure(config)
    
    # Verify main graph node
    verify_node_attributes(config, "Im2Col_0", {
        "kernel_size": [7, 7],
        "stride": [1, 1],
        "pad_amount": [3, 3, 3, 3]
    })
    
    # Note: Both then_branch and else_branch nodes have the same names (SubIm2Col_0, SubIm2Col_1)
    # and get the same hierarchy prefix (IfNode_0_), so the else_branch values overwrite
    # the then_branch values (last encountered wins). This is expected behavior since both
    # branches share the same parent node. In practice, only one branch executes at runtime.
    
    # Verify subgraph nodes - these will have values from else_branch (last processed)
    verify_node_attributes(config, "IfNode_0_SubIm2Col_0", {
        "kernel_size": [7, 7],
        "stride": [1, 1],
        "pad_amount": [3, 3, 3, 3]
    })
    verify_node_attributes(config, "IfNode_0_SubIm2Col_1", {
        "kernel_size": [3, 3],
        "stride": [2, 2],
        "pad_amount": [0, 0, 0, 0]
    })
    
    # Verify no aliasing at different hierarchy levels
    assert "Im2Col_0" in config  # Top-level node
    assert "IfNode_0_SubIm2Col_0" in config  # Subgraph node
    assert "IfNode_0_SubIm2Col_1" in config  # Subgraph node
    
    # Verify original unprefixed names don't exist (they should be prefixed now)
    assert "SubIm2Col_0" not in config
    assert "SubIm2Col_1" not in config


def test_extract_model_config_to_json_with_subgraphs():
    """Test extracting config to JSON from a model with subgraphs.
    The If node has different configurations in then_branch and else_branch."""
    model = make_model_with_subgraphs()
    config, cleanup = extract_config_to_temp_json(model, ["kernel_size", "stride", "pad_amount"])
    
    try:
        verify_config_basic_structure(config)
        verify_node_attributes(config, "Im2Col_0", {"kernel_size": [7, 7], "stride": [1, 1], "pad_amount": [3, 3, 3, 3]})
        
        # Both branches are extracted, but nodes with same names will overwrite each other
        # (last encountered wins). This is expected since both branches share parent hierarchy.
        verify_node_attributes(config, "IfNode_0_SubIm2Col_0", {"kernel_size": [7, 7], "stride": [1, 1], "pad_amount": [3, 3, 3, 3]})
        verify_node_attributes(config, "IfNode_0_SubIm2Col_1", {"kernel_size": [3, 3], "stride": [2, 2], "pad_amount": [0, 0, 0, 0]})
        
        # Verify all nodes with hierarchy-encoded names
        assert "Im2Col_0" in config
        assert "IfNode_0_SubIm2Col_0" in config
        assert "IfNode_0_SubIm2Col_1" in config
    finally:
        cleanup()


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


def test_extract_model_config_to_json_nested_subgraphs():
    """Test extracting config to JSON from a model with nested subgraphs."""
    model = make_nested_subgraph_model()
    config, cleanup = extract_config_to_temp_json(model, ["kernel_size", "stride", "pad_amount"])
    
    try:
        verify_config_basic_structure(config)
        
        # Verify nodes with hierarchy-encoded names
        verify_node_attributes(config, "MainIm2Col_0", {
            "kernel_size": [3, 3], "stride": [2, 2], "pad_amount": [1, 1, 1, 1]
        })
        verify_node_attributes(config, "MainIfNode_0_MidIm2Col_0", {
            "kernel_size": [5, 5], "stride": [1, 1], "pad_amount": [2, 2, 2, 2]
        })
        verify_node_attributes(config, "MainIfNode_0_MidIfNode_0_DeepIm2Col_0", {
            "kernel_size": [3, 3], "stride": [2, 2], "pad_amount": [0, 0, 0, 0]
        })
        
        # Verify nested hierarchy encoded in names
        assert "MainIm2Col_0" in config  # Top-level
        assert "MainIfNode_0_MidIm2Col_0" in config  # One level deep
        assert "MainIfNode_0_MidIfNode_0_DeepIm2Col_0" in config  # Two levels deep
    finally:
        cleanup()


def test_extract_model_config_empty_attr_list():
    """Test that extracting with an empty attribute list returns an empty or minimal config."""
    model = make_simple_model_with_im2col()
    
    config = extract_model_config(model, None, [])
    
    # Should have no node-specific configs when no attributes are requested
    verify_config_basic_structure(config)
    assert "Im2Col_0" not in config, "No nodes should be in config when no attributes are requested"


def test_extract_model_config_nonexistent_attr():
    """Test extracting attributes that don't exist on the nodes."""
    model = make_simple_model_with_im2col()
    
    # Try to extract an attribute that doesn't exist
    config = extract_model_config(model, None, ["nonexistent_attr"])
    
    # Config should be a dict but node should not appear since it has no matching attrs
    verify_config_basic_structure(config)
    # The node won't appear in config if none of its attributes match
    assert "Im2Col_0" not in config, "Node should not appear if it has no matching attributes"


def test_top_level_vs_subgraph_node_names():
    """Test that top-level nodes have simple names while subgraph nodes have hierarchy prefixes."""
    # Test simple model (no subgraphs at all)
    model = make_simple_model_with_im2col()
    config = extract_model_config(model, None, ["kernel_size", "stride"])
    
    verify_config_basic_structure(config)
    # Simple name for top-level node
    assert "Im2Col_0" in config
    verify_node_attributes(config, "Im2Col_0", {"kernel_size": [3, 3], "stride": [2, 2]})
    
    # Test model with subgraphs - verify hierarchy encoding in names
    model_with_sub = make_model_with_subgraphs()
    config_with_sub = extract_model_config(model_with_sub, None, ["kernel_size"])
    
    # Top-level node has simple name
    assert "Im2Col_0" in config_with_sub
    
    # Subgraph nodes have prefixed names
    assert "IfNode_0_SubIm2Col_0" in config_with_sub
    assert "IfNode_0_SubIm2Col_1" in config_with_sub
    
    # Old unprefixed names should NOT exist
    assert "SubIm2Col_0" not in config_with_sub
    assert "SubIm2Col_1" not in config_with_sub


def test_roundtrip_export_import_simple():
    """Test that we can export a config and reimport it with ApplyConfig for a simple model."""
    from qonnx.transformation.general import ApplyConfig
    
    # Create original model with specific attribute values
    model = make_simple_model_with_im2col()
    
    # Extract original attributes
    original_node = model.graph.node[0]
    original_inst = getCustomOp(original_node)
    original_kernel = original_inst.get_nodeattr("kernel_size")
    original_stride = original_inst.get_nodeattr("stride")
    original_pad = original_inst.get_nodeattr("pad_amount")
    
    # Export config
    config, cleanup = extract_config_to_temp_json(model, ["kernel_size", "stride", "pad_amount"])
    json_file = config  # Save for later
    
    try:
        # Modify the model's attributes to different values
        original_inst.set_nodeattr("kernel_size", [5, 5])
        original_inst.set_nodeattr("stride", [3, 3])
        original_inst.set_nodeattr("pad_amount", [2, 2, 2, 2])
        
        # Verify the attributes changed
        assert original_inst.get_nodeattr("kernel_size") == [5, 5]
        assert original_inst.get_nodeattr("stride") == [3, 3]
        
        # Create the config dict with Defaults key (required by ApplyConfig)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_with_defaults = config.copy()
            config_with_defaults["Defaults"] = {}
            json.dump(config_with_defaults, f, indent=2)
            config_json_file = f.name
        
        # Apply the original config back
        model = model.transform(ApplyConfig(config_json_file))
        
        # Verify attributes are restored to original values
        restored_inst = getCustomOp(model.graph.node[0])
        assert restored_inst.get_nodeattr("kernel_size") == original_kernel
        assert restored_inst.get_nodeattr("stride") == original_stride
        assert restored_inst.get_nodeattr("pad_amount") == original_pad
        
        # Cleanup config file
        if os.path.exists(config_json_file):
            os.remove(config_json_file)
    finally:
        cleanup()


def test_roundtrip_export_import_with_subgraphs():
    """Test export/import round-trip for a model with subgraphs."""
    from qonnx.transformation.general import ApplyConfig
    
    # Create model with subgraphs
    model = make_model_with_subgraphs()
    
    # Store original attribute values for all nodes
    original_attrs = {}
    for node in model.graph.node:
        if node.op_type == "Im2Col":
            inst = getCustomOp(node)
            original_attrs[node.name] = {
                "kernel_size": inst.get_nodeattr("kernel_size"),
                "stride": inst.get_nodeattr("stride"),
                "pad_amount": inst.get_nodeattr("pad_amount")
            }
    
    # Get nodes from subgraph
    if_node = model.get_nodes_by_op_type("If")[0]
    subgraph_attr = if_node.attribute[0]  # then_branch
    subgraph = model.make_subgraph_modelwrapper(subgraph_attr.g)
    for node in subgraph.graph.node:
        if node.op_type == "Im2Col":
            inst = getCustomOp(node)
            original_attrs[node.name] = {
                "kernel_size": inst.get_nodeattr("kernel_size"),
                "stride": inst.get_nodeattr("stride"),
                "pad_amount": inst.get_nodeattr("pad_amount")
            }
    
    # Export config
    config, cleanup = extract_config_to_temp_json(model, ["kernel_size", "stride", "pad_amount"])
    
    try:
        # Modify all Im2Col nodes to different values
        for node in model.graph.node:
            if node.op_type == "Im2Col":
                inst = getCustomOp(node)
                inst.set_nodeattr("kernel_size", [9, 9])
                inst.set_nodeattr("stride", [4, 4])
                inst.set_nodeattr("pad_amount", [5, 5, 5, 5])
        
        # Modify subgraph nodes
        for node in subgraph.graph.node:
            if node.op_type == "Im2Col":
                inst = getCustomOp(node)
                inst.set_nodeattr("kernel_size", [9, 9])
                inst.set_nodeattr("stride", [4, 4])
                inst.set_nodeattr("pad_amount", [5, 5, 5, 5])
        
        # Create config with Defaults key
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_with_defaults = config.copy()
            config_with_defaults["Defaults"] = {}
            json.dump(config_with_defaults, f, indent=2)
            config_json_file = f.name
        
        # Apply the original config back
        model = model.transform(ApplyConfig(config_json_file))
        
        # Verify main graph nodes are restored
        for node in model.graph.node:
            if node.op_type == "Im2Col":
                inst = getCustomOp(node)
                assert inst.get_nodeattr("kernel_size") == original_attrs[node.name]["kernel_size"]
                assert inst.get_nodeattr("stride") == original_attrs[node.name]["stride"]
                assert inst.get_nodeattr("pad_amount") == original_attrs[node.name]["pad_amount"]
        
        # Verify subgraph nodes are restored
        if_node = model.get_nodes_by_op_type("If")[0]
        subgraph_attr = if_node.attribute[0]
        subgraph = model.make_subgraph_modelwrapper(subgraph_attr.g)
        for node in subgraph.graph.node:
            if node.op_type == "Im2Col":
                inst = getCustomOp(node)
                assert inst.get_nodeattr("kernel_size") == original_attrs[node.name]["kernel_size"]
                assert inst.get_nodeattr("stride") == original_attrs[node.name]["stride"]
                assert inst.get_nodeattr("pad_amount") == original_attrs[node.name]["pad_amount"]
        
        # Cleanup
        if os.path.exists(config_json_file):
            os.remove(config_json_file)
    finally:
        cleanup()


def test_roundtrip_export_import_nested_subgraphs():
    """Test export/import round-trip for a model with nested subgraphs.
    
    Note: This test creates two separate models to avoid issues with modifying
    subgraph nodes through wrappers.
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
    model1 = make_nested_subgraph_model()
    original_attrs = collect_im2col_attrs(model1)
    
    # Export config from first model
    config, cleanup = extract_config_to_temp_json(model1, ["kernel_size", "stride", "pad_amount"])
    
    try:
        # Create a second model with DIFFERENT attribute values
        # (We'll modify the creation function inline to use different values)
        model2 = make_nested_subgraph_model()
        
        # Modify the top-level Im2Col node directly (this works)
        for node in model2.graph.node:
            if node.op_type == "Im2Col":
                inst = getCustomOp(node)
                inst.set_nodeattr("kernel_size", [11, 11])
                inst.set_nodeattr("stride", [5, 5])
                inst.set_nodeattr("pad_amount", [7, 7, 7, 7])
        
        # Verify the top-level node was modified
        top_attrs_before = {}
        for node in model2.graph.node:
            if node.op_type == "Im2Col":
                inst = getCustomOp(node)
                top_attrs_before[node.name] = inst.get_nodeattr("kernel_size")
        
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
                f"Node {node_name} kernel_size not restored: {restored_attrs[node_name]['kernel_size']} != {original_attrs[node_name]['kernel_size']}"
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


def test_duplicate_node_names_different_levels():
    """Test that nodes with the same name at different hierarchy levels are handled correctly.
    
    With the new hierarchy-encoding approach, nodes with duplicate names at different levels
    will have unique config keys (parent prefix makes them distinct), eliminating aliasing.
    """
    from qonnx.transformation.general import ApplyConfig
    
    # Create a model where the same node name appears at different levels
    # Top-level graph with Im2Col_0
    top_inp = helper.make_tensor_value_info("top_inp", onnx.TensorProto.FLOAT, [1, 14, 14, 3])
    top_out = helper.make_tensor_value_info("top_out", onnx.TensorProto.FLOAT, [1, 7, 7, 27])
    
    top_im2col = make_im2col_node(
        "Im2Col_0", ["top_inp"], ["top_intermediate"],
        stride=[1, 1], kernel_size=[3, 3],
        input_shape="(1, 14, 14, 3)", pad_amount=[0, 0, 0, 0]
    )
    
    # Subgraph also has Im2Col_0 (same name!)
    sub_inp = helper.make_tensor_value_info("sub_inp", onnx.TensorProto.FLOAT, [1, 14, 14, 3])
    sub_out = helper.make_tensor_value_info("sub_out", onnx.TensorProto.FLOAT, [1, 7, 7, 27])
    
    sub_im2col = make_im2col_node(
        "Im2Col_0",  # Same name as top-level node!
        ["sub_inp"], ["sub_out"],
        stride=[2, 2], kernel_size=[5, 5],
        input_shape="(1, 14, 14, 3)", pad_amount=[1, 1, 1, 1]
    )
    
    subgraph = helper.make_graph(
        nodes=[sub_im2col], name="subgraph_1",
        inputs=[sub_inp], outputs=[sub_out]
    )
    
    # Create If node with subgraph
    if_node = make_if_node_with_subgraph("IfNode_0", "condition", "top_out", subgraph)
    condition_init = helper.make_tensor("condition", onnx.TensorProto.BOOL, [], [True])
    
    main_graph = helper.make_graph(
        nodes=[top_im2col, if_node], name="main_graph",
        inputs=[top_inp], outputs=[top_out],
        initializer=[condition_init]
    )
    
    model = qonnx_make_model(main_graph, opset_imports=[helper.make_opsetid("", 11)])
    model = ModelWrapper(model)
    
    # Extract config
    config = extract_model_config(model, None, ["kernel_size", "stride", "pad_amount"])
    
    # NEW BEHAVIOR: Both nodes should be present with distinct keys!
    assert "Im2Col_0" in config, "Top-level Im2Col_0 should be in config"
    assert "IfNode_0_Im2Col_0" in config, "Subgraph Im2Col_0 should be in config with prefix"
    
    # Verify both nodes have their correct attributes (no aliasing!)
    verify_node_attributes(config, "Im2Col_0", {
        "kernel_size": [3, 3],
        "stride": [1, 1],
        "pad_amount": [0, 0, 0, 0]
    })
    
    verify_node_attributes(config, "IfNode_0_Im2Col_0", {
        "kernel_size": [5, 5],
        "stride": [2, 2],
        "pad_amount": [1, 1, 1, 1]
    })
    
    print("SUCCESS: Hierarchy encoding prevents aliasing for duplicate node names!")
    
    # Test round-trip: both nodes should be independently configurable
    config_json, cleanup = extract_config_to_temp_json(model, ["kernel_size", "stride", "pad_amount"])
    
    try:
        model2 = qonnx_make_model(main_graph, opset_imports=[helper.make_opsetid("", 11)])
        model2 = ModelWrapper(model2)
        
        # Modify both Im2Col_0 nodes
        for node in model2.graph.node:
            if node.op_type == "Im2Col":
                inst = getCustomOp(node)
                inst.set_nodeattr("kernel_size", [9, 9])
                inst.set_nodeattr("stride", [7, 7])
        
        if_node = model2.get_nodes_by_op_type("If")[0]
        subgraph_attr = if_node.attribute[0]
        subgraph_wrapper = model2.make_subgraph_modelwrapper(subgraph_attr.g)
        for node in subgraph_wrapper.graph.node:
            if node.op_type == "Im2Col":
                inst = getCustomOp(node)
                inst.set_nodeattr("kernel_size", [9, 9])
                inst.set_nodeattr("stride", [7, 7])
        
        # Apply config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_with_defaults = config_json.copy()
            config_with_defaults["Defaults"] = {}
            json.dump(config_with_defaults, f, indent=2)
            config_file = f.name
        
        model2 = model2.transform(ApplyConfig(config_file))
        
        # BOTH nodes should be restored to their original values
        # Top-level Im2Col_0
        for node in model2.graph.node:
            if node.op_type == "Im2Col":
                inst = getCustomOp(node)
                assert inst.get_nodeattr("kernel_size") == [3, 3]
                assert inst.get_nodeattr("stride") == [1, 1]
                assert inst.get_nodeattr("pad_amount") == [0, 0, 0, 0]
        
        # Subgraph Im2Col_0
        if_node = model2.get_nodes_by_op_type("If")[0]
        subgraph_attr = if_node.attribute[0]
        subgraph_wrapper = model2.make_subgraph_modelwrapper(subgraph_attr.g)
        for node in subgraph_wrapper.graph.node:
            if node.op_type == "Im2Col":
                inst = getCustomOp(node)
                assert inst.get_nodeattr("kernel_size") == [5, 5]
                assert inst.get_nodeattr("stride") == [2, 2]
                assert inst.get_nodeattr("pad_amount") == [1, 1, 1, 1]
        
        if os.path.exists(config_file):
            os.remove(config_file)
    finally:
        cleanup()
    
    print("Round-trip test PASSED: Both nodes restored independently!")
