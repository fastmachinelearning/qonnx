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
    assert "Defaults" in config


def verify_node_attributes(config, node_name, expected_attrs):
    """Helper to verify node attributes in config.
    
    Args:
        config: The extracted config dictionary
        node_name: Name of the node to check
        expected_attrs: Dict of attribute_name -> expected_value
    """
    assert node_name in config
    for attr_name, expected_value in expected_attrs.items():
        assert config[node_name][attr_name] == expected_value


def verify_subgraph_hierarchy(config, expected_hier_path=None):
    """Helper to verify that subgraph hierarchy tracking is present and optionally matches expected path.
    
    Args:
        config: The extracted config dictionary
        expected_hier_path: Optional string or list of strings representing expected hierarchy path(s).
                          If None, just checks that 'subgraph_hier' key exists.
                          If string, checks that subgraph_hier equals that string.
                          If list, checks that subgraph_hier contains at least one of the paths.
    """
    assert "subgraph_hier" in config, "subgraph_hier key not found in config"
    
    if expected_hier_path is not None:
        actual_hier = config["subgraph_hier"]
        
        if isinstance(expected_hier_path, str):
            # Single expected path - check exact match or that actual contains it
            assert expected_hier_path in actual_hier, \
                f"Expected hierarchy path '{expected_hier_path}' not found in '{actual_hier}'"
        elif isinstance(expected_hier_path, list):
            # Multiple possible paths - check that at least one matches
            found = any(path in actual_hier for path in expected_hier_path)
            assert found, \
                f"None of the expected hierarchy paths {expected_hier_path} found in '{actual_hier}'"
    else:
        # subgraph_hier key should not be present
        assert "subgraph_hier" not in config, "subgraph_hier found in config when not expected"


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
    """Create a model with nodes that contain subgraphs with Im2Col operations."""
    # Create subgraph with Im2Col nodes
    subgraph_inp = helper.make_tensor_value_info("sub_inp", onnx.TensorProto.FLOAT, [1, 14, 14, 3])
    subgraph_out = helper.make_tensor_value_info("sub_out", onnx.TensorProto.FLOAT, [1, 7, 7, 27])
    
    sub_im2col_1 = make_im2col_node(
        "SubIm2Col_0", ["sub_inp"], ["sub_intermediate"],
        stride=[2, 2], kernel_size=[3, 3],
        input_shape="(1, 14, 14, 3)", pad_amount=[1, 1, 1, 1]
    )
    sub_im2col_2 = make_im2col_node(
        "SubIm2Col_1", ["sub_intermediate"], ["sub_out"],
        stride=[1, 1], kernel_size=[5, 5],
        input_shape="(1, 7, 7, 27)", pad_amount=[2, 2, 2, 2]
    )
    
    subgraph = helper.make_graph(
        nodes=[sub_im2col_1, sub_im2col_2], name="subgraph_1",
        inputs=[subgraph_inp], outputs=[subgraph_out]
    )
    
    # Create main graph
    main_inp = helper.make_tensor_value_info("main_inp", onnx.TensorProto.FLOAT, [1, 14, 14, 3])
    main_out = helper.make_tensor_value_info("main_out", onnx.TensorProto.FLOAT, [1, 7, 7, 27])
    
    main_im2col = make_im2col_node(
        "Im2Col_0", ["main_inp"], ["main_intermediate"],
        stride=[1, 1], kernel_size=[7, 7],
        input_shape="(1, 14, 14, 3)", pad_amount=[3, 3, 3, 3]
    )
    
    if_node = make_if_node_with_subgraph("IfNode_0", "condition", "main_out", subgraph)
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
    config = extract_model_config(model, None, ["kernel_size", "stride"])
    
    verify_config_basic_structure(config)
    verify_node_attributes(config, "Im2Col_0", {
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
    """Test extracting config from a model with subgraphs."""
    model = make_model_with_subgraphs()
    config = extract_model_config(model, None, ["kernel_size", "stride", "pad_amount"])
    
    verify_config_basic_structure(config)
    
    # Verify main graph and subgraph nodes
    verify_node_attributes(config, "Im2Col_0", {
        "kernel_size": [7, 7],
        "stride": [1, 1],
        "pad_amount": [3, 3, 3, 3]
    })
    verify_node_attributes(config, "SubIm2Col_0", {
        "kernel_size": [3, 3],
        "stride": [2, 2],
        "pad_amount": [1, 1, 1, 1]
    })
    verify_node_attributes(config, "SubIm2Col_1", {
        "kernel_size": [5, 5],
        "stride": [1, 1],
        "pad_amount": [2, 2, 2, 2]
    })
    
    # Verify subgraph hierarchy tracking
    verify_subgraph_hierarchy(config, "IfNode_0")


def test_extract_model_config_to_json_with_subgraphs():
    """Test extracting config to JSON from a model with subgraphs."""
    model = make_model_with_subgraphs()
    config, cleanup = extract_config_to_temp_json(model, ["kernel_size", "stride", "pad_amount"])
    
    try:
        verify_config_basic_structure(config)
        verify_node_attributes(config, "Im2Col_0", {"kernel_size": [7, 7], "stride": [1, 1]})
        verify_node_attributes(config, "SubIm2Col_0", {"kernel_size": [3, 3], "pad_amount": [1, 1, 1, 1]})
        verify_node_attributes(config, "SubIm2Col_1", {"kernel_size": [5, 5], "pad_amount": [2, 2, 2, 2]})
        verify_subgraph_hierarchy(config, "IfNode_0")
    finally:
        cleanup()


def test_extract_model_config_nested_subgraphs():
    """Test extracting config from a model with nested subgraphs."""
    model = make_nested_subgraph_model()
    config = extract_model_config(model, None, ["kernel_size", "stride"])
    
    verify_config_basic_structure(config)
    
    # Verify nodes from all nesting levels
    verify_node_attributes(config, "MainIm2Col_0", {"kernel_size": [3, 3], "stride": [2, 2]})
    verify_node_attributes(config, "MidIm2Col_0", {"kernel_size": [5, 5], "stride": [1, 1]})
    verify_node_attributes(config, "DeepIm2Col_0", {"kernel_size": [3, 3], "stride": [2, 2]})


def test_extract_model_config_to_json_nested_subgraphs():
    """Test extracting config to JSON from a model with nested subgraphs."""
    model = make_nested_subgraph_model()
    config, cleanup = extract_config_to_temp_json(model, ["kernel_size", "stride", "pad_amount"])
    
    try:
        verify_config_basic_structure(config)
        verify_node_attributes(config, "MainIm2Col_0", {"kernel_size": [3, 3]})
        verify_node_attributes(config, "MidIm2Col_0", {"kernel_size": [5, 5]})
        verify_node_attributes(config, "DeepIm2Col_0", {"kernel_size": [3, 3]})
        # Verify nested hierarchy (MainIfNode_0 -> MidIfNode_0 -> DeepIm2Col_0)
        verify_subgraph_hierarchy(config, ["MainIfNode_0", "MidIfNode_0"])
    finally:
        cleanup()


def test_extract_model_config_empty_attr_list():
    """Test that extracting with an empty attribute list returns only Defaults."""
    model = make_simple_model_with_im2col()
    
    config = extract_model_config(model, None, [])
    
    # Should only have Defaults, no node-specific configs
    assert "Defaults" in config
    assert "Im2Col_0" not in config


def test_extract_model_config_nonexistent_attr():
    """Test extracting attributes that don't exist on the nodes."""
    model = make_simple_model_with_im2col()
    
    # Try to extract an attribute that doesn't exist
    config = extract_model_config(model, None, ["nonexistent_attr"])
    
    # Should have Defaults but node should not appear since it has no matching attrs
    assert "Defaults" in config
    # The node won't appear in config if none of its attributes match
    assert "Im2Col_0" not in config


def test_verify_subgraph_hierarchy_validation():
    """Test that subgraph hierarchy verification works correctly."""
    model = make_model_with_subgraphs()
    config = extract_model_config(model, None, ["kernel_size"])
    
    # Should pass with correct hierarchy node
    verify_subgraph_hierarchy(config, "IfNode_0")
    
    # Should pass with list containing correct hierarchy node
    verify_subgraph_hierarchy(config, ["IfNode_0", "SomeOtherNode"])
    
    # Should fail with incorrect hierarchy node
    try:
        verify_subgraph_hierarchy(config, "NonExistentNode")
        assert False, "Should have raised assertion error for incorrect hierarchy"
    except AssertionError as e:
        assert "not found" in str(e)


def test_top_level_nodes_no_subgraph_hier():
    """Test that top-level models without subgraphs don't have subgraph_hier key."""
    # Test simple model (no subgraphs at all)
    model = make_simple_model_with_im2col()
    config = extract_model_config(model, None, ["kernel_size", "stride"])
    
    # Should have the expected structure
    verify_config_basic_structure(config)
    verify_node_attributes(config, "Im2Col_0", {"kernel_size": [3, 3], "stride": [2, 2]})
    
    # Should NOT have subgraph_hier since there are no subgraphs
    assert "subgraph_hier" not in config, "Top-level model without subgraphs should not have subgraph_hier key"
    
    # Test model with subgraphs - verify main level nodes exist but subgraph_hier is separate
    model_with_sub = make_model_with_subgraphs()
    config_with_sub = extract_model_config(model_with_sub, None, ["kernel_size"])
    
    # Should have both main graph and subgraph nodes
    assert "Im2Col_0" in config_with_sub  # Main graph node
    assert "SubIm2Col_0" in config_with_sub  # Subgraph node
    
    # Should have subgraph_hier tracking since there ARE subgraphs
    assert "subgraph_hier" in config_with_sub, "Model with subgraphs should have subgraph_hier key"
