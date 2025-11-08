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


def make_simple_model_with_im2col():
    """Create a simple model with Im2Col nodes that have configurable attributes."""
    
    # Create input/output tensors
    inp = helper.make_tensor_value_info("inp", onnx.TensorProto.FLOAT, [1, 14, 14, 3])
    out = helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [1, 7, 7, 27])
    
    # Create Im2Col node with some attributes
    im2col_node = helper.make_node(
        "Im2Col",
        inputs=["inp"],
        outputs=["out"],
        domain="qonnx.custom_op.general",
        stride=[2, 2],
        kernel_size=[3, 3],
        input_shape="(1, 14, 14, 3)",
        pad_amount=[0, 0, 0, 0],
        name="Im2Col_0"
    )
    
    graph = helper.make_graph(
        nodes=[im2col_node],
        name="simple_graph",
        inputs=[inp],
        outputs=[out],
    )
    
    model = qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    return ModelWrapper(model)


def make_model_with_subgraphs():
    """Create a model with nodes that contain subgraphs with Im2Col operations."""
    
    # Create a subgraph with Im2Col nodes
    subgraph_inp = helper.make_tensor_value_info("sub_inp", onnx.TensorProto.FLOAT, [1, 14, 14, 3])
    subgraph_out = helper.make_tensor_value_info("sub_out", onnx.TensorProto.FLOAT, [1, 7, 7, 27])
    
    # Create Im2Col nodes in the subgraph with different attributes
    sub_im2col_1 = helper.make_node(
        "Im2Col",
        inputs=["sub_inp"],
        outputs=["sub_intermediate"],
        domain="qonnx.custom_op.general",
        stride=[2, 2],
        kernel_size=[3, 3],
        input_shape="(1, 14, 14, 3)",
        pad_amount=[1, 1, 1, 1],
        name="SubIm2Col_0"
    )
    
    sub_im2col_2 = helper.make_node(
        "Im2Col",
        inputs=["sub_intermediate"],
        outputs=["sub_out"],
        domain="qonnx.custom_op.general",
        stride=[1, 1],
        kernel_size=[5, 5],
        input_shape="(1, 7, 7, 27)",
        pad_amount=[2, 2, 2, 2],
        name="SubIm2Col_1"
    )
    
    subgraph = helper.make_graph(
        nodes=[sub_im2col_1, sub_im2col_2],
        name="subgraph_1",
        inputs=[subgraph_inp],
        outputs=[subgraph_out],
    )
    
    # Create main graph with a node that has a subgraph attribute
    main_inp = helper.make_tensor_value_info("main_inp", onnx.TensorProto.FLOAT, [1, 14, 14, 3])
    main_out = helper.make_tensor_value_info("main_out", onnx.TensorProto.FLOAT, [1, 7, 7, 27])
    
    # Create a top-level Im2Col node
    main_im2col = helper.make_node(
        "Im2Col",
        inputs=["main_inp"],
        outputs=["main_intermediate"],
        domain="qonnx.custom_op.general",
        stride=[1, 1],
        kernel_size=[7, 7],
        input_shape="(1, 14, 14, 3)",
        pad_amount=[3, 3, 3, 3],
        name="Im2Col_0"
    )
    
    # Create a node with subgraph (using If node from ONNX standard)
    # In ONNX, nodes like If, Loop, and Scan have graph attributes
    if_node = helper.make_node(
        "If",
        inputs=["condition"],
        outputs=["main_out"],
        domain="",  # Standard ONNX operator
        name="IfNode_0"
    )
    # Add the subgraph as the 'then_branch' and 'else_branch' attributes
    if_node.attribute.append(helper.make_attribute("then_branch", subgraph))
    if_node.attribute.append(helper.make_attribute("else_branch", subgraph))
    
    # Create a condition input for the If node
    condition_init = helper.make_tensor("condition", onnx.TensorProto.BOOL, [], [True])
    
    main_graph = helper.make_graph(
        nodes=[main_im2col, if_node],
        name="main_graph",
        inputs=[main_inp],
        outputs=[main_out],
        initializer=[condition_init],
    )
    
    model = qonnx_make_model(main_graph, opset_imports=[helper.make_opsetid("", 11)])
    return ModelWrapper(model)


def make_nested_subgraph_model():
    """Create a model with nested subgraphs (subgraph within a subgraph)."""
    
    # Create the deepest subgraph (level 2)
    deep_inp = helper.make_tensor_value_info("deep_inp", onnx.TensorProto.FLOAT, [1, 8, 8, 16])
    deep_out = helper.make_tensor_value_info("deep_out", onnx.TensorProto.FLOAT, [1, 4, 4, 144])
    
    deep_im2col = helper.make_node(
        "Im2Col",
        inputs=["deep_inp"],
        outputs=["deep_out"],
        domain="qonnx.custom_op.general",
        stride=[2, 2],
        kernel_size=[3, 3],
        input_shape="(1, 8, 8, 16)",
        pad_amount=[0, 0, 0, 0],
        name="DeepIm2Col_0"
    )
    
    deep_subgraph = helper.make_graph(
        nodes=[deep_im2col],
        name="deep_subgraph",
        inputs=[deep_inp],
        outputs=[deep_out],
    )
    
    # Create middle subgraph (level 1) that contains the deep subgraph
    mid_inp = helper.make_tensor_value_info("mid_inp", onnx.TensorProto.FLOAT, [1, 14, 14, 3])
    mid_out = helper.make_tensor_value_info("mid_out", onnx.TensorProto.FLOAT, [1, 4, 4, 144])
    
    mid_im2col = helper.make_node(
        "Im2Col",
        inputs=["mid_inp"],
        outputs=["mid_intermediate"],
        domain="qonnx.custom_op.general",
        stride=[1, 1],
        kernel_size=[5, 5],
        input_shape="(1, 14, 14, 3)",
        pad_amount=[2, 2, 2, 2],
        name="MidIm2Col_0"
    )
    
    mid_if_node = helper.make_node(
        "If",
        inputs=["mid_condition"],
        outputs=["mid_out"],
        domain="",  # Standard ONNX operator
        name="MidIfNode_0"
    )
    mid_if_node.attribute.append(helper.make_attribute("then_branch", deep_subgraph))
    mid_if_node.attribute.append(helper.make_attribute("else_branch", deep_subgraph))
    
    mid_condition_init = helper.make_tensor("mid_condition", onnx.TensorProto.BOOL, [], [True])
    
    mid_subgraph = helper.make_graph(
        nodes=[mid_im2col, mid_if_node],
        name="mid_subgraph",
        inputs=[mid_inp],
        outputs=[mid_out],
        initializer=[mid_condition_init],
    )
    
    # Create main graph
    main_inp = helper.make_tensor_value_info("main_inp", onnx.TensorProto.FLOAT, [1, 28, 28, 1])
    main_out = helper.make_tensor_value_info("main_out", onnx.TensorProto.FLOAT, [1, 4, 4, 144])
    
    main_im2col = helper.make_node(
        "Im2Col",
        inputs=["main_inp"],
        outputs=["main_intermediate"],
        domain="qonnx.custom_op.general",
        stride=[2, 2],
        kernel_size=[3, 3],
        input_shape="(1, 28, 28, 1)",
        pad_amount=[1, 1, 1, 1],
        name="MainIm2Col_0"
    )
    
    main_if_node = helper.make_node(
        "If",
        inputs=["main_condition"],
        outputs=["main_out"],
        domain="",  # Standard ONNX operator
        name="MainIfNode_0"
    )
    main_if_node.attribute.append(helper.make_attribute("then_branch", mid_subgraph))
    main_if_node.attribute.append(helper.make_attribute("else_branch", mid_subgraph))
    
    main_condition_init = helper.make_tensor("main_condition", onnx.TensorProto.BOOL, [], [True])
    
    main_graph = helper.make_graph(
        nodes=[main_im2col, main_if_node],
        name="main_graph",
        inputs=[main_inp],
        outputs=[main_out],
        initializer=[main_condition_init],
    )
    
    model = qonnx_make_model(main_graph, opset_imports=[helper.make_opsetid("", 11)])
    return ModelWrapper(model)


def test_extract_model_config_simple():
    """Test extracting config from a simple model without subgraphs."""
    model = make_simple_model_with_im2col()
    
    # Extract config for kernel_size and stride attributes
    config = extract_model_config(model, None, ["kernel_size", "stride"])
    
    # Check that the config contains the expected keys
    assert "Defaults" in config
    assert "Im2Col_0" in config
    
    # Check that the attributes were extracted correctly
    assert config["Im2Col_0"]["kernel_size"] == [3, 3]
    assert config["Im2Col_0"]["stride"] == [2, 2]


def test_extract_model_config_to_json_simple():
    """Test extracting config to JSON from a simple model without subgraphs."""
    model = make_simple_model_with_im2col()
    
    # Create a temporary file for the JSON output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_filename = f.name
    
    try:
        # Extract config to JSON
        extract_model_config_to_json(model, json_filename, ["kernel_size", "stride", "pad_amount"])
        
        # Read the JSON file and verify its contents
        with open(json_filename, 'r') as f:
            config = json.load(f)
        
        assert "Defaults" in config
        assert "Im2Col_0" in config
        assert config["Im2Col_0"]["kernel_size"] == [3, 3]
        assert config["Im2Col_0"]["stride"] == [2, 2]
        assert config["Im2Col_0"]["pad_amount"] == [0, 0, 0, 0]
    finally:
        # Clean up the temporary file
        if os.path.exists(json_filename):
            os.remove(json_filename)


def test_extract_model_config_with_subgraphs():
    """Test extracting config from a model with subgraphs."""
    model = make_model_with_subgraphs()
    
    # Extract config for kernel_size, stride, and pad_amount attributes
    config = extract_model_config(model, None, ["kernel_size", "stride", "pad_amount"])
    
    # Check that the config contains the expected keys
    assert "Defaults" in config
    
    # Check main graph node
    assert "Im2Col_0" in config
    assert config["Im2Col_0"]["kernel_size"] == [7, 7]
    assert config["Im2Col_0"]["stride"] == [1, 1]
    assert config["Im2Col_0"]["pad_amount"] == [3, 3, 3, 3]
    
    # Check subgraph nodes
    assert "SubIm2Col_0" in config
    assert config["SubIm2Col_0"]["kernel_size"] == [3, 3]
    assert config["SubIm2Col_0"]["stride"] == [2, 2]
    assert config["SubIm2Col_0"]["pad_amount"] == [1, 1, 1, 1]
    
    assert "SubIm2Col_1" in config
    assert config["SubIm2Col_1"]["kernel_size"] == [5, 5]
    assert config["SubIm2Col_1"]["stride"] == [1, 1]
    assert config["SubIm2Col_1"]["pad_amount"] == [2, 2, 2, 2]
    
    # Check that subgraph hierarchy is tracked
    assert "subgraph_hier" in config


def test_extract_model_config_to_json_with_subgraphs():
    """Test extracting config to JSON from a model with subgraphs."""
    model = make_model_with_subgraphs()
    
    # Create a temporary file for the JSON output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_filename = f.name
    
    try:
        # Extract config to JSON
        extract_model_config_to_json(model, json_filename, ["kernel_size", "stride", "pad_amount"])
        
        # Read the JSON file and verify its contents
        with open(json_filename, 'r') as f:
            config = json.load(f)
        
        # Verify the structure
        assert "Defaults" in config
        assert "Im2Col_0" in config
        assert "SubIm2Col_0" in config
        assert "SubIm2Col_1" in config
        
        # Verify main graph node attributes
        assert config["Im2Col_0"]["kernel_size"] == [7, 7]
        assert config["Im2Col_0"]["stride"] == [1, 1]
        
        # Verify subgraph node attributes
        assert config["SubIm2Col_0"]["kernel_size"] == [3, 3]
        assert config["SubIm2Col_0"]["pad_amount"] == [1, 1, 1, 1]
        
        assert config["SubIm2Col_1"]["kernel_size"] == [5, 5]
        assert config["SubIm2Col_1"]["pad_amount"] == [2, 2, 2, 2]
        
        # Check that subgraph hierarchy information is present
        assert "subgraph_hier" in config
        
    finally:
        # Clean up the temporary file
        if os.path.exists(json_filename):
            os.remove(json_filename)


def test_extract_model_config_nested_subgraphs():
    """Test extracting config from a model with nested subgraphs."""
    model = make_nested_subgraph_model()
    model.save('nested_subgraph_model.onnx')
    # Extract config for kernel_size and stride attributes
    config = extract_model_config(model, None, ["kernel_size", "stride"])
    
    # Check that the config contains nodes from all levels
    assert "Defaults" in config
    
    # Main graph
    assert "MainIm2Col_0" in config
    assert config["MainIm2Col_0"]["kernel_size"] == [3, 3]
    assert config["MainIm2Col_0"]["stride"] == [2, 2]
    
    # Middle subgraph
    assert "MidIm2Col_0" in config
    assert config["MidIm2Col_0"]["kernel_size"] == [5, 5]
    assert config["MidIm2Col_0"]["stride"] == [1, 1]
    
    # Deep subgraph
    assert "DeepIm2Col_0" in config
    assert config["DeepIm2Col_0"]["kernel_size"] == [3, 3]
    assert config["DeepIm2Col_0"]["stride"] == [2, 2]


def test_extract_model_config_to_json_nested_subgraphs():
    """Test extracting config to JSON from a model with nested subgraphs."""
    model = make_nested_subgraph_model()
    
    # Create a temporary file for the JSON output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_filename = f.name
    
    try:
        # Extract config to JSON
        extract_model_config_to_json(model, json_filename, ["kernel_size", "stride", "pad_amount"])
        
        # Read the JSON file and verify its contents
        with open(json_filename, 'r') as f:
            config = json.load(f)
        
        # Verify all nodes from all nesting levels are present
        assert "Defaults" in config
        assert "MainIm2Col_0" in config
        assert "MidIm2Col_0" in config
        assert "DeepIm2Col_0" in config
        
        # Verify attributes from each level
        assert config["MainIm2Col_0"]["kernel_size"] == [3, 3]
        assert config["MidIm2Col_0"]["kernel_size"] == [5, 5]
        assert config["DeepIm2Col_0"]["kernel_size"] == [3, 3]
        
        # Verify subgraph hierarchy tracking
        assert "subgraph_hier" in config
        
    finally:
        # Clean up the temporary file
        if os.path.exists(json_filename):
            os.remove(json_filename)


def test_extract_model_config_empty_attr_list():
    """Test that extracting with an empty attribute list returns only Defaults."""
    model = make_simple_model_with_im2col()
    mod
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
