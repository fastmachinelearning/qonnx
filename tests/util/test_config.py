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
from typing import List, Dict, Any, Tuple
# this is a pretend opset so that we can create
# qonnx custom ops with onnxscript
qops = Opset("qonnx.custom_op.general", 1)

@script(default_opset=op)
def main_graph_fn(main_inp: FLOAT[1, 28, 28, 1], condition: BOOL, nested_condition: BOOL) -> FLOAT[1, 4, 4, 144]:
    """Main graph with nested if statement in else branch."""
    im2col_0 = qops.Im2Col(main_inp, stride=[1, 1], kernel_size=[3, 3], 
                           pad_amount=[1, 1, 1, 1], input_shape=[1, 28, 28, 1])
    
    # Python if statement → ONNX If node with subgraph
    if condition:
        # Then branch: simple subgraph (2 levels)
        main_out = qops.Im2Col(im2col_0, stride=[2, 1], kernel_size=[5, 5], 
                               pad_amount=[2, 2, 2, 2], input_shape=[1, 14, 14, 144])
    else:
        im2col_1 = qops.Im2Col(im2col_0, stride=[2, 1], kernel_size=[7, 7], 
                               pad_amount=[3, 3, 3, 3], input_shape=[1, 14, 14, 144])
        # Else branch: nested if statement (3 levels)
        if nested_condition:
            main_out = qops.Im2Col(im2col_1, stride=[3, 1], kernel_size=[3, 2], 
                                   pad_amount=[1, 1, 1, 1], input_shape=[1, 4, 4, 144])
        else:
            main_out = qops.Im2Col(im2col_1, stride=[3, 2], kernel_size=[7, 7], 
                                   pad_amount=[3, 3, 3, 3], input_shape=[1, 4, 4, 144])
    
    return main_out


def build_expected_config_from_node(node: onnx.NodeProto, prefix = '') -> Dict[str, Any]:
    """Build expected config dictionary from a given ONNX node."""
    custom_op = getCustomOp(node)
    attrs = {}
    for attr in node.attribute:
        attrs[attr.name] = custom_op.get_nodeattr(attr.name)
    return {prefix + node.name: attrs}    
    

def make_im2col_test_model():
    """Create a simple ONNX model with a single Im2Col node."""    
    
    model_proto = main_graph_fn.to_model_proto()

    im2col_node = model_proto.graph.node[0]
    if_im2col_then_node = model_proto.graph.node[1].attribute[0].g.node[0]
    if_im2col_else_node = model_proto.graph.node[1].attribute[1].g.node[0]
    nested_if_im2col_then_node = model_proto.graph.node[1].attribute[1].g.node[1].attribute[0].g.node[0]
    nested_if_im2col_else_node = model_proto.graph.node[1].attribute[1].g.node[1].attribute[1].g.node[0]

    # this test assumes that all Im2Col nodes have the same name
    # to verify that node aliasing is handled correctly between nodes on 
    # the same and different levels of the hierarchy
    assert im2col_node.name == if_im2col_then_node.name
    assert im2col_node.name == if_im2col_else_node.name 
    assert im2col_node.name == nested_if_im2col_then_node.name
    assert im2col_node.name == nested_if_im2col_else_node.name
   
    expected_config = {}
    expected_config.update(build_expected_config_from_node(im2col_node))
    expected_config.update(build_expected_config_from_node(if_im2col_then_node, prefix='n1_then_branch_'))
    expected_config.update(build_expected_config_from_node(if_im2col_else_node, prefix='n1_else_branch_'))
    expected_config.update(build_expected_config_from_node(nested_if_im2col_then_node, prefix='n1_else_branch_n1_then_branch_'))
    expected_config.update(build_expected_config_from_node(nested_if_im2col_else_node, prefix='n1_else_branch_n1_else_branch_'))
    
    return ModelWrapper(model_proto), expected_config

def test_extract_model_config():
    """Test extraction of model config from models with and without subgraphs."""
    
    model, expected_config = make_im2col_test_model()
    
    attrs_to_extract = ["kernel_size", "stride", "pad_amount", "input_shape"]
    
    extracted_config = extract_model_config(model, subgraph_hier=None, attr_names_to_extract=attrs_to_extract)
    assert extracted_config == expected_config, "Extracted config does not match expected config"
    
    

# @pytest.mark.parametrize("model_name,model_factory", [
#     ("simple", make_simple_model_with_im2col),
#     ("nested", make_nested_subgraph_model),
# ])
# def test_extract_model_config_edge_cases(model_name, model_factory):
#     """Test edge cases: empty attribute list and nonexistent attributes.
    
#     Parameterized to ensure edge cases work for both simple and nested models.
#     """
#     model = model_factory()
    
#     # Edge case 1: Empty attribute list - no attributes requested
#     config = extract_model_config(model, None, [])
#     verify_config_basic_structure(config)
#     assert len(config) == 0, "Config should be empty when no attributes are requested"
    
#     # Edge case 2: Nonexistent attribute - attribute doesn't exist on any nodes
#     config = extract_model_config(model, None, ["nonexistent_attr"])
#     verify_config_basic_structure(config)
#     assert len(config) == 0, "Config should be empty when no nodes have matching attributes"

# @pytest.mark.parametrize("model_name,model_factory", [
#     ("simple", make_simple_model_with_im2col),
#     ("nested", make_nested_subgraph_model),
# ])
# def test_roundtrip_export_import(model_name, model_factory):
#     """Test export/import round-trip for models with and without subgraphs.
    
#     Parameterized test covering:
#     - simple: Model without subgraphs
#     - nested: Model with nested subgraphs (tests multi-level hierarchy)
#     """
#     from qonnx.transformation.general import ApplyConfig
    
#     # Helper to collect all Im2Col nodes from model and subgraphs recursively
#     def collect_im2col_attrs(model_wrapper, collected_attrs=None):
#         if collected_attrs is None:
#             collected_attrs = {}
        
#         for node in model_wrapper.graph.node:
#             if node.op_type == "Im2Col":
#                 inst = getCustomOp(node)
#                 collected_attrs[node.name] = {
#                     "kernel_size": inst.get_nodeattr("kernel_size"),
#                     "stride": inst.get_nodeattr("stride"),
#                     "pad_amount": inst.get_nodeattr("pad_amount")
#                 }
            
#             # Recursively check subgraphs
#             for attr in node.attribute:
#                 if attr.type == onnx.AttributeProto.GRAPH:
#                     subgraph = model_wrapper.make_subgraph_modelwrapper(attr.g)
#                     collect_im2col_attrs(subgraph, collected_attrs)
        
#         return collected_attrs
    
#     # Create first model and collect original attributes
#     model1 = model_factory()
#     original_attrs = collect_im2col_attrs(model1)
    
#     # Export config from first model
#     config, cleanup = extract_config_to_temp_json(model1, ["kernel_size", "stride", "pad_amount"])
    
#     try:
#         # Create a second model and modify its attributes
#         model2 = model_factory()
        
#         # Modify all Im2Col nodes to different values
#         def modify_all_nodes(model_wrapper):
#             for node in model_wrapper.graph.node:
#                 if node.op_type == "Im2Col":
#                     inst = getCustomOp(node)
#                     inst.set_nodeattr("kernel_size", [11, 11])
#                     inst.set_nodeattr("stride", [5, 5])
#                     inst.set_nodeattr("pad_amount", [7, 7, 7, 7])
                
#                 # Recursively modify subgraphs
#                 for attr in node.attribute:
#                     if attr.type == onnx.AttributeProto.GRAPH:
#                         subgraph = model_wrapper.make_subgraph_modelwrapper(attr.g)
#                         modify_all_nodes(subgraph)
        
#         modify_all_nodes(model2)
        
#         # Apply the original config to model2
#         with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
#             config_with_defaults = config.copy()
#             config_with_defaults["Defaults"] = {}
#             json.dump(config_with_defaults, f, indent=2)
#             config_json_file = f.name
        
#         model2 = model2.transform(ApplyConfig(config_json_file))
        
#         # Collect attributes from model2 after applying config
#         restored_attrs = collect_im2col_attrs(model2)
        
#         # Verify all nodes in model2 now match original_attrs from model1
#         assert len(restored_attrs) == len(original_attrs), \
#             f"Expected {len(original_attrs)} nodes, got {len(restored_attrs)}"
        
#         for node_name in original_attrs:
#             assert node_name in restored_attrs, f"Node {node_name} not found after applying config"
#             assert restored_attrs[node_name]["kernel_size"] == original_attrs[node_name]["kernel_size"], \
#                 f"Node {node_name} kernel_size not restored"
#             assert restored_attrs[node_name]["stride"] == original_attrs[node_name]["stride"], \
#                 f"Node {node_name} stride not restored"
#             assert restored_attrs[node_name]["pad_amount"] == original_attrs[node_name]["pad_amount"], \
#                 f"Node {node_name} pad_amount not restored"
        
#         # Cleanup
#         if os.path.exists(config_json_file):
#             os.remove(config_json_file)
#     finally:
#         cleanup()


# @pytest.mark.parametrize("model_name,model_factory", [
#     ("simple", make_simple_model_with_im2col),
# ])
# def test_roundtrip_partial_config(model_name, model_factory):
#     """Test that ApplyConfig only modifies specified attributes, leaving others unchanged.
    
#     Note: Only testing with simple model as nested model config application through subgraphs
#     has complexities that make partial config verification difficult.
#     """
#     from qonnx.transformation.general import ApplyConfig
    
#     # Helper to collect and modify all Im2Col nodes recursively
#     def collect_and_store_attrs(model_wrapper, original_attrs=None):
#         if original_attrs is None:
#             original_attrs = {}
#         for node in model_wrapper.graph.node:
#             if node.op_type == "Im2Col":
#                 inst = getCustomOp(node)
#                 original_attrs[node.name] = {
#                     "kernel_size": inst.get_nodeattr("kernel_size"),
#                     "stride": inst.get_nodeattr("stride"),
#                     "pad_amount": inst.get_nodeattr("pad_amount")
#                 }
#             for attr in node.attribute:
#                 if attr.type == onnx.AttributeProto.GRAPH:
#                     subgraph = model_wrapper.make_subgraph_modelwrapper(attr.g)
#                     collect_and_store_attrs(subgraph, original_attrs)
#         return original_attrs
    
#     def modify_all_attrs(graph_proto):
#         """Modify attributes directly in the graph proto (not through wrapper)."""
#         for node in graph_proto.node:
#             if node.op_type == "Im2Col":
#                 inst = getCustomOp(node)
#                 inst.set_nodeattr("kernel_size", [7, 7])
#                 inst.set_nodeattr("stride", [4, 4])
#                 inst.set_nodeattr("pad_amount", [9, 9, 9, 9])
#             for attr in node.attribute:
#                 if attr.type == onnx.AttributeProto.GRAPH:
#                     modify_all_attrs(attr.g)
    
#     def verify_attrs(model_wrapper, original_attrs):
#         for node in model_wrapper.graph.node:
#             if node.op_type == "Im2Col":
#                 inst = getCustomOp(node)
#                 # kernel_size and stride should be restored
#                 assert inst.get_nodeattr("kernel_size") == original_attrs[node.name]["kernel_size"]
#                 assert inst.get_nodeattr("stride") == original_attrs[node.name]["stride"]
#                 # pad_amount should remain modified (not in config)
#                 assert inst.get_nodeattr("pad_amount") == [9, 9, 9, 9]
#             for attr in node.attribute:
#                 if attr.type == onnx.AttributeProto.GRAPH:
#                     subgraph = model_wrapper.make_subgraph_modelwrapper(attr.g)
#                     verify_attrs(subgraph, original_attrs)
    
#     # Create model and store original values
#     model = model_factory()
#     original_attrs = collect_and_store_attrs(model)
    
#     # Export only kernel_size and stride (not pad_amount)
#     config, cleanup = extract_config_to_temp_json(model, ["kernel_size", "stride"])
    
#     try:
#         # Modify all attributes (work directly with graph proto)
#         modify_all_attrs(model.graph)
        
#         # Create config with Defaults
#         with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
#             config_with_defaults = config.copy()
#             config_with_defaults["Defaults"] = {}
#             json.dump(config_with_defaults, f, indent=2)
#             config_json_file = f.name
        
#         # Apply config
#         model = model.transform(ApplyConfig(config_json_file))
        
#         # Verify partial restoration
#         verify_attrs(model, original_attrs)
        
#         # Cleanup
#         if os.path.exists(config_json_file):
#             os.remove(config_json_file)
#     finally:
#         cleanup()
