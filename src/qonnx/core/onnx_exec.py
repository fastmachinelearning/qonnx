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


import copy
import numpy as np
import onnx.helper as helper
import onnxruntime as rt
import warnings

import qonnx.analysis.topology as ta
import qonnx.core.execute_custom_node as ex_cu_node
from qonnx.util.basic import (
    get_preferred_onnx_opset,
    get_sanitize_quant_tensors,
    is_finn_op,
    qonnx_make_model,
    sanitize_quant_values,
)


def execute_node(node, context, graph, return_full_exec_context=False, opset_version=get_preferred_onnx_opset()):
    """Executes a single node by using onnxruntime or with a custom function.

    Input/output provided via context."""

    if is_finn_op(node.domain):
        ex_cu_node.execute_custom_node(node, context, graph, onnx_opset_version=opset_version)
    else:
        # onnxruntime unfortunately does not implement run_node as defined by ONNX,
        # it can only execute entire models -- so we create a model which solely
        # consists of our current node.
        # note: ensure that the same ValueInfo does not appear both in
        # graph.value_info as well as graph.output or graph.input
        # nodes with multiple outputs that are a mix of value_info and
        # input/outputs may get them reordered below
        # note: a node's input may (also) be a top-level input or output
        node_inputs = list(filter(lambda x: x.name in node.input, graph.input))
        node_inputs += list(filter(lambda x: x.name in node.input, graph.output))
        node_inputs += list(filter(lambda x: x.name in node.input, graph.value_info))
        node_outputs = list(filter(lambda x: x.name in node.output, graph.output))
        node_outputs += list(filter(lambda x: x.name in node.output, graph.value_info))
        for attr in node.attribute:
            if attr.type == 5:
                subgraph = attr.g
                for subgraph_node in subgraph.node:
                    subgraph_node_inputs = list(filter(lambda x: x.name in subgraph_node.input, graph.value_info))
                    new_inps = list(filter(lambda x: x not in node_inputs, subgraph_node_inputs))
                    node_inputs += new_inps
        node_graph = helper.make_graph(
            nodes=[node],
            name="single-node-exec",
            inputs=node_inputs,
            outputs=node_outputs,
        )
        node_model = qonnx_make_model(node_graph)
        node_model.opset_import[0].version = opset_version
        input_dict = dict()
        for inp in node.input:
            input_dict[inp] = context[inp]

        sess = rt.InferenceSession(node_model.SerializeToString())
        output_list = sess.run(None, input_dict)

        for output_ind in range(len(node.output)):
            # get the name of the target buffer from node.output
            outp = node.output[output_ind]

            # retrieve the index of that name in node_outputs
            for i in range(len(node_outputs)):
                if outp == node_outputs[i].name:
                    list_ind = i

            # use that index to index output_list
            if output_list[list_ind].shape != context[outp].shape:
                warnings.warn(
                    """Output shapes disagree after node %s execution:
                    found %s vs expected %s"""
                    % (str(node), str(output_list[list_ind].shape), str(context[outp].shape))
                )
            context[outp] = output_list[list_ind]


def execute_onnx(model, input_dict, return_full_exec_context=False, start_node=None, end_node=None):
    """Executes given ONNX ModelWrapper with given named inputs.

    If return_full_exec_context is False, a dict of named outputs is returned
    as indicated by the model.graph.output.

    If return return_full_exec_context is True, the full set of tensors used by
    the execution (including inputs, weights, activations and final outputs)
    will be returned as a dict.

    When start_node and end_node are set to None, the whole graph is executed.
    If they are set to particular ONNX nodes, only the subgraph between (and
    including) those nodes is executed.
    """

    if not model.check_all_tensor_shapes_specified():
        raise Exception("Found unspecified tensor shapes, try infer_shapes")
    ret = model.analysis(ta.nodes_topologically_sorted)
    assert (
        ret["nodes_topologically_sorted"] is True
    ), """Nodes must be
    topologically sorted."""

    graph = model.graph
    # first, we need to make sure that every variable required by the graph has
    # some buffer associated with it. this includes graph inputs (which includes
    # the input data as well as the trained parameters) and the graph ValueInfo
    # (intermediate tensors between layers)
    # this is provided by the execution_context, which is a dict of np.ndarray
    execution_context = model.make_empty_exec_context()
    # fill in any inputs provided to this function
    for inp_name in input_dict.keys():
        if inp_name in execution_context:
            if execution_context[inp_name].shape == input_dict[inp_name].shape:
                execution_context[inp_name] = input_dict[inp_name]
            else:
                raise Exception(
                    "Shape mismatch for provided input %s: found %s expected %s "
                    % (
                        inp_name,
                        str(execution_context[inp_name].shape),
                        str(input_dict[inp_name].shape),
                    )
                )
        # else:
        # raise Exception("Provided input not found in graph context: %s" % inp_name)

    # check if model has an execution mode set
    # if None, execute model node by node using execute_node()
    model_exec_mode = model.get_metadata_prop("exec_mode")
    if (model_exec_mode is None) or (model_exec_mode == ""):
        # extract opset version for node-by-node execution
        opset_version = model.model.opset_import[0].version
        # execute the model node by node
        # we can simply walk down the list since the ONNX spec guarantees that it is
        # topologically sorted
        subgraph = []
        if start_node is None:
            start_node = model.graph.node[0]
        if end_node is None:
            end_node = model.graph.node[-1]
        # select the nodes between specified start/end nodes
        start_ind = model.get_node_index(start_node)
        end_ind = model.get_node_index(end_node) + 1
        assert end_ind >= start_ind, "Start/end nodes must define valid subgraph"
        subgraph = graph.node[start_ind:end_ind]
        for node in subgraph:
            if get_sanitize_quant_tensors() != 0:
                # round input values to match quantization annotation
                execution_context = sanitize_quant_values(model, node.input, execution_context)
            execute_node(node, execution_context, graph, return_full_exec_context, opset_version)
            if get_sanitize_quant_tensors() != 0:
                # round output values to quantization annotation
                execution_context = sanitize_quant_values(model, node.output, execution_context)
    else:
        raise Exception('Metadata property "exec_mode" is set to an unknown value.')

    if return_full_exec_context:
        return execution_context
    else:
        # provide outputs as dict
        output_dict = dict()
        for out_tensor in graph.output:
            out_name = out_tensor.name
            output_dict[out_name] = execution_context[out_name]
        return output_dict


def execute_onnx_and_make_model(model, input_dict):
    """Executes given ONNX ModelWrapper with given named inputs and return a new
    ModelWrapper where an initializer is provided for each tensor as taken from
    the execution. This new model is useful for debugging, since it contains
    all the intermediate activation values."""

    # retrieve the full execution context
    execution_context = execute_onnx(model, input_dict, True)
    new_model = copy.deepcopy(model)
    # create value_info entries and initializers for everything
    for i in execution_context.keys():
        new_model.set_initializer(i, execution_context[i])
    for vi in new_model.graph.value_info:
        new_model.graph.output.append(vi)
    return new_model


def compare_execution(
    model_a,
    model_b,
    input_dict,
    compare_fxn=lambda x, y: np.isclose(x, y, atol=1e-3).all(),
):
    """Executes two ONNX models and compare their outputs using given function.

    compare_fxn should take in two tensors and return a Boolean"""
    # compare values from first output tensors produced
    res_a = list(execute_onnx(model_a, input_dict).items())[0][1]
    res_b = list(execute_onnx(model_b, input_dict).items())[0][1]
    return compare_fxn(res_a, res_b)
