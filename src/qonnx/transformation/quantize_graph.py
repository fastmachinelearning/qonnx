# Copyright (c) 2024 Advanced Micro Devices, Inc.
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
# * Neither the name of qonnx nor the names of its
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


import numpy as np
import onnx
from onnx import TensorProto

from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.cleanup import cleanup_model


def create_quantnode(
    model,
    quantnode_input,
    quantnode_output_shape,
    scale_value,
    zeropoint_value,
    bitwidth_value,
    narrow,
    signed,
    rounding_mode,
):
    quant_tensor = onnx.helper.make_tensor_value_info(
        model.make_new_valueinfo_name(), TensorProto.FLOAT, quantnode_output_shape
    )
    model.graph.value_info.append(quant_tensor)

    scale_tensor = np.array(scale_value).astype(np.float32)
    s_value = onnx.helper.make_tensor_value_info(model.make_new_valueinfo_name(), TensorProto.FLOAT, quantnode_output_shape)
    model.graph.value_info.append(s_value)
    model.set_initializer(s_value.name, scale_tensor)

    zeropt_tensor = np.array(zeropoint_value).astype(np.float32)
    z_value = onnx.helper.make_tensor_value_info(model.make_new_valueinfo_name(), TensorProto.FLOAT, quantnode_output_shape)
    model.graph.value_info.append(z_value)
    model.set_initializer(z_value.name, zeropt_tensor)

    bitwidth_tensor = np.array(bitwidth_value).astype(np.float32)
    b_value = onnx.helper.make_tensor_value_info(model.make_new_valueinfo_name(), TensorProto.FLOAT, [1])
    model.graph.value_info.append(b_value)
    model.set_initializer(b_value.name, bitwidth_tensor)

    quantnode = onnx.helper.make_node(
        "Quant",
        inputs=[quantnode_input, s_value.name, z_value.name, b_value.name],
        outputs=[quant_tensor.name],
        name="Quant_" + quantnode_input,
        narrow=narrow,
        signed=signed,
        rounding_mode=rounding_mode,
    )

    return quantnode, quant_tensor


def adjust_graph(model, input_positions, node_name, quantized_nodes):
    for pos in input_positions:
        node_details = (node_name, pos[0])
        if node_details not in quantized_nodes:  # not quantizing for same node_inp/out index.
            node_in_focus = model.get_node_from_name(node_name)

            if pos[0][0] == "input":
                quantnode_input = node_in_focus.input[pos[0][1]]
                consumer_node = node_in_focus
                producer_node = model.find_producer(quantnode_input)
                if producer_node is None or producer_node.op_type != "Quant":
                    quantization_to_perform = True
                else:
                    quantization_to_perform = False
            else:
                quantnode_input = node_in_focus.output[pos[0][1]]
                consumer_node = model.find_consumer(quantnode_input)
                producer_node = model.find_producer(quantnode_input)
                if consumer_node is None or consumer_node.op_type != "Quant":
                    quantization_to_perform = True
                else:
                    quantization_to_perform = False
            if quantization_to_perform is True:
                quantnode_output_shape = model.get_tensor_shape(quantnode_input)  # Step: 3
                quantnode, quant_tensor = create_quantnode(
                    model,
                    quantnode_input,
                    quantnode_output_shape,
                    scale_value=pos[1][0],
                    zeropoint_value=pos[1][1],
                    bitwidth_value=pos[1][2],
                    narrow=pos[1][3],
                    signed=pos[1][4],
                    rounding_mode=pos[1][5],
                )

                if consumer_node is not None:
                    node_pos = model.get_node_index(consumer_node)
                    model.graph.node[node_pos].input[pos[0][1]] = quant_tensor.name
                    model.graph.node.append(quantnode)
                else:
                    model.graph.value_info.remove(quant_tensor)
                    model.graph.node.append(quantnode)
                    model.graph.output.insert(0, quant_tensor)
                    model.graph.output.pop(1)

                model = model.transform(SortGraph())
                quantized_nodes.append(node_details)
            else:
                print(f"{pos[0][0]} index {pos[0][1]} of {node_name} is already quantized.")
        else:
            print(f"{pos[0][0]} index {pos[0][1]} of {node_name} is already quantized.")
            continue

    return model


class QuantizeGraph(Transformation):
    """This transformation can be used to introduce a Quant node for a specific type of node in the graph.
    Users would be able to specify the location of the quant node by providing the input and output index
    as the parameters.

        1) Expectations:
            a) Onnx model in the modelwraper format.
            b) Model must be cleaned using qonnx.util.cleanup.cleanup_model()
            c) Batchsize to be set.

        2) Steps to transform are:
            Step1: Finding the input for the quant node.
            Step2: Finding the consumer of the quant node output.
            Step3: Finding the shape for the output tensor of quant node.
            Note: The output tensor of the quant node must have the same shape as the consumer of the input
                    to the quant node.

        3) Input:
            A dict "quantnode_map" specifying the criterion, positions, and input parameters like
            scale, bitwidth, zeropoint, and others for a specific quantnode.

            Criterion:
                a) name: This will allow users to add quant nodes for specific node like "Conv_0" and "Gemm_0".
                        Note: using this users can have quant nodes with different parameters. Ex: quantizing
                        "Conv_0" and "Conv_1"  with bitwidth of 4 and 6, respectively.
                b) op_type: This will allow users to add quant nodes for all nodes of a particular op_type such
                            as, "Conv", "Gemm", and others.
                            Note: All quant nodes created using op_type criterion will have the same input
                            parameters (scale, zeropoint, bitwidth, and others.)
                c) name and op_type: In this case, quant nodes will be added with precedence to "Name"
                                    in comparison to "op_type".

            Positions:  ("input", index) or  ("output", index)
                a) "input":  indicates that the user want to quantize the input of the selected node.
                b) "output": indicates that the user want to quantize the output of the selected node.
                c) index: refers to the input/output index to quantize (a node can have multiple inputs and outputs)

            Parameters (to quant node) are provided as (scale, zeropoint, bitwidth, narrow, signed, rounding_mode)

                a) Inputs: scale, zeropoint, bitwidth.
                b) Attributes: narrow, signed, rounding_mode.

        4) Assert:
                a) The input is a dictionary representing the node names as keys and a list of quant positions
                   as values.
                b) The input dictionary must have atleast one mac node (Conv, gemm, matmul) for the transformation.

        5) Return:
                Returns a model with new quant nodes created at the positions specified using the "quantnode_map".

        6) Example:
                quantnode_map = {"name": {"Conv_0": [(("input", 0), (1, 0, 8, 0, 1, "ROUND")),
                                                (("input", 1), (1, 0, 8, 0, 1, "ROUND")),
                                                (("output", 0), (1, 0, 8, 0, 1, "ROUND"))],
                                        "Conv_1": [(("input", 0), (1, 0, 8, 0, 1, "ROUND"))],
                                        "Conv_2": [(("input", 1), (1, 0, 8, 0, 1, "ROUND")),
                                                (("output", 0), (1, 0, 8, 0, 1, "ROUND"))]},

                                "op_type": {"Gemm": [(("input", 0), (1, 0, 8, 0, 1, "ROUND")),
                                                   (("input", 1), (1, 0, 8, 0, 1, "ROUND")),
                                                   (("input", 2), (1, 0, 8, 0, 1, "ROUND")),
                                                   (("output", 0), (1, 0, 8, 0, 1, "ROUND"))]}}
    """

    def __init__(self, quantnode_map):
        super().__init__()
        self.quantnode_map = quantnode_map

    def apply(self, model):
        model = model.transform(InferShapes())
        if type(self.quantnode_map) == dict:
            selection_type = self.quantnode_map.keys()
            if set(selection_type) <= {"name", "op_type"}:
                quantized_nodes = []
                if "name" in selection_type:
                    by_name = self.quantnode_map["name"]  # dict with unique names and list of positions.
                    node_list_by_name = by_name.keys()  # node names specified by the user for quant nodes.
                    for node_name in node_list_by_name:
                        input_positions = by_name[node_name]  # input positions to introduce quant nodes.
                        model = adjust_graph(model, input_positions, node_name, quantized_nodes)
                if "op_type" in selection_type:
                    by_op_type = self.quantnode_map["op_type"]  # dict with the unique names and list of positions.
                    op_list = by_op_type.keys()
                    for op in op_list:
                        node_list = model.get_nodes_by_op_type(op)  # List of all nodes with the operation type "op".
                        input_positions = by_op_type[op]
                        for node in node_list:
                            node_name = node.name
                            model = adjust_graph(model, input_positions, node_name, quantized_nodes)
                model = cleanup_model(model)
            else:
                raise Exception("Unsupported selection type")
        else:
            raise TypeError("Input must be a dictionary.")

        graph_modified = False

        return (model, graph_modified)
