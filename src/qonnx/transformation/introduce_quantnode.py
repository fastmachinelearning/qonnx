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

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model
from qonnx.util.cleanup import cleanup_model


class graph_util:
    def get_node_id(self, model):
        node_index = {}
        node_ind = 0
        for node in model.graph.node:
            node_index[node.name] = node_ind
            node_ind += 1
        return node_index

    def node_from_name(self, model, node_name):
        for node in model.graph.node:
            if node.name == node_name:
                return node

    def identify_nodes(self, model, node_type):
        node_list = []
        for node in model.graph.node:
            if node.op_type == node_type:
                node_list.append(node)
        return node_list

    def create_node(
        self,
        model,
        quantnode_input,
        quantnode_output_shape,
        node_count,
        tensor_count,
        scale_value,
        zeropoint_value,
        bitwidth_value,
        narrow,
        signed,
        rounding_mode,
    ):
        quantnode_output_dtype = DataType["UINT8"]
        quant_tensor = onnx.helper.make_tensor_value_info(
            model.make_new_valueinfo_name(), TensorProto.FLOAT, quantnode_output_shape
        )
        model.graph.value_info.append(quant_tensor)
        model.set_tensor_datatype(quant_tensor.name, quantnode_output_dtype)

        stationary_input_dtype = DataType["FLOAT32"]
        scale_tensor = np.array(scale_value).astype(np.float32)
        s_value = onnx.helper.make_tensor_value_info(
            model.make_new_valueinfo_name(), TensorProto.FLOAT, quantnode_output_shape
        )
        model.graph.value_info.append(s_value)
        model.set_tensor_datatype(s_value.name, stationary_input_dtype)
        model.set_initializer(s_value.name, scale_tensor)

        zeropt_tensor = np.array(zeropoint_value).astype(np.float32)
        z_value = onnx.helper.make_tensor_value_info(
            model.make_new_valueinfo_name(), TensorProto.FLOAT, quantnode_output_shape
        )
        model.graph.value_info.append(z_value)
        model.set_tensor_datatype(z_value.name, stationary_input_dtype)
        model.set_initializer(z_value.name, zeropt_tensor)

        bitwidth_tensor = np.array(bitwidth_value).astype(np.float32)
        b_value = onnx.helper.make_tensor_value_info(model.make_new_valueinfo_name(), TensorProto.FLOAT, [1])
        model.graph.value_info.append(b_value)
        model.set_tensor_datatype(b_value.name, stationary_input_dtype)
        model.set_initializer(b_value.name, bitwidth_tensor)

        quant_node = onnx.helper.make_node(
            "Quant",
            inputs=[quantnode_input, s_value.name, z_value.name, b_value.name],
            outputs=[quant_tensor.name],
            name="Quant_node_" + str(node_count) + str(tensor_count),
            narrow=narrow,
            signed=signed,
            rounding_mode=rounding_mode,
        )

        return quant_node, quant_tensor

    def adjust_graph(self, model, input_positions, node_in_focus, quantized_nodes, node_count):
        tensor_count = 0
        for pos in input_positions:
            node_details = (node_in_focus.name, pos[0])
            if (
                node_details not in quantized_nodes
            ):  # This is to ensure that we don't quantize the same node for the same input/output index.
                if pos[0][0] == "input":
                    input_to_quantnode = node_in_focus.input[pos[0][1]]
                    consumer_node = node_in_focus
                    producer_node = model.find_producer(input_to_quantnode)
                    if producer_node is None or producer_node.op_type != "Quant":
                        quantization_to_perform = "yes"
                    else:
                        quantization_to_perform = "no"
                else:
                    input_to_quantnode = node_in_focus.output[pos[0][1]]
                    consumer_node = model.find_consumer(input_to_quantnode)
                    producer_node = model.find_producer(input_to_quantnode)
                    if consumer_node is None or consumer_node.op_type != "Quant":
                        quantization_to_perform = "yes"
                    else:
                        quantization_to_perform = "no"
                if quantization_to_perform == "yes":
                    node_indx = self.get_node_id(model)  # Getting index of each node in the graph.
                    quantnode_output_shape = model.get_tensor_shape(input_to_quantnode)  # Step: 3

                    quant_node, quant_tensor = self.create_node(
                        model,
                        input_to_quantnode,
                        quantnode_output_shape,
                        node_count,
                        tensor_count,
                        scale_value=pos[1][0],
                        zeropoint_value=pos[1][1],
                        bitwidth_value=pos[1][2],
                        narrow=pos[1][3],
                        signed=pos[1][4],
                        rounding_mode=pos[1][5],
                    )

                    if consumer_node is not None:
                        node_pos = node_indx[consumer_node.name]
                        model.graph.node[node_pos].input[pos[0][1]] = quant_tensor.name
                        model.graph.node.append(quant_node)
                    else:
                        model.graph.value_info.remove(quant_tensor)
                        model.graph.node.append(quant_node)
                        model.graph.output.insert(0, quant_tensor)
                        model.graph.output.pop(1)

                    model = model.transform(SortGraph())
                    tensor_count += 1
                    quantized_nodes.append(node_details)
                else:
                    print(f"{pos[0][0]} index {pos[0][1]} of {node_in_focus.name} is already quantized.")
            else:
                print(f"{pos[0][0]} index {pos[0][1]} of {node_in_focus.name} is already quantized.")
                continue

        return model


class IntroduceQuantnode(Transformation):
    """This transformation can be used to introduce a Quant node for a specific type of node in the graph.
    Users would be able to specify the location of the quant node by providing the input and output indexs
    as the parameters.

             1) Expectations:
                a) Onnx model in the modelwraper format.
                b) Model must be cleaned using cleanup_model qonnx.util.cleanup.cleanup_model()
                c) Batchsize to be set.

            2) Steps to transform are
                Step1: Finding the input for the quant node.
                Step2: Finding the consumer of the quant node output.
                Step3: Finding the shape for the output tensor of quant node.
                Note: The output tensor of the quant node must have the same shape as the
                      consumer of the input to the quant node.

            3) Introduction to quantnodes will be done with precedence to "Name" in comparison to "op_type".

            4) Assert:
                a) The input is a dictionary representing the node names as keys and a list of quant positions
                   as values.
                b) The input dictionary must have atleast one mac node (Conv, gemm, matmul) for the transformation.

            5) Return:
                Returns a cleaned version of the model.

    """

    def __init__(self, quant_node_inputs):
        super().__init__()
        self.quant_node_inputs = quant_node_inputs
        self.graph_util = graph_util()

    def apply(self, model):
        model = model.transform(InferShapes())
        if type(self.quant_node_inputs) == dict:
            selection_type = self.quant_node_inputs.keys()
            if set(selection_type) <= {"name", "op_type"}:
                node_count = 0
                quantized_nodes = []
                if "name" in selection_type:
                    by_name = self.quant_node_inputs[
                        "name"
                    ]  # by_name is a dictionary with the unique node names as keys and the list of positions as values.
                    node_list_by_name = by_name.keys()  # name of all the nodes specified by the user for an quant node.
                    for node_name in node_list_by_name:
                        node_in_focus = self.graph_util.node_from_name(model, node_name)
                        input_positions = by_name[
                            node_name
                        ]  # input positions specified by the user to introduce quant node.
                        model = self.graph_util.adjust_graph(
                            model, input_positions, node_in_focus, quantized_nodes, node_count
                        )
                        node_count += 1
                if "op_type" in selection_type:
                    by_op_type = self.quant_node_inputs[
                        "op_type"
                    ]  # by_name is a dictionary with the unique node names as keys and the list of positions as values.
                    op_list = by_op_type.keys()
                    for op in op_list:
                        node_list = self.graph_util.identify_nodes(
                            model, op
                        )  # List of all nodes with the operation type "op".
                        input_positions = by_op_type[op]
                        for node_in_focus in node_list:
                            model = self.graph_util.adjust_graph(
                                model, input_positions, node_in_focus, quantized_nodes, node_count
                            )
                            node_count += 1
                model = qonnx_make_model(model.graph)
                model = ModelWrapper(model)
                model = cleanup_model(model)
            else:
                raise Exception("Unsupported selection type")
        else:
            raise TypeError("Input must be a dictionary.")

        graph_modified = False

        return (model, graph_modified)
