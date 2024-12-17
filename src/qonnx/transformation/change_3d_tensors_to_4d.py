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

import numpy as np
import warnings

from qonnx.transformation.base import Transformation
from qonnx.transformation.general import RemoveUnusedTensors
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name

# FINN currently handles convolutions (when e.g lowering them
# to matmuls) with the assumption that they operate on 4D tensors
# shaped as (N,C,H,W). H/W can be 1 for convolutions on 1D data.
# This transformation converts a graph with 3D tensors to the expected
# 4D format. Note: the transformation only works for certain node types;
# see _find_invalid_nodes below.


def _find_invalid_nodes(model):
    """
    Check whether the graph contains any node types that are not supported by the
    Change3Dto4DTensors transformation.

    """
    valid_nodes = [
        "Add",
        "Mul",
        "BatchNormalization",
        "MultiThreshold",
        "Conv",
        "Transpose",
        "LogSoftmax",
        "ArgMax",
        "Div",
        "TopK",
        "MatMul",
        "Flatten",
        "Reshape",
        "MaxPool",
        "Upsample",
        "Resize",
    ]
    invalid_nodes = []
    for n in model.graph.node:
        node_op_type = n.op_type
        if node_op_type in valid_nodes:
            continue
        else:
            invalid_nodes.append(node_op_type)

    return invalid_nodes


class Change3DTo4DTensors(Transformation):
    """
    Replaces 3D tensors with 4D tensors assuming the following format:
    [N, C, H] -> [N, C, H, 1].
    The attributes of a (specific) set of supported nodes are changed accordingly.
    If the graph contains unsupported nodes, a warning is raised and the transformation
    is not applied.
    """

    def apply(self, model):
        graph_modified = False

        invalid_nodes = _find_invalid_nodes(model)
        if len(invalid_nodes) > 0:
            warnings.warn(
                "Transformation is not applied,\
                 found unsupported nodes in the graph: {}.".format(
                    invalid_nodes
                )
            )
            return (model, graph_modified)

        # Infer the shapes of each tensor, remove unused tensors
        # and give each tensor a readable name
        model = model.transform(InferShapes())
        model = model.transform(RemoveUnusedTensors())

        # This list contains all nodes with initializers that need to be converted
        nodes_with_initializers = ["Mul", "Conv", "Add", "Div", "Reshape"]
        # Obtain a list of initializer names (used to filter out only value infos)
        initializers_names = [x.name for x in model.graph.initializer]

        all_tensors = {}
        # Extract the inputs
        all_tensors = {
            **all_tensors,
            **{x.name: [x.type.tensor_type.elem_type, model.get_tensor_shape(x.name)] for x in model.graph.input},
        }
        # Extract only the output tensors
        all_tensors = {
            **all_tensors,
            **{
                x.name: [x.type.tensor_type.elem_type, model.get_tensor_shape(x.name)]
                for x in model.graph.value_info
                if x.name not in initializers_names
            },
        }
        # Extract only initializers from nodes that are relevant for conversion
        all_tensors = {
            **all_tensors,
            **{
                x.name: [x.data_type, x.dims]
                for x in model.graph.initializer
                if model.find_consumer(x.name).op_type in nodes_with_initializers
            },
        }
        # Extract the outputs
        all_tensors = {
            **all_tensors,
            **{x.name: [x.type.tensor_type.elem_type, model.get_tensor_shape(x.name)] for x in model.graph.output},
        }

        # The list below contains tensor names that are the output of nodes that
        # reduce the tensor's dimension. The shape of these tensors also needs
        # to be extended
        tensors_reduced_dimension = []
        for n in model.graph.node:
            node_op_type = n.op_type
            input_shape = model.get_tensor_shape(n.input[0])
            # Find tensors that are the output of nodes that reduce the dimension
            if node_op_type == "ArgMax":
                keep_dims = get_by_name(n.attribute, "keepdims", "name").i
                if len(input_shape) == 3 and keep_dims == 0:
                    node_out = n.output
                    for n_o in node_out:
                        tensors_reduced_dimension.append(n_o)
            # Each node from the list of supported nodes is made compatible
            # with 4D tensors
            if node_op_type == "Transpose":
                perm = get_by_name(n.attribute, "perm", "name").ints
                if len(perm) == 3:  # Meaning that the transpose operation was on a 3D tensor
                    perm.append(3)  # append 4th dimension
            elif node_op_type in ["ArgMax", "LogSoftMax", "TopK", "Flatten"]:
                axis = get_by_name(n.attribute, "axis", "name")
                if len(input_shape) == 3 and axis.i < 0:
                    axis.i = 3 + axis.i  # count dimensions from the front
            elif node_op_type == "Conv":
                dilations = get_by_name(n.attribute, "dilations", "name").ints
                kernel_shape = get_by_name(n.attribute, "kernel_shape", "name").ints
                pads = get_by_name(n.attribute, "pads", "name").ints
                strides = get_by_name(n.attribute, "strides", "name").ints
                if len(dilations) == 1:  # we must add another dimension to it
                    dilations.append(1)  # only equal dilation value along each spatial axis is supported
                if len(kernel_shape) == 1:  # we must add another dimension to it
                    kernel_shape.append(1)
                if len(pads) == 2:  # pads = [x1_begin, x1_end] --> [x1_begin, x2_begin, x1_end, x2_end]
                    pads.insert(1, 0)
                    pads.append(0)
                if len(strides) == 1:  # strides = [stride_h, stride_w]
                    strides.append(1)
            elif node_op_type == "MaxPool":
                kernel_shape = get_by_name(n.attribute, "kernel_shape", "name").ints
                pads = get_by_name(n.attribute, "pads", "name").ints
                strides = get_by_name(n.attribute, "strides", "name").ints
                if len(kernel_shape) == 1:  # we must add another dimension to it
                    kernel_shape.append(1)
                if len(pads) == 2:  # pads = [x1_begin, x1_end] --> [x1_begin, x2_begin, x1_end, x2_end]
                    pads.insert(1, 0)
                    pads.append(0)
                if len(strides) == 1:  # strides = [stride_h, stride_w]
                    strides.append(1)
            elif node_op_type == "Upsample":
                # the parameter for the Upsample layer is a 1d list of
                # upsampling factors along each axis
                scales = model.get_initializer(n.input[1])
                assert list(scales.shape) == [3]
                scales = np.append(scales, np.asarray(1.0, dtype=np.float32))
                model.set_initializer(n.input[1], scales)
            elif node_op_type == "Resize":
                assert "axes" not in [x.name for x in n.attribute], "%s: Axes attribute is not supported." % n.name
                assert not (len(n.input) in (3, 4) and model.get_initializer(n.input[1]) is not None), (
                    "%s: ROI input is not supported." % n.name
                )
                if len(n.input) == 2:
                    # Resize version 10
                    scales = model.get_initializer(n.input[1])
                    scales = np.append(scales, np.asarray(1.0, dtype=np.float32))
                    model.set_initializer(n.input[1], scales)
                elif len(n.input) == 3:
                    # Resize version 11 and up (no size input)
                    scales = model.get_initializer(n.input[2])
                    scales = np.append(scales, np.asarray(1.0, dtype=np.float32))
                    model.set_initializer(n.input[2], scales)
                elif len(n.input) == 4:
                    scales_exists = (model.get_initializer(n.input[2]) is not None) and (
                        len(model.get_initializer(n.input[2])) != 0
                    )
                    sizes_exists = (model.get_initializer(n.input[3]) is not None) and (
                        len(model.get_initializer(n.input[3])) != 0
                    )
                    assert scales_exists ^ sizes_exists, (
                        "%s: Either scales or the target output size must "
                        "be specified. Specifying both is prohibited." % n.name
                    )
                    if scales_exists:
                        # Scales parameter is a 1d list of upsampling factors along each axis
                        scales = model.get_initializer(n.input[2])
                        scales = np.append(scales, np.asarray(1.0, dtype=np.float32))
                        model.set_initializer(n.input[2], scales)
                    else:
                        # Size parameter is a 1d list of the target size along each axis
                        sizes = model.get_initializer(n.input[3])
                        sizes = np.append(sizes, np.asarray(1.0, dtype=np.int64))
                        model.set_initializer(n.input[3], sizes)
                input_shape.append(1)

        # Change format of each input/value_info/output tensor
        for k, v in all_tensors.items():
            tensor_type = v[0]
            shape = v[1]
            # Add extra dimension for tensors that either:
            # 1) Have 3 dimensions ( (N,C,H) -> (N,C,H,1) )
            # 2) Come after operations that reduce their dimension: e.g. {Argmax, ...}
            if len(shape) == 3 or k in tensors_reduced_dimension:
                shape.append(1)
                model.set_tensor_shape(k, shape, tensor_type)

        return (model, graph_modified)
