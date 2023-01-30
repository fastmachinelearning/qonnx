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

import pytest

import numpy as np
from onnx import TensorProto
from onnx import helper as oh

import qonnx.core.onnx_exec as oxe
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.create_generic_partitions import PartitionFromDict
from qonnx.transformation.extend_partition import ExtendPartition
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model


def create_model():
    MultiThreshold0_node = oh.make_node(
        "MultiThreshold",
        inputs=["in1_multithreshold0", "in2_multithreshold0"],
        outputs=["out_multithreshold0"],
        name="MultiThreshold0",
        domain="qonnx.custom_op.general",
        out_dtype="UINT4",
    )

    Conv0_node = oh.make_node(
        "Conv",
        inputs=["out_multithreshold0", "in2_conv0"],
        outputs=["out_conv0"],
        name="Conv0",
        dilations=[1, 1],
        group=1,
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )

    Conv1_node = oh.make_node(
        "Conv",
        inputs=["out_multithreshold0", "in2_conv1"],
        outputs=["out_conv1"],
        name="Conv1",
        dilations=[1, 1],
        group=1,
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )

    MultiThreshold1_node = oh.make_node(
        "MultiThreshold",
        inputs=["out_conv0", "in2_multithreshold1"],
        outputs=["out_multithreshold1"],
        name="MultiThreshold1",
        domain="qonnx.custom_op.general",
        out_dtype="UINT4",
    )

    MultiThreshold2_node = oh.make_node(
        "MultiThreshold",
        inputs=["out_conv1", "in2_multithreshold2"],
        outputs=["out_multithreshold2"],
        name="MultiThreshold2",
        domain="qonnx.custom_op.general",
        out_dtype="UINT4",
    )

    Add0_node = oh.make_node(
        "Add",
        inputs=["out_multithreshold1", "out_multithreshold2"],
        outputs=["out_add0"],
        name="Add0",
    )

    MultiThreshold3_node = oh.make_node(
        "MultiThreshold",
        inputs=["out_add0", "in2_multithreshold3"],
        outputs=["out_multithreshold3"],
        name="MultiThreshold3",
        domain="qonnx.custom_op.general",
        out_dtype="UINT4",
    )

    Conv2_node = oh.make_node(
        "Conv",
        inputs=["out_multithreshold3", "in2_conv2"],
        outputs=["out_conv2"],
        name="Conv2",
        dilations=[1, 1],
        group=1,
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )

    Conv3_node = oh.make_node(
        "Conv",
        inputs=["out_multithreshold3", "in2_conv3"],
        outputs=["out_conv3"],
        name="Conv3",
        dilations=[1, 1],
        group=1,
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )

    MultiThreshold4_node = oh.make_node(
        "MultiThreshold",
        inputs=["out_conv2", "in2_multithreshold4"],
        outputs=["out_multithreshold4"],
        name="MultiThreshold4",
        domain="qonnx.custom_op.general",
        out_dtype="UINT4",
    )

    MultiThreshold5_node = oh.make_node(
        "MultiThreshold",
        inputs=["out_conv3", "in2_multithreshold5"],
        outputs=["out_multithreshold5"],
        name="MultiThreshold5",
        domain="qonnx.custom_op.general",
        out_dtype="UINT4",
    )

    Add1_node = oh.make_node(
        "Add",
        inputs=["out_multithreshold4", "out_multithreshold5"],
        outputs=["out_add1"],
        name="Add1",
    )

    # Inputs/outputs (global)
    t_type = TensorProto.FLOAT
    t_shape = [1, 256, 128, 1]
    in1_multithreshold0 = oh.make_tensor_value_info("in1_multithreshold0", t_type, t_shape)
    out_add1 = oh.make_tensor_value_info("out_add1", t_type, t_shape)

    # Initializers
    in2_multithreshold0 = oh.make_tensor_value_info("in2_multithreshold0", t_type, [256, 15])
    in2_conv0 = oh.make_tensor_value_info("in2_conv0", t_type, [256, 256, 1, 1])
    in2_conv1 = oh.make_tensor_value_info("in2_conv1", t_type, [256, 256, 1, 1])
    in2_multithreshold1 = oh.make_tensor_value_info("in2_multithreshold1", t_type, [256, 15])
    in2_multithreshold2 = oh.make_tensor_value_info("in2_multithreshold2", t_type, [256, 15])
    in2_multithreshold3 = oh.make_tensor_value_info("in2_multithreshold3", t_type, [256, 15])
    in2_conv2 = oh.make_tensor_value_info("in2_conv2", t_type, [256, 256, 1, 1])
    in2_conv3 = oh.make_tensor_value_info("in2_conv3", t_type, [256, 256, 1, 1])
    in2_multithreshold4 = oh.make_tensor_value_info("in2_multithreshold4", t_type, [256, 15])
    in2_multithreshold5 = oh.make_tensor_value_info("in2_multithreshold5", t_type, [256, 15])

    # Value_infos
    out_multithreshold0 = oh.make_tensor_value_info("out_multithreshold0", t_type, t_shape)
    out_conv0 = oh.make_tensor_value_info("out_conv0", t_type, t_shape)
    out_conv1 = oh.make_tensor_value_info("out_conv1", t_type, t_shape)
    out_multithreshold1 = oh.make_tensor_value_info("out_multithreshold1", t_type, t_shape)
    out_multithreshold2 = oh.make_tensor_value_info("out_multithreshold2", t_type, t_shape)
    out_add0 = oh.make_tensor_value_info("out_add0", t_type, t_shape)
    out_multithreshold3 = oh.make_tensor_value_info("out_multithreshold3", t_type, t_shape)
    out_conv2 = oh.make_tensor_value_info("out_conv2", t_type, t_shape)
    out_conv3 = oh.make_tensor_value_info("out_conv3", t_type, t_shape)
    out_multithreshold4 = oh.make_tensor_value_info("out_multithreshold4", t_type, t_shape)
    out_multithreshold5 = oh.make_tensor_value_info("out_multithreshold5", t_type, t_shape)

    graph = oh.make_graph(
        nodes=[
            MultiThreshold0_node,
            Conv0_node,
            Conv1_node,
            MultiThreshold1_node,
            MultiThreshold2_node,
            Add0_node,
            MultiThreshold3_node,
            Conv2_node,
            Conv3_node,
            MultiThreshold4_node,
            MultiThreshold5_node,
            Add1_node,
        ],
        name="test_graph",
        inputs=[in1_multithreshold0],
        outputs=[out_add1],
        value_info=[
            in2_multithreshold0,
            in2_conv0,
            in2_conv1,
            in2_multithreshold1,
            in2_multithreshold2,
            in2_multithreshold3,
            in2_conv2,
            in2_conv3,
            in2_multithreshold4,
            in2_multithreshold5,
            out_multithreshold0,
            out_conv0,
            out_conv1,
            out_multithreshold1,
            out_multithreshold2,
            out_add0,
            out_multithreshold3,
            out_conv2,
            out_conv3,
            out_multithreshold4,
            out_multithreshold5,
        ],
    )

    onnx_model = qonnx_make_model(graph, producer_name="test_model")
    model = ModelWrapper(onnx_model)

    mt_weights = np.random.randint(low=-1000, high=1000, size=[6, 256, 15])
    mt_weights = np.sort(mt_weights, 2)
    for i in range(0, 6):
        model.set_initializer("in2_multithreshold" + str(i), mt_weights[i])

    conv_weights = np.random.randint(low=-8, high=7, size=[4, 256, 256, 1, 1]).astype(np.float32)
    for i in range(0, 4):
        model.set_initializer("in2_conv" + str(i), conv_weights[i])
        model.set_tensor_datatype("in2_conv" + str(i), DataType["INT4"])

    return model


# Partitioning
@pytest.mark.parametrize("p", [0, 1, 2])
# Extending
@pytest.mark.parametrize("extend_id", [[0], [1], [0, 1]])
def test_extend_partition(p, extend_id):
    if p == 0:
        if extend_id != [0]:
            pytest.skip("Only the first partition node can be extended")
    if p == 1:
        if extend_id != [1]:
            pytest.skip("Only the second partition node can be extended")
        else:
            extend_id = [6]  # The 6th node is the index of the GenericPartition
            # node, so we set the index to the right value

    model = create_model()

    # Partition the model first
    partitionings = [
        {0: range(0, 6)},
        {0: range(6, 12)},
        {0: range(0, 6), 1: range(6, 12)},
    ]
    partitioning = partitionings[p]

    model = model.transform(PartitionFromDict(partitioning))

    # Create input data
    input0_tensor_name = model.graph.input[0].name

    input_shape = model.get_tensor_shape(input0_tensor_name)
    input_dtype = model.get_tensor_datatype(input0_tensor_name)
    input_val = gen_finn_dt_tensor(input_dtype, input_shape)
    input_dict = {}
    input_dict[input0_tensor_name] = input_val

    # Extend the model
    model_extended = model.transform(ExtendPartition(extend_id))

    assert oxe.compare_execution(model, model_extended, input_dict)

    # Check if FINN data_types are retained
    for n in model_extended.graph.node:
        if n.op_type == "Conv":
            assert model_extended.get_tensor_datatype(n.input[1]) == DataType["INT4"]
