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

import json
import numpy as np
import onnx
import onnx.parser as oprs
import os
from pkgutil import get_data

import qonnx.core.onnx_exec as oxe
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import ApplyConfig, ConvertDivToMul, GiveUniqueNodeNames, GiveUniqueParameterTensors
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model


def test_mul_to_div():
    shp = (1, 2, 4)
    dt0 = DataType["UINT8"]
    np.random.seed(0)
    shp_str = str(list(shp))

    input = f"""
    <
        ir_version: 7,
        opset_import: ["" : 9]
    >
    agraph (float{shp_str} in0) => (float{shp_str} out0)
    <
        float param_div = {{2.0}}
    >
    {{
        out0 = Div(in0, param_div)
    }}
    """
    model = oprs.parse_model(input)
    model = ModelWrapper(model)
    model.set_tensor_datatype("in0", dt0)
    model = model.transform(InferShapes())
    inp = gen_finn_dt_tensor(dt0, shp)
    input_dict = {"in0": inp}
    out_expected = oxe.execute_onnx(model, input_dict)["out0"]
    new_model = model.transform(ConvertDivToMul())
    out_produced = oxe.execute_onnx(new_model, input_dict)["out0"]
    assert (out_expected == out_produced).all()


def test_give_unique_node_names():
    raw_m = get_data("qonnx.data", "onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    model = model.transform(GiveUniqueNodeNames())
    assert model.graph.node[0].name == "Reshape_0"
    assert model.graph.node[1].name == "Conv_0"
    assert model.graph.node[11].name == "Add_2"


def test_give_unique_node_names_missingshape():
    # see https://github.com/fastmachinelearning/qonnx/issues/33
    Add1_node = onnx.helper.make_node("Add", inputs=["in1", "Add_1_param0"], outputs=["sum1"], name="Add_1")

    Add0_node = onnx.helper.make_node(
        "Add",
        inputs=["sum1", "Add_0_param0"],
        outputs=["sum2"],
        name="Add_0",
    )

    Add2_node = onnx.helper.make_node(
        "Add",
        inputs=["abs1", "abs1"],
        outputs=["sum3"],
        name="Add_2",
    )

    Abs_node = onnx.helper.make_node("Abs", inputs=["sum2"], outputs=["abs1"], name="Abs")

    Round_node = onnx.helper.make_node(
        "Round",
        inputs=["sum3"],
        outputs=["out1"],
        name="Round",
    )

    in1 = onnx.helper.make_tensor_value_info("in1", onnx.TensorProto.FLOAT, [4, 4])
    out1 = onnx.helper.make_tensor_value_info("out1", onnx.TensorProto.FLOAT, [4, 4])

    graph = onnx.helper.make_graph(
        nodes=[
            Add1_node,
            Add0_node,
            Abs_node,
            Add2_node,
            Round_node,
        ],
        name="simple_graph",
        inputs=[in1],
        outputs=[out1],
        value_info=[
            # note: value_info for sum1 explicitly omitted to trigger bug #33
            # onnx.helper.make_tensor_value_info("sum1", onnx.TensorProto.FLOAT, [4, 4]),
            onnx.helper.make_tensor_value_info("sum2", onnx.TensorProto.FLOAT, [4, 4]),
            onnx.helper.make_tensor_value_info("abs1", onnx.TensorProto.FLOAT, [4, 4]),
            onnx.helper.make_tensor_value_info("sum3", onnx.TensorProto.FLOAT, [4, 4]),
        ],
        initializer=[
            onnx.helper.make_tensor("Add_1_param0", onnx.TensorProto.FLOAT, [4, 4], np.zeros(16).tolist()),
            onnx.helper.make_tensor("Add_0_param0", onnx.TensorProto.FLOAT, [4, 4], np.zeros(16).tolist()),
        ],
    )

    onnx_model = onnx.helper.make_model(graph, producer_name="simple-model")
    model = ModelWrapper(onnx_model)
    # sum1 is missing shape so should return False
    assert model.check_all_tensor_shapes_specified() is False
    model = model.transform(GiveUniqueNodeNames())
    assert [x.name for x in model.get_nodes_by_op_type("Add")] == ["Add_0", "Add_1", "Add_2"]
    assert [x.name for x in model.get_nodes_by_op_type("Abs")] == ["Abs_0"]
    assert [x.name for x in model.get_nodes_by_op_type("Round")] == ["Round_0"]


def test_give_unique_parameter_tensors():
    # Create model
    input_shape = [4, 4]
    in1 = onnx.helper.make_tensor_value_info("in1", onnx.TensorProto.FLOAT, input_shape)
    out1 = onnx.helper.make_tensor_value_info("out1", onnx.TensorProto.FLOAT, input_shape)

    graph_nodes = []
    graph_nodes += [onnx.helper.make_node("Add", inputs=["in1", "param1"], outputs=["t1"])]

    graph_nodes += [onnx.helper.make_node("Sum", inputs=["t1", "param1", "param1"], outputs=["t2"])]

    graph_nodes += [onnx.helper.make_node("Sum", inputs=["t2", "param2", "param1"], outputs=["t3"])]

    graph_nodes += [onnx.helper.make_node("Add", inputs=["t3", "param1"], outputs=["out1"])]

    onnx_graph = onnx.helper.make_graph(
        nodes=graph_nodes,
        name="simple_graph",
        inputs=[in1],
        outputs=[out1],
    )

    onnx_model = qonnx_make_model(onnx_graph, producer_name="simple-model")
    model = ModelWrapper(onnx_model)

    # Set param values
    np.random.seed(0)
    param1 = np.random.rand(*input_shape).astype(np.float32)
    param2 = np.random.rand(*input_shape).astype(np.float32)
    model.set_initializer("param1", param1)
    model.set_initializer("param2", param2)
    model = model.transform(InferShapes())

    # Apply transformation
    new_model = model.transform(GiveUniqueParameterTensors())
    new_model = new_model.transform(InferShapes())

    # Test
    # Breaks the model?
    input_tensor = np.random.rand(*input_shape).astype(np.float32)
    input_dict = {"in1": input_tensor}

    # run original
    expected_context = oxe.execute_onnx(model, input_dict)
    expected_output = expected_context[model.graph.output[0].name]

    # run modified
    produced_context = oxe.execute_onnx(new_model, input_dict)
    produced_output = produced_context[new_model.graph.output[0].name]

    assert np.isclose(
        expected_output, produced_output, atol=1e-8
    ).all(), " GiveUniqueParameterTensors() transform breaks the model"

    # Does the job?
    param_set = set()
    param_cnt = 0
    for n in new_model.graph.node:
        for i in range(1, len(n.input)):
            param_set |= {n.input[i]}
            param_cnt += 1

    assert len(param_set) == param_cnt, " There are still parameters reused"


def test_apply_config():
    raw_m = get_data("qonnx.data", "onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(GiveUniqueNodeNames())
    # set up a config in a dict, then dump it to JSON
    config = {}
    config["Defaults"] = {"kernel_size": [[3, 3], ["Im2Col"]]}
    config["Im2Col_0"] = {"kernel_size": [7, 7]}
    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)
    model = model.transform(ApplyConfig("config.json"))
    # check model
    assert getCustomOp(model.graph.node[2]).get_nodeattr("kernel_size") == [7, 7]
    assert getCustomOp(model.graph.node[9]).get_nodeattr("kernel_size") == [3, 3]
    os.remove("config.json")
