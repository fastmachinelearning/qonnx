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
import onnx
from pkgutil import get_data

import qonnx.core.data_layout as DataLayout
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model


def test_modelwrapper():
    raw_m = get_data("qonnx.data", "onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    assert model.check_all_tensor_shapes_specified() is True
    inp_name = model.graph.input[0].name
    inp_shape = model.get_tensor_shape(inp_name)
    assert inp_shape == [1, 1, 28, 28]
    conv_nodes = model.get_nodes_by_op_type("Conv")
    matmul_nodes = model.get_nodes_by_op_type("MatMul")
    assert len(conv_nodes) == 2
    assert len(matmul_nodes) == 1
    first_conv = conv_nodes[0]
    first_conv_iname = first_conv.input[0]
    first_conv_wname = first_conv.input[1]
    first_conv_oname = first_conv.output[0]
    assert first_conv_iname != "" and (first_conv_iname is not None)
    assert first_conv_wname != "" and (first_conv_wname is not None)
    assert first_conv_oname != "" and (first_conv_oname is not None)
    first_conv_weights = model.get_initializer(first_conv_wname)
    assert first_conv_weights.shape == (8, 1, 5, 5)
    first_conv_weights_rand = np.random.randn(8, 1, 5, 5)
    model.set_initializer(first_conv_wname, first_conv_weights_rand)
    assert (model.get_initializer(first_conv_wname) == first_conv_weights_rand).all()
    inp_cons = model.find_consumer(first_conv_iname)
    assert inp_cons == first_conv
    out_prod = model.find_producer(first_conv_oname)
    assert out_prod == first_conv
    inp_layout = model.get_tensor_layout(first_conv_iname)
    assert inp_layout is None
    inp_layout = DataLayout.NCHW
    model.set_tensor_layout(first_conv_iname, inp_layout)
    assert model.get_tensor_layout(first_conv_iname) == inp_layout
    inp_sparsity = model.get_tensor_sparsity(first_conv_iname)
    assert inp_sparsity is None
    inp_sparsity = {"dw": {"kernel_shape": [3, 3]}}
    model.set_tensor_sparsity(first_conv_iname, inp_sparsity)
    assert model.get_tensor_sparsity(first_conv_iname) == inp_sparsity


def test_modelwrapper_graph_order():
    # create small network with properties to be tested
    Neg_node = onnx.helper.make_node("Neg", inputs=["in1"], outputs=["neg1"])
    Round_node = onnx.helper.make_node("Round", inputs=["neg1"], outputs=["round1"])

    Ceil_node = onnx.helper.make_node("Ceil", inputs=["neg1"], outputs=["ceil1"])
    Add_node = onnx.helper.make_node("Add", inputs=["round1", "ceil1"], outputs=["out1"])

    in1 = onnx.helper.make_tensor_value_info("in1", onnx.TensorProto.FLOAT, [4, 4])
    out1 = onnx.helper.make_tensor_value_info("out1", onnx.TensorProto.FLOAT, [4, 4])

    graph = onnx.helper.make_graph(
        nodes=[Neg_node, Round_node, Ceil_node, Add_node],
        name="simple_graph",
        inputs=[in1],
        outputs=[out1],
        value_info=[
            onnx.helper.make_tensor_value_info("neg1", onnx.TensorProto.FLOAT, [4, 4]),
            onnx.helper.make_tensor_value_info("round1", onnx.TensorProto.FLOAT, [4, 4]),
            onnx.helper.make_tensor_value_info("ceil1", onnx.TensorProto.FLOAT, [4, 4]),
        ],
    )

    onnx_model = qonnx_make_model(graph, producer_name="simple-model")
    model = ModelWrapper(onnx_model)

    # test graph order functions
    assert model.find_consumers("in1") == [Neg_node]
    assert model.find_consumers("neg1") == [Round_node, Ceil_node]
    assert model.find_consumers("round1") == [Add_node]
    assert model.find_consumers("ceil1") == [Add_node]
    assert model.find_consumers("out1") == []

    assert model.find_direct_successors(Neg_node) == [Round_node, Ceil_node]
    assert model.find_direct_successors(Round_node) == [Add_node]
    assert model.find_direct_successors(Ceil_node) == [Add_node]
    assert model.find_direct_successors(Add_node) is None

    assert model.find_direct_predecessors(Neg_node) is None
    assert model.find_direct_predecessors(Round_node) == [Neg_node]
    assert model.find_direct_predecessors(Ceil_node) == [Neg_node]
    assert model.find_direct_predecessors(Add_node) == [Round_node, Ceil_node]

    assert model.get_node_index(Neg_node) == 0
    assert model.get_node_index(Round_node) == 1
    assert model.get_node_index(Ceil_node) == 2
    assert model.get_node_index(Add_node) == 3


def test_modelwrapper_detect_forks_n_joins():
    # create small network with properties to be tested
    Neg_node = onnx.helper.make_node("Neg", inputs=["in1"], outputs=["neg1"])
    Round_node = onnx.helper.make_node("Round", inputs=["neg1"], outputs=["round1"])

    Ceil_node = onnx.helper.make_node("Ceil", inputs=["neg1"], outputs=["ceil1"])
    Add_node = onnx.helper.make_node("Add", inputs=["round1", "ceil1"], outputs=["out1"])

    in1 = onnx.helper.make_tensor_value_info("in1", onnx.TensorProto.FLOAT, [4, 4])
    out1 = onnx.helper.make_tensor_value_info("out1", onnx.TensorProto.FLOAT, [4, 4])

    graph = onnx.helper.make_graph(
        nodes=[Neg_node, Round_node, Ceil_node, Add_node],
        name="simple_graph",
        inputs=[in1],
        outputs=[out1],
        value_info=[
            onnx.helper.make_tensor_value_info("neg1", onnx.TensorProto.FLOAT, [4, 4]),
            onnx.helper.make_tensor_value_info("round1", onnx.TensorProto.FLOAT, [4, 4]),
            onnx.helper.make_tensor_value_info("ceil1", onnx.TensorProto.FLOAT, [4, 4]),
        ],
    )

    onnx_model = qonnx_make_model(graph, producer_name="simple-model")
    model = ModelWrapper(onnx_model)

    # test
    assert model.is_fork_node(Neg_node)
    assert not model.is_fork_node(Round_node)
    assert not model.is_fork_node(Ceil_node)
    assert not model.is_fork_node(Add_node)

    assert not model.is_join_node(Neg_node)
    assert not model.is_join_node(Round_node)
    assert not model.is_join_node(Ceil_node)
    assert model.is_join_node(Add_node)


def test_modelwrapper_setting_unsetting_datatypes():
    # Set and unset some datatypes and check for expected return values
    raw_m = get_data("qonnx.data", "onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)

    test_node = model.graph.node[0]
    test_tensor = test_node.output[0]

    ret = model.get_tensor_datatype(test_tensor)
    assert ret == DataType["FLOAT32"], "Tensor datatype should be float32 for no initalization."

    model.set_tensor_datatype(test_tensor, None)
    ret = model.get_tensor_datatype(test_tensor)
    assert ret == DataType["FLOAT32"], "An unset datatype should return float32."

    model.set_tensor_datatype(test_tensor, DataType["INT3"])
    ret = model.get_tensor_datatype(test_tensor)
    assert ret == DataType["INT3"], "Tensor datatype should follow setting."

    model.set_tensor_datatype(test_tensor, DataType["UINT4"])
    ret = model.get_tensor_datatype(test_tensor)
    assert ret == DataType["UINT4"], "Tensor datatype should follow setting."

    model.set_tensor_datatype(test_tensor, None)
    ret = model.get_tensor_datatype(test_tensor)
    assert ret == DataType["FLOAT32"], "An unset datatype should return float32."

    model.set_tensor_datatype(test_tensor, DataType["BIPOLAR"])
    ret = model.get_tensor_datatype(test_tensor)
    assert ret == DataType["BIPOLAR"], "Tensor datatype should follow setting."
