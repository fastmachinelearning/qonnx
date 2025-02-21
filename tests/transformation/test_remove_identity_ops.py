# Copyright (c) 2023 Advanced Micro Devices, Inc.
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

import pytest

import numpy as np
from onnx import TensorProto, helper

import qonnx.core.onnx_exec as oxe
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import SortGraph
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model


def insert_identity_op(model, op, as_first_node, approx, fork_after_id):
    kwargs = {}
    inp_ndims = 4 if as_first_node else 2
    if approx:
        zero_val = 0.000001
        one_val = 0.999999
    else:
        zero_val = 0.0
        one_val = 1.0
    if op in ["Add", "Sub"]:
        val = np.asarray([zero_val], dtype=np.float32)
    elif op in ["Mul", "Div"]:
        val = np.asarray([one_val], dtype=np.float32)
    elif op in ["Identity"]:
        val = None
    elif op == "Pad":
        # opset 11 and above: padding specified as input and not attribute
        val = np.asarray([0] * 2 * inp_ndims, dtype=np.int64)
    elif op == "Dropout":
        val = None
        kwargs = {"ratio": 0.0}
    else:
        return

    graph = model.graph
    if val is None:
        inplist = ["inp" if as_first_node else "div_out"]
    else:
        model.set_initializer("value", val)
        inplist = ["inp" if as_first_node else "div_out", "value"]
    identity_node = helper.make_node(op, inplist, ["ident_out"], **kwargs)
    old_2nd_node = graph.node[1]
    old_last_node = graph.node[-1]
    graph.node.append(identity_node)
    if fork_after_id:
        graph.node.append(helper.make_node("Mul", ["ident_out", "mul2"], ["mulbranch0_out"]))
        model.set_initializer("mul2", np.asarray([2.0], dtype=np.float32))
        graph.node.append(helper.make_node("Mul", ["ident_out", "mul3"], ["mulbranch1_out"]))
        model.set_initializer("mul3", np.asarray([3.0], dtype=np.float32))
        graph.node.append(helper.make_node("Add", ["mulbranch0_out", "mulbranch1_out"], ["idfork_out"]))
        subgraph_out = "idfork_out"
    else:
        subgraph_out = "ident_out"

    if as_first_node:
        old_2nd_node.input[0] = subgraph_out
    else:
        old_last_node.input[0] = subgraph_out
    model = model.transform(SortGraph())

    return model


# identity operations to be inserted
@pytest.mark.parametrize("op", ["Add", "Sub", "Mul", "Div", "Identity", "Pad", "Dropout"])
@pytest.mark.parametrize("approx", [False, True])
@pytest.mark.parametrize("as_first_node", [False, True])
@pytest.mark.parametrize("fork_before_id", [False, True])
@pytest.mark.parametrize("fork_after_id", [False, True])
def test_remove_identity_ops(op, as_first_node, approx, fork_before_id, fork_after_id):
    if approx and not (op in ["Add", "Sub", "Mul", "Div"]):
        pytest.skip(f"approx=True not relevant for {op}")
    # set up onnx model
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4, 1, 1])
    mul = helper.make_tensor_value_info("mul", TensorProto.FLOAT, [])
    shape = helper.make_tensor_value_info("shape", TensorProto.FLOAT, [2])
    div = helper.make_tensor_value_info("div", TensorProto.FLOAT, [])
    matmul = helper.make_tensor_value_info("matmul", TensorProto.FLOAT, [4, 2])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 2])

    mul_node = helper.make_node("Mul", ["inp", "mul"], ["mul_out"])
    reshape_node = helper.make_node("Reshape", ["mul_out", "shape"], ["reshape_out"])
    div_node = helper.make_node("Div", ["reshape_out", "div"], ["div_out"])
    matmul_node = helper.make_node("MatMul", ["div_out", "matmul"], ["outp"])

    graph = helper.make_graph(
        nodes=[mul_node, reshape_node, div_node, matmul_node],
        name="identity-graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[mul, shape, div, matmul],
    )

    model = qonnx_make_model(graph, producer_name="mulpastconv-model")
    model = ModelWrapper(model)
    inp_values = gen_finn_dt_tensor(DataType["INT2"], [1, 4, 1, 1])
    mul_values = np.random.uniform(low=0.1, high=0.99, size=(1)).astype(np.float32)
    shape_values = np.asarray([1, -1], dtype=np.int64)
    div_values = np.random.uniform(low=0.1, high=0.99, size=(1)).astype(np.float32)
    matmul_values = gen_finn_dt_tensor(DataType["INT2"], [4, 2])
    model.set_initializer("mul", mul_values)
    model.set_initializer("shape", shape_values)
    model.set_initializer("div", div_values)
    model.set_initializer("matmul", matmul_values)
    insert_identity_op(model, op, as_first_node, approx, fork_after_id)
    model = model.transform(InferShapes())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    idict = {"inp": inp_values}
    odict_before = oxe.execute_onnx(model, idict)
    num_of_nodes_before = len(model.graph.node)
    if fork_before_id and not as_first_node:
        divout_vi = model.get_tensor_valueinfo("div_out")
        model.graph.output.append(divout_vi)
        model.graph.value_info.remove(divout_vi)
    model = model.transform(RemoveIdentityOps())
    num_of_nodes_after = len(model.graph.node)
    assert num_of_nodes_before - 1 == num_of_nodes_after

    odict_after = oxe.execute_onnx(model, idict)
    outputs_same = [np.isclose(odict_before[tname], odict_after[tname], atol=1e-3).all() for tname in odict_before.keys()]
    assert all(outputs_same)
