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

from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.range_analysis import (
    RangeInfo,
    calc_matmul_node_range,
    calc_matmul_range,
    calc_monotonic_range,
    promote_range_shape,
)
from qonnx.util.test import download_model


def test_promote_range_shape():
    tshape = (2, 2)
    tmin = 0
    tmax = 1
    trange_minimal = (tmin, tmax)
    trange_full = (np.zeros(tshape), np.ones(tshape))
    ret = promote_range_shape(trange_minimal, tshape)
    assert (ret[0] == trange_full[0]).all()
    assert (ret[1] == trange_full[1]).all()
    ret = promote_range_shape(trange_full, tshape)
    assert (ret[0] == trange_full[0]).all()
    assert (ret[1] == trange_full[1]).all()


def test_calc_monotonic_range():
    model_name = "MobileNetv1-w4a4"
    model = download_model(model_name, return_modelwrapper=True, do_cleanup=True)
    relu_node = model.get_nodes_by_op_type("Relu")[0]
    relu_in = relu_node.input[0]
    relu_out = relu_node.output[0]
    relu_in_shape = tuple(model.get_tensor_shape(relu_in))
    relu_in_vi = model.get_tensor_valueinfo(relu_in)
    relu_in_range = RangeInfo(range=promote_range_shape((-1.0, 1.0), relu_in_vi))
    range_dict = {relu_in: relu_in_range, relu_out: RangeInfo()}
    calc_monotonic_range(relu_node, model, range_dict)
    assert range_dict[relu_out].range[0].shape == relu_in_shape
    assert range_dict[relu_out].range[1].shape == relu_in_shape
    assert (range_dict[relu_out].range[0] == 0).all()
    assert (range_dict[relu_out].range[1] == 1).all()
    qnt_node = model.find_consumer(relu_out)
    qnt_out = qnt_node.output[0]
    range_dict[qnt_out] = RangeInfo()
    calc_monotonic_range(qnt_node, model, range_dict)
    assert range_dict[qnt_out].range[0].shape == relu_in_shape
    assert range_dict[qnt_out].range[1].shape == relu_in_shape
    assert (range_dict[qnt_out].range[0] == 0).all()
    ctx = {k: model.get_initializer(k) for k in qnt_node.input}
    ctx[relu_out] = range_dict[relu_out].range[1]
    getCustomOp(qnt_node).execute_node(ctx, model.graph)
    assert (range_dict[qnt_out].range[1] == ctx[qnt_out]).all()


@pytest.mark.parametrize("shapes", [((14, 14, 200), (200, 16)), ((12, 128, 32), (12, 32, 128))])
@pytest.mark.parametrize("A_dt", [DataType["UINT4"]])
@pytest.mark.parametrize("B_dt", [DataType["INT4"]])
def test_calc_matmul_range(shapes, A_dt, B_dt):
    A_shape, B_shape = shapes
    min_A = A_dt.min() * np.ones(A_shape)
    max_A = A_dt.max() * np.ones(A_shape)
    range_A = (min_A, max_A)
    B = np.random.randint(B_dt.min(), B_dt.max(), B_shape)
    range_B = (B, B)
    range_C = calc_matmul_range(range_A, range_B)
    A = np.random.randint(A_dt.min(), A_dt.max(), A_shape)
    C = np.matmul(A, B)
    assert (range_C[0] <= C).all()
    assert (C <= range_C[1]).all()


def test_calc_matmul_node_range():
    model_name = "FINN-TFC_W2A2"
    model = download_model(model_name, return_modelwrapper=True, do_cleanup=True)
    matmul_node = model.get_nodes_by_op_type("MatMul")[0]
    quant_in_node = model.get_node_from_name("Quant_4")
    quant_in_vi = model.get_tensor_valueinfo(quant_in_node.input[0])
    quant_act_range = RangeInfo(range=promote_range_shape((-1.0, 1.0), quant_in_vi))
    range_dict = {quant_in_node.input[0]: quant_act_range, quant_in_node.output[0]: RangeInfo()}
    calc_monotonic_range(quant_in_node, model, range_dict)
    quant_w_node = model.get_node_from_name("Quant_0")
    range_dict[quant_w_node.output[0]] = RangeInfo()
    calc_monotonic_range(quant_w_node, model, range_dict)
    range_dict[matmul_node.output[0]] = RangeInfo()
    calc_matmul_node_range(matmul_node, model, range_dict)
    assert range_dict[matmul_node.output[0]].range[0][0][0] == -233
    assert range_dict[matmul_node.output[0]].range[1][0][0] == 233
    assert range_dict[matmul_node.output[0]].range[0][0][1] == -288
    assert range_dict[matmul_node.output[0]].range[1][0][1] == 288
    assert range_dict[matmul_node.output[0]].range[0][0][-1] == -190
    assert range_dict[matmul_node.output[0]].range[1][0][-1] == 190
