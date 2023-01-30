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
import onnx.helper as helper
from onnx import TensorProto

import qonnx.core.onnx_exec as oxe
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

export_onnx_path = "test_xnorpopcountmatmul.onnx"


def test_xnorpopcountmatmul():
    M = 1
    K = 3
    N = 3
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [M, K])
    W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [K, N])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, ["x", "y"])
    node_def = helper.make_node("XnorPopcountMatMul", ["x", "W"], ["out"], domain="qonnx.custom_op.general")
    modelproto = qonnx_make_model(helper.make_graph([node_def], "test_model", [x], [out], value_info=[W]))
    model = ModelWrapper(modelproto)
    model.set_tensor_datatype("x", DataType["BINARY"])
    model.set_tensor_datatype("W", DataType["BINARY"])
    W_data = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    model.set_initializer("W", W_data)
    # test shape inference
    model = model.transform(InferShapes())
    assert model.get_tensor_shape("out") == [M, N]
    # test datatype inference
    assert model.get_tensor_datatype("out") == DataType["FLOAT32"]
    model = model.transform(InferDataTypes())
    assert model.get_tensor_datatype("out") == DataType["UINT32"]
    # test execution
    x_data = np.asarray([[1, 0, 0]], dtype=np.float32)
    inp_dict = {"x": x_data}
    out_dict = oxe.execute_onnx(model, inp_dict)
    Wb = 2 * W_data - 1
    xb = 2 * x_data - 1
    rb = np.matmul(xb, Wb)
    assert (2 * out_dict["out"] - K == rb).all()
