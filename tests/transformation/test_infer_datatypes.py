# Copyright (c) 2020, Xilinx
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
# * Neither the name of QONNX nor the names of its
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

from pkgutil import get_data

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes, infer_mac_result_dtype
from qonnx.transformation.infer_shapes import InferShapes


def test_infer_mac_dtype_result():
    # dtype prototypes
    is32 = DataType["INT32"]
    iu32 = DataType["UINT32"]
    f32 = DataType["FLOAT32"]
    is4 = DataType["INT4"]
    iu4 = DataType["UINT4"]
    fx4 = DataType["FIXED<4,2>"]
    si4 = DataType["SCALEDINT<4>"]
    si32 = DataType["SCALEDINT<32>"]
    # test several 2-input (e.g. weights, inputs) cases
    assert infer_mac_result_dtype([iu4, iu4], False) == iu32
    assert infer_mac_result_dtype([iu4, is4], False) == is32
    assert infer_mac_result_dtype([iu4, iu4], True) == is32
    assert infer_mac_result_dtype([iu4, fx4], False) == si32
    assert infer_mac_result_dtype([fx4, si4], False) == si32
    assert infer_mac_result_dtype([is4, si4], False) == si32
    assert infer_mac_result_dtype([f32, iu4], False) == f32
    assert infer_mac_result_dtype([f32, si4], False) == f32
    # test several 3-input (e.g. weights, inputs, biases) cases
    assert infer_mac_result_dtype([iu4, iu4, iu4], False) == iu32
    assert infer_mac_result_dtype([iu4, iu4, is4], False) == is32
    assert infer_mac_result_dtype([is4, iu4, fx4], False) == si32
    assert infer_mac_result_dtype([is4, iu4, f32], False) == f32


def test_infer_datatypes():
    raw_m = get_data("qonnx.data", "onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    # this model has no DataType info, so add some DataType annotation
    # to make things a bit more exciting
    model.set_tensor_datatype("global_in", DataType["UINT8"])
    # Conv with int weights + inputs will have int output datatype
    model.set_tensor_datatype("Conv_0_param0", DataType["INT4"])
    model = model.transform(InferDataTypes())
    assert model.get_tensor_datatype("global_in") == DataType["UINT8"]
    assert model.get_tensor_datatype("Conv_0_out0") == DataType["INT32"]
    assert model.get_tensor_datatype("Relu_0_out0") == DataType["FLOAT32"]
    assert model.get_tensor_datatype("global_out") == DataType["FLOAT32"]
