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

import qonnx.core.data_layout as DataLayout
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul


def test_infer_data_layouts():
    raw_m = get_data("qonnx.data", "onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataLayouts())

    assert model.get_tensor_layout("global_in") == DataLayout.NCHW
    assert model.get_tensor_layout("Conv_0_out0") == DataLayout.NCHW
    assert model.get_tensor_layout("MaxPool_0_out0") == DataLayout.NCHW
    assert model.get_tensor_layout("Reshape_0_out0") == DataLayout.NC
    assert model.get_tensor_layout("MatMul_0_out0") == DataLayout.NC
    assert model.get_tensor_layout("global_out") == DataLayout.NC

    model = model.transform(LowerConvsToMatMul())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataLayouts())

    assert model.get_tensor_layout("global_in") == DataLayout.NCHW
    assert model.get_tensor_layout("Transpose_0_out0") == DataLayout.NHWC
    assert model.get_tensor_layout("Im2Col_0_out0") == DataLayout.NHWC
    assert model.get_tensor_layout("MatMul_0_out0") == DataLayout.NHWC
    assert model.get_tensor_layout("MaxPool_0_out0") == DataLayout.NCHW
    assert model.get_tensor_layout("Reshape_0_out0") == DataLayout.NC
    assert model.get_tensor_layout("MatMul_2_out0") == DataLayout.NC
    assert model.get_tensor_layout("global_out") == DataLayout.NC
