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
from pkgutil import get_data

import qonnx.core.onnx_exec as oxe
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.util.basic import gen_finn_dt_tensor


def test_lower_and_channelslast_eltwiseops():
    raw_m = get_data("qonnx.data", "onnx/eltwise_chanlast_testcase.onnx")
    model = ModelWrapper(raw_m)
    iname = model.graph.input[0].name
    idt = model.get_tensor_datatype(iname)
    ishape = model.get_tensor_shape(iname)
    idict = {iname: gen_finn_dt_tensor(idt, ishape)}
    oname = model.graph.output[0].name
    expected_out = oxe.execute_onnx(model, idict)[oname]
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(ConvertToChannelsLastAndClean(make_input_channels_last=False))
    expected_ops = ["Transpose", "Im2Col", "MatMul", "Mul", "Add", "Relu", "Mul", "Quant", "Transpose"]
    ops = [x.op_type for x in model.graph.node]
    assert ops == expected_ops, "Did not found expected op sequence after lowering and channels-last"
    out = oxe.execute_onnx(model, idict)[oname]
    assert np.isclose(expected_out, out, atol=1e-4).all()
