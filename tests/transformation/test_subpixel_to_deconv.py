# Copyright (c) 2023, Advanced Micro Devices, Inc.
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

import numpy as np
from pkgutil import get_data

import qonnx.core.onnx_exec as oxe
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.subpixel_to_deconv import SubPixelToDeconvolution
from qonnx.util.basic import gen_finn_dt_tensor

np.random.seed(0)


def test_subpixel_to_deconv_espcn():
    raw_m = get_data("qonnx.data", "onnx/bsd300x3-espcn/model.onnx")
    model = ModelWrapper(raw_m)
    model = model.transform(InferShapes())
    iname = model.graph.input[0].name
    oname = model.graph.output[0].name
    ishape = model.get_tensor_shape(iname)
    rand_inp = gen_finn_dt_tensor(DataType["FLOAT32"], ishape)
    input_dict = {iname: rand_inp}
    expected = oxe.execute_onnx(model, input_dict)[oname]
    new_model = model.transform(SubPixelToDeconvolution())
    # check that there are no DepthToSpace ops left
    op_types = list(map(lambda x: x.op_type, new_model.graph.node))
    assert "DepthToSpace" not in op_types
    produced = oxe.execute_onnx(new_model, input_dict)[oname]
    assert np.isclose(expected, produced, atol=1e-4).all()
