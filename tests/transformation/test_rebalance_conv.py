# Copyright (c) 2022 Xilinx, Inc.
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
import onnx.parser as oprs

import qonnx.core.onnx_exec as oxe
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.rebalance_conv import RebalanceIm2Col
from qonnx.util.basic import gen_finn_dt_tensor


def test_rebalance_conv():
    ch_factor = 4
    ifmdim = 64
    kdim = 8
    ofmdim = ifmdim // kdim
    ifm = 1
    ofm = 16
    ishp = (1, ifm, ifmdim, ifmdim)
    oshp = (1, ofm, ofmdim, ofmdim)
    wshp = (ofm, ifm, kdim, kdim)
    dt0 = DataType["UINT8"]
    wdt = DataType["INT4"]
    np.random.seed(0)
    ishp_str = str(list(ishp))
    oshp_str = str(list(oshp))
    wshp_str = str(list(wshp))

    input = f"""
    <
        ir_version: 7,
        opset_import: ["" : 9]
    >
    agraph (float{ishp_str} in0) => (float{oshp_str} out0)
    <
        float{wshp_str} conv_param
    >
    {{
        out0 = Conv<
                dilations=[1,1], group=1, kernel_shape=[{kdim},{kdim}],
                strides=[{kdim},{kdim}], pads=[0,0,0,0]
            >(in0, conv_param)
    }}
    """
    model = oprs.parse_model(input)
    model = ModelWrapper(model)
    model.set_tensor_datatype("in0", dt0)
    w = gen_finn_dt_tensor(wdt, wshp)
    model.set_initializer("conv_param", w)
    model = model.transform(InferShapes())
    inp = gen_finn_dt_tensor(dt0, ishp)
    input_dict = {"in0": inp}
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    im2col_node = model.get_nodes_by_op_type("Im2Col")[0]
    old_im2col_ishape = model.get_tensor_shape(im2col_node.input[0])
    old_im2col_oshape = model.get_tensor_shape(im2col_node.output[0])
    assert tuple(old_im2col_ishape) == (1, ifmdim, ifmdim, ifm)
    out_expected = oxe.execute_onnx(model, input_dict)["out0"]
    model = model.transform(RebalanceIm2Col(ch_factor))
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    im2col_node = model.get_nodes_by_op_type("Im2Col")[0]
    new_im2col_ishape = model.get_tensor_shape(im2col_node.input[0])
    new_im2col_oshape = model.get_tensor_shape(im2col_node.output[0])
    out_produced = oxe.execute_onnx(model, input_dict)["out0"]
    assert len(model.get_nodes_by_op_type("Reshape")) == 1
    assert tuple(new_im2col_ishape) == (1, ifmdim, ifmdim // ch_factor, ch_factor)
    assert old_im2col_oshape == new_im2col_oshape
    assert (out_expected == out_produced).all()
