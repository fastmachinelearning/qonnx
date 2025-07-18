# Copyright (c) 2021 Xilinx, Inc.
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

from qonnx.core.datatype import DataType
from qonnx.custom_op.base import CustomOp


def binary_quant(inp_tensor, scale):
    # ToDo: Update this link, when the PR gets merged
    # Port of IntQuant class from Brevitas: https://bit.ly/2S6qvZJ

    # Quantizing
    y_int = inp_tensor
    y_ones = np.ones(y_int.shape, dtype=y_int.dtype)
    y_int = np.where(y_int >= 0.0, y_ones, -y_ones)
    # Scaling
    out_tensor = y_int * scale

    return out_tensor


class BipolarQuant(CustomOp):
    """Bipolar quantization operation for QONNX. Takes four inputs:
    - input tensor to quantize
    - the scale

    The output is a tensor of the same shape as the input tensor, with quantized
    values.
    """

    def get_nodeattr_types(self):
        return dict()

    def make_shape_compatible_op(self, model):
        node = self.onnx_node
        return helper.make_node("Identity", [node.input[0]], [node.output[0]])

    def get_integer_datatype(self, model):
        return DataType["BIPOLAR"]

    def get_output_dtype(self, model):
        node = self.onnx_node
        # scale must be read from initializers
        scale = model.get_initializer(node.input[1])
        # determine the QONNX DataType
        unit_scale = np.all(scale == 1.0)
        if unit_scale:
            finn_dt = self.get_integer_datatype(model)
        else:
            finn_dt = DataType["FLOAT32"]

        return finn_dt

    def infer_node_datatype(self, model):
        try:
            finn_dt = self.get_output_dtype(model)
        except AssertionError:
            finn_dt = DataType["FLOAT32"]
        node = self.onnx_node
        model.set_tensor_datatype(node.output[0], finn_dt)

    def execute_node(self, context, graph):
        node = self.onnx_node
        # save inputs
        inp_tensor = context[node.input[0]]
        scale = context[node.input[1]]
        # calculate output
        ret = binary_quant(inp_tensor, scale)
        # ensure output is ndarray (even if 0d)
        # since numpy silently flattens 0d arrays to scalars
        # more: https://github.com/numpy/numpy/issues/13105
        if not isinstance(ret, np.ndarray):
            ret = np.asarray(ret, dtype=np.float32)
        # set context according to output name
        context[node.output[0]] = ret

    def verify_node(self):
        pass
