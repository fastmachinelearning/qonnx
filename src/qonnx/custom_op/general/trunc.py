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
from qonnx.custom_op.general.quant import max_int, min_int, resolve_rounding_mode


def trunc(inp_tensor, scale, zeropt, input_bit_width, narrow, signed, output_scale, output_bit_width, rounding_mode):
    # Port of TruncIntQuant class from Brevitas: https://bit.ly/3wzIpTR

    # Scaling
    y = inp_tensor / scale
    y = y + zeropt
    # Rounding
    y = np.round(y)
    # Rescale
    trunc_scale = 2 ** np.round(
        np.log2(output_scale / scale)
    )  # Trunc scale should be a power-of-two - ensure that is the case
    y = y / trunc_scale

    # Clamping
    min_int_val = min_int(signed, narrow, output_bit_width)
    max_int_val = max_int(signed, narrow, output_bit_width)
    y = np.where(y > max_int_val, max_int_val.astype(y.dtype), y)
    y = np.where(y < min_int_val, min_int_val.astype(y.dtype), y)
    # To int (truncate)
    rounding_fx = resolve_rounding_mode(rounding_mode)
    y = rounding_fx(y)

    # Rescale
    output_zeropt = zeropt / trunc_scale  # Rescale zero-point
    y = y - output_zeropt
    y = y * output_scale

    return y


class Trunc(CustomOp):
    """Generic truncation operation for QONNX. Takes four inputs:  
    - input tensor to truncate  
    - the scale  
    - the zero-point  
    - the truncation scale  
    - the truncation bit-width

    The output is a tensor of the same shape as the input tensor, with truncated
    values.
    """

    def get_nodeattr_types(self):
        return {
            # The rounding mode, which is used for the trunc function
            "rounding_mode": ("s", True, "FLOOR"),
            "narrow": ("i", False, 0, {0, 1}),
            "signed": ("i", False, 1, {0, 1}),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node
        return helper.make_node("Identity", [node.input[0]], [node.output[0]])

    def infer_node_datatype(self, model):
        node = self.onnx_node
        model.set_tensor_datatype(node.output[0], DataType["FLOAT32"])

    def execute_node(self, context, graph):
        node = self.onnx_node
        # save inputs
        inp_tensor = context[node.input[0]]
        scale = context[node.input[1]]
        zeropt = context[node.input[2]]
        input_bit_width = context[node.input[3]]
        output_scale = context[node.input[4]]
        output_bit_width = context[node.input[5]]
        # save attributes
        rounding_mode = self.get_nodeattr("rounding_mode")
        narrow = self.get_nodeattr("narrow")
        signed = self.get_nodeattr("signed")
        # calculate output
        ret = trunc(
            inp_tensor, scale, zeropt, input_bit_width, narrow, signed, output_scale, output_bit_width, rounding_mode
        )
        # set context according to output name
        context[node.output[0]] = ret

    def verify_node(self):
        pass
