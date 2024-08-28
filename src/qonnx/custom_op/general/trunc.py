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
from qonnx.custom_op.general.quant import resolve_rounding_mode


def trunc(inp_tensor, scale, zeropt, input_bit_width, output_bit_width, rounding_mode):
    # Port of TruncIntQuant class from Brevitas: https://bit.ly/3wzIpTR

    # Scaling
    y = inp_tensor / scale
    y = y + zeropt
    # Rounding
    y = np.round(y)
    # Truncate
    trunc_bit_width = input_bit_width - output_bit_width
    trunc_scale = 2.0**trunc_bit_width
    y = y / trunc_scale

    # To int
    rounding_fx = resolve_rounding_mode(rounding_mode)
    y = rounding_fx(y)

    # Rescale
    y = y - zeropt
    y = y * scale

    return y


class Trunc(CustomOp):
    """Generic truncation operation for QONNX. Takes four inputs:
    - input tensor to truncate
    - the scale
    - the zero-point
    - the truncation bit-width

    The output is a tensor of the same shape as the input tensor, with truncated
    values.
    """

    def get_nodeattr_types(self):
        return {
            # The rounding mode, which is used for the trunc function
            "rounding_mode": ("s", True, "FLOOR"),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node
        return helper.make_node("Identity", [node.input[0]], [node.output[0]])

    def get_integer_datatype(self, model):
        signed = model.get_tensor_datatype(self.onnx_node.input[0]).signed()
        bit_width = model.get_initializer(self.onnx_node.input[4])
        bit_width = int(bit_width)
        if bit_width == 1:
            if signed:
                finn_dt = DataType["BIPOLAR"]
            else:
                finn_dt = DataType["BINARY"]
        else:
            if signed:
                finn_dt = DataType["INT" + str(bit_width)]
            else:
                finn_dt = DataType["UINT" + str(bit_width)]
        return finn_dt

    def get_scaled_integer_datatype(self, model):
        bit_width = model.get_initializer(self.onnx_node.input[4])
        bit_width = int(bit_width)
        finn_dt = DataType["SCALEDINT<%d>" % (bit_width)]
        return finn_dt

    def get_output_dtype(self, model):
        node = self.onnx_node
        # scale, zero-point and bitwidth must be read from initializers
        scale = model.get_initializer(node.input[1])
        zeropt = model.get_initializer(node.input[2])
        bitwidth = model.get_initializer(node.input[4])
        assert scale is not None, "Found unspecified scale for Trunc node: " + str(node)
        assert zeropt is not None, "Found unspecified zero point for Trunc node: " + str(node)
        assert bitwidth is not None, "Found unspecified output bitwidth for Trunc node: " + str(node)
        # extract the bitwidth (assume scalar)
        assert bitwidth.ndim == 0, "Bitwidth must be scalar for Trunc node: " + str(node)
        bitwidth = bitwidth.item()
        assert int(bitwidth) == bitwidth, "Bitwidth must be integer for Trunc node: " + str(node)
        bitwidth = int(bitwidth)
        # determine the FINN DataType
        unit_scale = np.all(scale == 1.0)
        zero_zeropt = np.all(zeropt == 0.0)
        assert zero_zeropt, "Only zero_point=0 Trunc nodes supported for now"
        if unit_scale and zero_zeropt:
            finn_dt = self.get_integer_datatype(model)
        else:
            finn_dt = self.get_scaled_integer_datatype(model)
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
        zeropt = context[node.input[2]]
        input_bit_width = context[node.input[3]]
        output_bit_width = context[node.input[4]]
        # save attributes
        rounding_mode = self.get_nodeattr("rounding_mode")
        # calculate output
        ret = trunc(inp_tensor, scale, zeropt, input_bit_width, output_bit_width, rounding_mode)
        # set context according to output name
        context[node.output[0]] = ret

    def verify_node(self):
        pass
