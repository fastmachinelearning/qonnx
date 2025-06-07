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
from HGQ.proxy.fixed_point_quantizer import gfixed_quantizer
from onnx import TensorProto, helper

from qonnx.core.datatype import DataType
from qonnx.custom_op.base import CustomOp


class FixedPoint(CustomOp):
    """Generic quantization operation for HGQ FixedPoint layer to QONNX.

    Takes four inputs:

    - input tensor to quantize
    - the integer_bits
    - the keep_negative
    - the bits

    The output is a tensor of the same shape as the input tensor, with quantized
    values.
    """

    def get_nodeattr_types(self):
        return {
            # The rounding mode, which is used for the quant function
            # (e.g. "TRN": Truncate towards negative infinity. Fast. Preferred when possible.)
            "RND": ("s", True, "TRN"),
            # Saturate between highest and lowest representable values.
            # (e.g. "WRAP" Wrap around.)
            "SAT": ("s", True, "WRAP"),
        }

    def make_shape_compatible_op(self, model):
        """Returns a standard ONNX op which is compatible with this CustomOp
        for performing shape inference."""
        return helper.make_node(
            "Cast",
            inputs=[self.onnx_node.input[0]],
            outputs=[self.onnx_node.output[0]],
            to=int(TensorProto.FLOAT),
        )

    def get_integer_datatype(self, model):
        signed = self.get_nodeattr("signed")
        bit_width = model.get_initializer(self.onnx_node.input[3])
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
        bit_width = model.get_initializer(self.onnx_node.input[3])
        bit_width = int(bit_width)
        finn_dt = DataType["SCALEDINT<%d>" % (bit_width)]
        return finn_dt

    def get_output_dtype(self, model):
        raise NotImplementedError("get_output_dtype for FixedPoint is not implemented")

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
        # TODO: we assume here an order that the name of the input[1] = keep_negative etc.
        keep_negative = context[node.input[1]]
        bits = context[node.input[2]]
        integer_bits = context[node.input[3]]
        # save attributes
        RND = self.get_nodeattr("RND")
        SAT = self.get_nodeattr("SAT")
        # calculate output
        ret = gfixed_quantizer(inp_tensor, keep_negative, bits, integer_bits, RND=RND, SAT=SAT)
        # ensure output is ndarray (even if 0d)
        # since numpy silently flattens 0d arrays to scalars
        # more: https://github.com/numpy/numpy/issues/13105
        if not isinstance(ret, np.ndarray):
            ret = np.asarray(ret, dtype=np.float32)
        if not ret.dtype == np.float32:
            ret = ret.astype(np.float32)
        # set context according to output name
        context[node.output[0]] = ret

    def verify_node(self):
        pass
