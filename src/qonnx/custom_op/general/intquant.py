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
from onnx import TensorProto, helper

from qonnx.core.datatype import DataType
from qonnx.custom_op.base import CustomOp


def min_int(signed: bool, narrow_range: bool, bit_width: int) -> int:
    """Compute the minimum integer representable by a given number of bits.
    Args:

        signed (bool): Indicates whether the represented integer is signed or not.
        narrow_range (bool): Indicates whether to narrow the minimum value
        represented by 1.
        bit_width (int): Number of bits available for the representation.

    Returns:

        int: Maximum unsigned integer that can be represented according to
        the input arguments.

    Examples:

        >>> min_int(signed=True, narrow_range=True, bit_width=8)
        int(-127)
        >>> min_int(signed=False, narrow_range=True, bit_width=8)
        int(0)
        >>> min_int(signed=True, narrow_range=False, bit_width=8)
        int(-128)
        >>> min_int(signed=False, narrow_range=False, bit_width=8)
        int(0)

    """
    if signed and narrow_range:
        value = -(2 ** (bit_width - 1)) + 1
    elif signed and not narrow_range:
        value = -(2 ** (bit_width - 1))
    else:
        value = 0 * bit_width
    return value


def max_int(signed: bool, narrow_range: bool, bit_width: int) -> int:
    """Compute the maximum integer representable by a given number of bits.
    Args:

        signed (bool): Indicates whether the represented integer is signed or not.
        narrow_range (bool): Indicates whether to narrow the maximum unsigned value
        represented by 1.
        bit_width (int): Number of bits available for the representation.

    Returns:

        Tensor: Maximum integer that can be represented according to
        the input arguments.

    Examples:

        >>> max_int(signed=True, narrow_range=True, bit_width=8)
        int(127)
        >>> max_int(signed=False, narrow_range=True, bit_width=8)
        int(254)
        >>> max_int(signed=True, narrow_range=False, bit_width=8)
        int(127)
        >>> max_int(signed=False, narrow_range=False, bit_width=8)
        int(255)

    """
    if not signed and not narrow_range:
        value = (2**bit_width) - 1
    elif not signed and narrow_range:
        value = (2**bit_width) - 2
    else:
        value = (2 ** (bit_width - 1)) - 1
    return value


def int_quant(inp_tensor, scale, zeropt, bitwidth, signed, narrow, rounding_mode):
    # ToDo: Update this link, when the PR gets merged
    # Port of IntQuant class from Brevitas: https://bit.ly/2S6qvZJ

    # Scaling
    y_int = inp_tensor / scale
    y_int = y_int + zeropt
    if bitwidth == 1 and signed:
        # BUG: 1-bit IntQuant ops currently not exported correctly
        # manually convert to bipolar values
        y_ones = np.ones(y_int.shape, dtype=y_int.dtype)
        y_int = np.where(y_int >= 0.0, y_ones, -y_ones)
    else:
        # Clamping
        min_int_val = min_int(signed, narrow, bitwidth)
        max_int_val = max_int(signed, narrow, bitwidth)
        y_int = np.where(y_int > max_int_val, max_int_val.astype(y_int.dtype), y_int)
        y_int = np.where(y_int < min_int_val, min_int_val.astype(y_int.dtype), y_int)
        # Rounding
        rounding_fx = resolve_rounding_mode(rounding_mode)
        y_int = rounding_fx(y_int)
    # Re-scaling
    out_tensor = y_int - zeropt
    out_tensor = out_tensor * scale

    return out_tensor


def resolve_rounding_mode(mode_string):
    """Resolve the rounding mode string of IntQuant and Trunc ops
    to the corresponding numpy functions."""
    normalized_mode_string = mode_string.upper()
    if normalized_mode_string == "ROUND" or normalized_mode_string == "HALF_EVEN":
        return np.round
    elif normalized_mode_string == "CEIL":
        return np.ceil
    elif normalized_mode_string == "FLOOR":
        return np.floor
    elif normalized_mode_string == "UP":

        def round_up(x):
            return np.sign(x) * np.ceil(np.abs(x))

        return round_up
    elif normalized_mode_string == "DOWN":
        return np.fix
    elif normalized_mode_string == "HALF_UP":

        def round_half_up(x):
            return np.sign(x) * np.floor(np.abs(x) + 0.5)

        return round_half_up
    elif normalized_mode_string == "HALF_DOWN":

        def round_half_down(x):
            return np.sign(x) * np.ceil(np.abs(x) - 0.5)

        return round_half_down
    else:
        raise ValueError(f"Could not resolve rounding mode called: {normalized_mode_string}")


class IntQuant(CustomOp):
    """Generic quantization operation for QONNX. Takes four inputs:
    - input tensor to quantize
    - the scale
    - the zero-point
    - the bit-width

    The output is a tensor of the same shape as the input tensor, with quantized
    values.
    """

    def get_nodeattr_types(self):
        return {
            # whether the quantization interval should be signed or not
            # (e.g. at 8b unsigned=[0, 255] vs signed=[-128, 127])
            "signed": ("i", True, 1),
            # when signed=1, whether to use narrow range or not
            # (e.g. at 8b regular=[-128, 127] vs narrow=[-127, 127])
            "narrow": ("i", True, 1),
            # The rounding mode, which is used for the int_quant function
            # ToDo: This should be required (True) instead of optional (False)
            "rounding_mode": ("s", False, "ROUND"),
        }

    def make_shape_compatible_op(self, model):
        """Returns a standard ONNX op which is compatible with this CustomOp
        for performing shape inference."""
        node_out = self.onnx_node.output[0]
        # preserve existing ONNX tensor type if it exists
        node_out_vi = model.get_tensor_valueinfo(node_out)
        if node_out_vi is None:
            return helper.make_node(
                "Cast",
                inputs=[self.onnx_node.input[0]],
                outputs=[node_out],
                to=int(TensorProto.FLOAT),
            )
        else:
            return helper.make_node(
                "Cast",
                inputs=[self.onnx_node.input[0]],
                outputs=[node_out],
                to=int(node_out_vi.type.tensor_type.elem_type),
            )
        # For Quant the output shape should be the same as the input shape.
        # Get the output shape from the input
        out_shape = model.get_tensor_shape(self.onnx_node.input[0])

        # implement tensor with correct shape
        values = np.random.randn(*out_shape).astype(np.float32)
        return helper.make_node(
            "Constant",
            inputs=[],
            outputs=[self.onnx_node.output[0]],
            value=helper.make_tensor(
                name="const_tensor",
                data_type=TensorProto.FLOAT,
                dims=values.shape,
                vals=values.flatten().astype(float),
            ),
            name=self.onnx_node.name,
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
        node = self.onnx_node
        # scale, zero-point and bitwidth must be read from initializers
        scale = model.get_initializer(node.input[1])
        zeropt = model.get_initializer(node.input[2])
        bitwidth = model.get_initializer(node.input[3])
        assert scale is not None, "Found unspecified scale for IntQuant node: " + str(node)
        assert zeropt is not None, "Found unspecified zero point for IntQuant node: " + str(node)
        assert bitwidth is not None, "Found unspecified bitwidth for IntQuant node: " + str(node)
        # extract the bitwidth (assume scalar)
        assert bitwidth.ndim == 0, "Bitwidth must be scalar for IntQuant node: " + str(node)
        bitwidth = bitwidth.item()
        assert int(bitwidth) == bitwidth, "Bitwidth must be integer for IntQuant node: " + str(node)
        bitwidth = int(bitwidth)
        # determine the FINN DataType
        unit_scale = np.all(scale == 1.0)
        zero_zeropt = np.all(zeropt == 0.0)
        assert zero_zeropt, "Only zero_point=0 IntQuant nodes supported for now"
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
        bitwidth = context[node.input[3]]
        # save attributes
        signed = self.get_nodeattr("signed")
        narrow = self.get_nodeattr("narrow")
        rounding_mode = self.get_nodeattr("rounding_mode")
        # calculate output
        ret = int_quant(inp_tensor, scale, zeropt, bitwidth, signed, narrow, rounding_mode)
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
