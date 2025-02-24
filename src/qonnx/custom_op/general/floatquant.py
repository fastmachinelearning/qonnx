# Copyright (c) 2024 Nicolo Ghielmetti
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
from onnx import TensorProto, helper

from qonnx.custom_op.base import CustomOp
from qonnx.custom_op.general.quant import resolve_rounding_mode


def compute_default_exponent_bias(exponent_bitwidth):
    return (2.0 ** (exponent_bitwidth - 1)) - 1


def compute_max_val(exponent_bitwidth, mantissa_bitwidth, exponent_bias=None):
    if exponent_bias is None:
        exponent_bias = compute_default_exponent_bias(exponent_bitwidth)
    max_exponent = (2.0**exponent_bitwidth) - 1.0 - exponent_bias
    max_mantissa = np.sum((2.0 ** np.arange(0, -1.0 * mantissa_bitwidth - 1.0, -1.0)))
    max_val = max_mantissa * (2**max_exponent)
    return max_val


def float_quant(
    X,
    scale,
    exponent_bitwidth,
    mantissa_bitwidth,
    exponent_bias,
    signed,
    max_val=None,
    has_inf=False,
    has_nan=False,
    has_subnormal=False,
    rounding_mode="ROUND",
    saturation=True,
):
    # the comments are left to track the correspondence with the brevitas code
    # np version of brevitas function
    def inf_nan_clamp(X, inf_mask, p_max_val_mask, n_max_val_mask):
        if has_inf:
            X[p_max_val_mask] = np.inf
            X[n_max_val_mask] = -np.inf
        elif has_nan:
            full_max_val_mask = np.logical_or(p_max_val_mask, n_max_val_mask)
            X[full_max_val_mask] = np.nan
            X[inf_mask] = np.nan
        else:
            raise RuntimeError("Clamping is not saturating, but neither `inf_values` nor `nan_values` is specified")
        return X

    # consistency check
    # if bit_width != exponent_bitwidth + mantissa_bitwidth + int(signed):
    #         raise RuntimeError("Mismatch between total bit-width, exponent, mantissa and sign.")

    # x = self.input_view_impl(x) # assuming input_view_impl is Identity

    # the following lines (up to max_value assignment) implements the float_internal_scale function from brevitas using numpy
    # internal_scale = float_internal_scale(
    #     scaled_x, self.mantissa_bit_width(), self.fp_internal_scale_min(), self.eps)

    X = X / scale

    eps = np.finfo(X.dtype).tiny  # the datatype used here and in brevitas must be the same to have the same eps
    fp_internal_scale_min = 1.0 - exponent_bias - mantissa_bitwidth

    internal_scale = np.floor(np.log2(np.abs(X) + eps)) - mantissa_bitwidth
    internal_scale = np.maximum(
        internal_scale, fp_internal_scale_min
    )  # np version of: internal_scale = torch.ok(internal_scale, fp_internal_scale_min)
    internal_scale = np.exp2(internal_scale)

    x_q = internal_scale * resolve_rounding_mode(rounding_mode)(
        X / internal_scale
    )  # self.float_to_int_impl(x / internal_scale)

    max_value = compute_max_val(exponent_bitwidth, mantissa_bitwidth, exponent_bias)
    max_value = max_value if max_val is None else np.minimum(max_value, max_val)
    min_value = 0.0 if not signed else -max_value

    # Compute masks
    inf_mask = np.isinf(x_q)
    p_max_val_mask = x_q > max_value
    n_max_val_mask = x_q < min_value

    # first clamp everything to  [min_value,max_value], basically the saturating case
    x_q = np.clip(x_q, min_value, max_value)  # self.saturating_clamp(x_q, max_value, min_value)

    if not saturation:
        x_q = inf_nan_clamp(x_q, inf_mask, p_max_val_mask, n_max_val_mask)

    return x_q * scale  # , self.saturating, self.inf_values, self.nan_values


class FloatQuant(CustomOp):
    """Floating point quantization operation for QONNX.

    The output is a tensor of the same shape as the input tensor, with quantized
    values.
    """

    def get_nodeattr_types(self):
        return {
            # Integer value interpreted as boolean, defines whether the representation supports signed values.
            # This attribute has no effect on the execution of this operation and is intended purely to inform backends.
            # "signed": ("i", True, 1),
            # Defines how rounding should be applied during quantization.
            # Currently available modes are: "ROUND", "CEIL" and "FLOOR". Here "ROUND" implies a round-to-even operation.
            # Lowercase variants for the rounding mode string are also supported: "round", "ceil", "floor".
            "rounding_mode": ("s", False, "ROUND"),
            # Integer value interpreted as boolean, defines whether the representation supports infinity values.
            # The ability to represent infinity values will decrease the representable numerical range.
            # This attribute has no effect on the execution of this operation and is intended purely to inform backends.
            "has_inf": ("i", True, 0),
            # Integer value interpreted as boolean, defines whether the representation supports not-a-number (NaN) values.
            # The ability to represent NaN values will decrease the representable numerical range.
            # This attribute has no effect on the execution of this operation and is intended purely to inform backends.
            "has_nan": ("i", True, 0),
            # # Integer value interpreted as boolean, defines whether the representation supports subnormal values.
            # Subnormal values have an exponent value of 0 and
            # are interpreted to have a leading significand digit of zero rather than one.
            # Supporting subnormals will increase the complexity of the required arithmetic datapath.
            # This attribute has no effect on the execution of this operation and is intended purely to inform backends.
            "has_subnormal": ("i", True, 0),
            # Integer value interpreted as boolean, defines whether the representation will saturate during arithmetic.
            # This attribute has no effect on the execution of this operation and is intended purely to inform backends.
            "saturation": ("i", True, 1),
        }

    def execute_node(self, context, graph):
        node = self.onnx_node
        # save inputs
        inp_tensor = context[node.input[0]]
        scale = context[node.input[1]]
        exponent_bitwidth = context[node.input[2]]
        mantissa_bitwidth = context[node.input[3]]
        exponent_bias = context[node.input[4]]
        max_val = context[node.input[5]]
        # save attributes
        # signed = self.get_nodeattr("signed")
        signed = True
        rounding_mode = self.get_nodeattr("rounding_mode")
        has_inf = self.get_nodeattr("has_inf")
        has_nan = self.get_nodeattr("has_nan")
        has_subnormal = self.get_nodeattr("has_subnormal")  # not supported in Brevitas, so not supported for the moment
        saturation = self.get_nodeattr("saturation")

        # calculate output
        ret = float_quant(
            inp_tensor,
            scale,
            exponent_bitwidth,
            mantissa_bitwidth,
            exponent_bias,
            signed,
            max_val,
            has_inf,
            has_nan,
            has_subnormal,
            rounding_mode,
            saturation,
        )  # signed, max_val, rounding_mode, has_inf, has_nan, saturating)
        # ensure output is ndarray (even if 0d)
        # since numpy silently flattens 0d arrays to scalars
        # more: https://github.com/numpy/numpy/issues/13105
        if not isinstance(ret, np.ndarray):
            ret = np.asarray(ret, dtype=np.float32)
        if not ret.dtype == np.float32:
            ret = ret.astype(np.float32)
        # set context according to output name
        context[node.output[0]] = ret

    def make_shape_compatible_op(self, model):
        """Returns a standard ONNX op which is compatible with this CustomOp
        for performing shape inference."""
        return helper.make_node(
            "Cast",
            inputs=[self.onnx_node.input[0]],
            outputs=[self.onnx_node.output[0]],
            to=int(TensorProto.FLOAT),
        )

    def infer_node_datatype(self, model):
        pass

    def verify_node(self):
        pass
