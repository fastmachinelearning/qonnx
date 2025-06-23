# Copyright (c) 2025 Advanced Micro Devices, Inc.
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
from onnxscript import ir
from onnxscript.rewriter import pattern, rewrite
from warnings import warn

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import MovePadAttributeToTensor, RemoveUnusedTensors

# TODO: operate on IntQuant when Quant -> IntQuant is added to cleanup steps
# TODO: add transformation for Trunc to standard ops
# TODO: add transformation for BipolarQuant to standard ops
# TODO: add transformation for FloatQuant to standard ops


# Target patterns
def quant_pattern_brevitas(qonnx_op, x, scale, zero_point, bitwidth, signed, narrow, rounding_mode):
    return qonnx_op.Quant(
        x, scale, zero_point, bitwidth, signed=signed, narrow=narrow, _allow_other_attributes=True, _domain="onnx.brevitas"
    )


def quant_pattern_qonnx(qonnx_op, x, scale, zero_point, bitwidth, signed, narrow, rounding_mode):
    return qonnx_op.Quant(
        x,
        scale,
        zero_point,
        bitwidth,
        signed=signed,
        narrow=narrow,
        _allow_other_attributes=True,
        _domain="qonnx.custom_op.general",
    )


# Replacement pattern
def qcdq_pattern(op, x, scale, zero_point, bitwidth, signed, narrow, rounding_mode):
    # New datatype for the internal tensors (zero_point, min_val, max_val)
    new_dtype = TensorProto.INT8 if signed.value else TensorProto.UINT8
    np_dtype = np.int8 if signed.value else np.uint8

    # Create the QuantizeLinear node
    scale_np = scale.const_value.numpy()
    scale_np_new = scale_np.squeeze()
    # TODO add support for non-zero zero_point, taking different definitions into account:
    # DequantizeLinear(QuantizeLinear(x)) uses scale * ((saturate((x / scale) + zero_point) - zero_point))
    # Quant(x) uses  scale * (round(clip(x / scale + zero_point)) - zero_point)
    if scale_np_new.ndim == 1:
        qnt_axis = scale_np.shape.index(scale_np_new.shape[0])
        c_scale = helper.make_tensor("scale", scale.dtype, scale_np_new.shape, scale_np_new)
        c_zero_point = helper.make_tensor(
            "zero_point", new_dtype, scale_np_new.shape, np.zeros(scale_np_new.shape, dtype=np_dtype)
        )
    else:
        qnt_axis = None
        c_scale = helper.make_tensor("scale", scale.dtype, (), [scale_np_new.item()])
        c_zero_point = helper.make_tensor("zero_point", new_dtype, (), [0])
    scale = op.Constant(value=c_scale)
    zero_point = op.Constant(value=c_zero_point)

    qnt = op.QuantizeLinear(x, scale, zero_point, axis=qnt_axis)

    bw_val = bitwidth.const_value.numpy().squeeze()
    if (signed.value and narrow.value) or (bw_val < 8):
        # Compute clipping values
        if signed.value:
            max_val = 2 ** (bw_val - 1) - 1
            if narrow.value:
                min_val = -(2 ** (bw_val - 1)) + 1
            else:
                min_val = -(2 ** (bw_val - 1))
        else:
            min_val = 0
            if narrow.value:
                max_val = 2**bw_val - 2
            else:
                max_val = 2**bw_val - 1

        if isinstance(min_val, np.ndarray):
            min_val = min_val.astype(np_dtype)
        elif isinstance(min_val, int):
            pass
        elif isinstance(min_val, float):
            min_val = int(min_val)
        if isinstance(max_val, np.ndarray):
            max_val = max_val.astype(np_dtype)
        elif isinstance(max_val, int):
            pass
        elif isinstance(max_val, float):
            max_val = int(max_val)
        c_min_val = helper.make_tensor("min_val", new_dtype, (), [min_val])
        c_max_val = helper.make_tensor("max_val", new_dtype, (), [max_val])
        min_val = op.Constant(value=c_min_val)
        max_val = op.Constant(value=c_max_val)

        qnt = op.Clip(qnt, min_val, max_val)

    # Create the DequantizeLinear node
    return op.DequantizeLinear(qnt, scale, zero_point, axis=qnt_axis)


def is_valid_qcdq_transformation(context, x, scale, zero_point, bitwidth, signed, narrow, rounding_mode, **_) -> bool:
    """Condition to check if the Quant node can be replaced.
    The following conditions must be satisfied:
    - the scale, zero-point and bitwidth inputs for Quant must be statically specified
      by an initializer
    - the bitwidth must be an integer in the range [2, 8] # TODO: Change max bitwidth to 16 for opset >= 21
    - the zero-point tensor must be zero
    - the scale must be a scalar value or 1D tensor
    - the rounding_mode attribute must be ROUND
    """

    # Check scale
    scale_val = scale.const_value
    if scale_val is None or (scale_val.numpy().squeeze().ndim not in [0, 1]):
        return False

    # Check zero point
    zero_val = zero_point.const_value
    if zero_val is None or (zero_val.numpy().squeeze() != 0):
        return False

    # Check bitwidth
    bitwidth_val = bitwidth.const_value
    if bitwidth_val is None or (len(bitwidth_val.shape) != 0):
        return False
    bitwidth_val = bitwidth_val.numpy().squeeze()
    if bitwidth_val < 2 or bitwidth_val > 8:
        return False

    # Check rounding mode
    if rounding_mode is None:  # No rounding_mode specified, assume default to be `ROUND`
        return True
    if rounding_mode.value != "ROUND":
        return False
    return True


class QuantToQCDQ(Transformation):
    """Replace QONNX Quant-style quantization nodes with QuantizeLinear
    -> Clip -> DequantizeLinear (QCDQ)-style quantization nodes. The following
    restictions apply on the Quant:
    - the scale, zero-point and bitwidth inputs for Quant must be statically specified
      by an initializer
    - the bitwidth must be an integer in the range [2, 8]
    - the zero-point tensor must be zero
    - the scale must be a scalar value or 1D tensor
    - the rounding_mode attribute must be ROUND
    BipolarQuant is not (yet) supported.
    """

    def __init__(self):
        super().__init__()
        rewrite_rule_qcdq_brevitas = pattern.RewriteRule(quant_pattern_brevitas, qcdq_pattern, is_valid_qcdq_transformation)
        rewrite_rule_qcdq_qonnx = pattern.RewriteRule(quant_pattern_qonnx, qcdq_pattern, is_valid_qcdq_transformation)
        self._rewrite_rule_set = pattern.RewriteRuleSet([rewrite_rule_qcdq_brevitas, rewrite_rule_qcdq_qonnx], commute=True)

        self._preserve_qnt_optypes = ["Quant", "BipolarQuant", "QuantizeLinear", "DequantizeLinear"]

    def apply(self, model: ModelWrapper):
        model = ir.from_proto(model.model)
        model = rewrite(model, pattern_rewrite_rules=self._rewrite_rule_set)
        model = ir.to_proto(model)
        model = ModelWrapper(model)

        # Ensure opset version is new enough if there were Clip nodes inserted
        if len(model.get_nodes_by_op_type("Clip")) > 0:
            model = model.transform(RemoveUnusedTensors())

            clip_min_opset = 13
            if model.model.opset_import[0].version < clip_min_opset:
                warn(f"QCDQ Clip requires ONNX opset >= {clip_min_opset} but found {model.model.opset_import[0].version}")
                warn(f"Forcing opset version {clip_min_opset} upgrade to ensure valid ONNX")
                model.model.opset_import[0].version = clip_min_opset
                # Ensure new Pad node requirements are respected
                model = model.transform(MovePadAttributeToTensor())

        # Ensure opset version is new enough if there were QuantizeLinear nodes inserted
        # Fixes the corner case where none of the Quant nodes need a Clip node
        elif len(model.get_nodes_by_op_type("QuantizeLinear")) > 0:
            model = model.transform(RemoveUnusedTensors())

            qdq_min_opset = 10
            if model.model.opset_import[0].version < qdq_min_opset:
                warn(
                    f"QCDQ QuantizeLinear requires ONNX opset >= {qdq_min_opset} but found"
                    " {model.model.opset_import[0].version}"
                )
                warn(f"Forcing opset version {qdq_min_opset} upgrade to ensure valid ONNX")
                model.model.opset_import[0].version = qdq_min_opset
                # Ensure new Pad node requirements are respected
                model = model.transform(MovePadAttributeToTensor())

        return (model, False)
