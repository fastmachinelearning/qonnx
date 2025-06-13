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
# * Neither the name of AMD nor the names of its
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
from onnx import helper as oh

from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.util.basic import get_by_name

from qonnx.core.modelwrapper import ModelWrapper
from onnxscript import opset15 as op
from onnxscript import script
from onnxscript.rewriter import pattern, rewrite
from onnxscript import ir

from qonnx.util.onnxscript import ReplacePattern

def target_pattern(op, x, scale, bias, mean, var):
    return op.BatchNormalization(x, scale, bias, mean, var)

def replace_pattern(op, x, scale, bias, mean, var, **kwargs):

    # Get epsilon from matched pattern
    batch_norm = kwargs['match'].nodes[0]
    epsilon_attr = batch_norm.attributes.get('epsilon', None)
    epsilon_value = 1e-5 if epsilon_attr is None else epsilon_attr.value
    epsilon_tensor = helper.make_tensor("epsilon", TensorProto.FLOAT, (1,), [epsilon_value])
    epsilon = op.Constant(value=epsilon_tensor)

    A = op.Div(scale, op.Sqrt(op.Add(var, epsilon)))
    B = op.Sub(bias, op.Mul(A, mean))

    # Unsqueeze A and B
    input_shape = x.shape
    assert input_shape is not None and len(input_shape) >= 2
    n_spatial_dims = len(input_shape) - 2
    axes = [0] + [i + 2 for i in range(n_spatial_dims)]
    A = op.Unsqueeze(A, axes=axes)
    B = op.Unsqueeze(B, axes=axes)

    return op.Add(op.Mul(x, A), B)

rule1 = pattern.RewriteRule(target_pattern, ReplacePattern(replace_pattern), verbose=10)
rewrite_rules = pattern.RewriteRuleSet([rule1])

class BatchNormToAffine(Transformation):
    """Replaces any test-time BatchNorm layers with Mul-Add layers."""

    def apply(self, model):
        model = ir.from_proto(model.model)
        model = rewrite(model, pattern_rewrite_rules=rewrite_rules)
        model = ir.to_proto(model)
        model = ModelWrapper(model)
        model = model.transform(InferShapes())
        model = model.transform(FoldConstants())
        return (model, False)

