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
from warnings import warn

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.custom_op.general.intquant import resolve_rounding_mode
from qonnx.core.datatype import DataType


def default_op_filter(op):
    return op.op_type in ["Add", "Mul"]


class FixedPointQuantizeParamsFromDict(Transformation):
    """
    Quantize model parameters to a given fixed-point representation.
    The self.max_err dictionary stores the maximum error for each quantized input after calling.
    Parameters:
        fixedpt_dict: Dictionary containing tensor names and their corresponding target fixed-point data type or its canonical name
        rounding_mode: Rounding mode used for conversion into fixed point.
                       Default is "ROUND",
                       possible values: ["ROUND", "HALF_EVEN", "CEIL", "FLOOR", "UP", "DOWN", "HALF_UP", "HALF_DOWN"]
    """

    def __init__(self, fixedpt_dict, rounding_mode="ROUND"):
        super().__init__()
        self.fixedpt_dict = fixedpt_dict
        self.max_err = {}
        self.round_func = resolve_rounding_mode(rounding_mode)

    def apply(self, model: ModelWrapper):
        for tname, tdtype in self.fixedpt_dict.items():
            if (in1_t := model.get_initializer(tname)) is not None:
                if isinstance(tdtype, str):
                    tdtype = DataType[tdtype]
                current_dtype = model.get_tensor_datatype(tname)
                if current_dtype.is_fixed_point():
                    warn(f"Tensor {tname} is already a {current_dtype.get_canonical_name()} type. Recasting to {tdtype.get_canonical_name()}")

                in1_t_new = self.round_func(in1_t.astype(np.float32) / tdtype.scale_factor()) * tdtype.scale_factor()
                if (in1_t_new.max() > tdtype.max()) or (in1_t_new.min() < tdtype.min()):
                    warn(
                        f"Range of {tname} [{in1_t_new.min():.3f}, {in1_t_new.max():.3f}] greater than"
                        f"{tdtype.get_canonical_name()} [{tdtype.min():.3f}, {tdtype:.max():.3f}], clipping.")
                    in1_t_new = np.clip(in1_t_new, tdtype.min(), tdtype.max())
                model.set_tensor_datatype(tname, tdtype)
                model.set_initializer(tname, in1_t_new)

                self.max_err[tname] = np.linalg.norm(in1_t.flatten() - in1_t_new.flatten(), ord=np.inf)

        return (model, False)

class FixedPointQuantizeParams(Transformation):
    """
    Quantize model parameters to a given fixed-point representation.
    Identifies specific operations in a model (e.g., "Add", "Mul") using a filter function,
    and quantizes any non-quantized input initializers to the given fixed-point representation.
    The self.max_err dictionary stores the maximum error for each quantized input after calling.
    Parameters:
        fixedpt_dtype: The fixed-point data type or its canonical name to use for quantization.
        op_filter: A lambda function to filter operations in the model graph
                   that should be quantized. By default, it selects operations
                   of type "Add" and "Mul".
        rounding_mode: Rounding mode used for conversion into fixed point.
                       Default is "ROUND",
                       possible values: ["ROUND", "HALF_EVEN", "CEIL", "FLOOR", "UP", "DOWN", "HALF_UP", "HALF_DOWN"]
    """
    def __init__(self, fixedpt_dtype, op_filter=default_op_filter, rounding_mode="ROUND"):
        super().__init__()
        if isinstance(fixedpt_dtype, str):
            self.fixedpt_dtype = DataType[fixedpt_dtype]
        else:
            self.fixedpt_dtype = fixedpt_dtype
        self.op_filter = op_filter
        self.max_err = {}
        self.rounding_mode = rounding_mode

    def apply(self, model: ModelWrapper):
        ops = [op for op in model.graph.node if self.op_filter(op)]
        fixedpt_dict = {}
        for op in ops:
            for inp_name in op.input:
                if (model.get_initializer(inp_name)) is not None:
                    fixedpt_dict[inp_name] = self.fixedpt_dtype

        fxpdict_transform = FixedPointQuantizeParamsFromDict(fixedpt_dict, self.rounding_mode)
        model = model.transform(fxpdict_transform)
        self.max_err = fxpdict_transform.max_err

        return (model, False)
