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
from fxpmath import Fxp

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

def default_op_filter(op):
    return op.op_type in ["Add", "Mul"]

class FixedPointQuantizeParamsFromDict(Transformation):
    """
    Quantize model parameters to a given fixed-point representation.
    Uses
    The self.max_err dictionary stores the maximum error for each quantized input after calling.
    Parameters:
        fixedpt_dtype: The fixed-point data type to use for quantization.
        op_filter: A lambda function to filter operations in the model graph
                   that should be quantized. By default, it selects operations
                   of type "Add" and "Mul".
    """

    def __init__(self, fixedpt_dict):
        super().__init__()
        self.fixedpt_dict = fixedpt_dict
        self.max_err = {}

    def apply(self, model: ModelWrapper):
        for tname, tdtype in self.fixedpt_dict.items():
            if (in1_t := model.get_initializer(tname)) is not None:
                fixpt = Fxp(None, signed=True, n_word=tdtype.bitwidth(), n_frac=tdtype.frac_bits())
                model.set_tensor_datatype(tname, tdtype)
                fixpt.set_val(in1_t)
                # .astype() twice to workaround a bug in fxpmath
                # (typecast only works for ndarrays and somehow the
                # val stops being an ndarray at some point...)
                in1_t_new = np.asarray(fixpt.astype(np.float32), dtype=np.float32)
                model.set_initializer(tname, in1_t_new)

                rel_err = np.abs(in1_t.flatten() - in1_t_new.flatten()) / np.abs(in1_t.flatten())
                
                self.max_err[tname] = rel_err.max()
        return (model, False)

class FixedPointQuantizeParams(Transformation):
    """
    Quantize model parameters to a given fixed-point representation.
    Identifies specific operations in a model (e.g., "Add", "Mul") using a filter function,
    and quantizes any non-quantized input initializers to the given fixed-point representation.
    The self.max_err dictionary stores the maximum error for each quantized input after calling.
    Parameters:
        fixedpt_dtype: The fixed-point data type to use for quantization.
        op_filter: A lambda function to filter operations in the model graph
                   that should be quantized. By default, it selects operations
                   of type "Add" and "Mul".

    """

    def __init__(self, fixedpt_dtype, op_filter=default_op_filter):
        super().__init__()
        self.fixedpt_dtype = fixedpt_dtype
        self.n_word = fixedpt_dtype.bitwidth()
        self.n_frac = fixedpt_dtype.frac_bits()
        self.op_filter = op_filter
        self.max_err = {}

    def apply(self, model: ModelWrapper):
        ops = [op for op in model.graph.node if self.op_filter(op)]
        fixpt = Fxp(0.0, signed=True, n_word=self.n_word, n_frac=self.n_frac)
        modified = False
        for op in ops:
            for inp_name in op.input:
                if (in1_t := model.get_initializer(inp_name)) is not None:
                    current_dtype = model.get_tensor_datatype(inp_name)
                    if current_dtype is None or (not current_dtype.is_fixed_point()):
                        fixpt.set_val(in1_t)
                        # double .astype to workaround bug in fixpt
                        # for 0d numpy arrays it seems .astype is skipped
                        in1_t_new = fixpt.astype(np.float32).astype(np.float32)
                        model.set_initializer(inp_name, in1_t_new)
                        model.set_tensor_datatype(inp_name, self.fixedpt_dtype)
                        self.max_err[inp_name] = np.linalg.norm(in1_t.flatten() - in1_t_new.flatten(), ord=np.inf)

        return (model, modified)
