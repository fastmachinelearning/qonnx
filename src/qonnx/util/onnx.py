# Copyright (c) 2022 Advanced Micro Devices, Inc.
# Copyright (c) 2020-21 Xilinx, Inc.
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
import onnx

import qonnx.core.data_layout as DataLayout

# optypes with the (elementwise) monotonic property
monotonic_optypes = {
    "Identity",
    "Relu",
    "LeakyRelu",
    "Clip",
    "Selu",
    "Celu",
    "Elu",
    "Sigmoid",
    "HardSigmoid",
    "Tanh",
    "Softplus",
    "Exp",
    "Log",
    "Sqrt",
    "Erf",
    "Floor",
    "Ceil",
    "Round",
    "Sign",
}

# optypes that operate in an elementwise fashion
# (with numpy-style broadcasting when shapes mismatch for binary ops)
eltwise_optypes = monotonic_optypes | {
    "Quant",
    "Mul",
    "Div",
    "Sub",
    "Add",
    "Mod",
    "And",
    "Or",
    "Xor",
    "Equal",
    "Less",
    "LessOrEqual",
    "Greater",
    "GreaterOrEqual",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
    "Maximum",
    "Minimum",
    "BitShift",
    "Pow",
}


def is_eltwise_monotonic_optype(optype):
    "Checks whether given ONNX optype is a monotonic elementwise op."
    return optype in monotonic_optypes


def is_eltwise_optype(optype):
    "Checks whether given ONNX optype is an elementwise op."
    return optype in eltwise_optypes


def valueinfo_to_tensor(vi):
    """Creates an all-zeroes numpy tensor from a ValueInfoProto."""

    dims = [x.dim_value for x in vi.type.tensor_type.shape.dim]
    return np.zeros(dims, dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[vi.type.tensor_type.elem_type])


def nchw_to_nhwc(t, model, idx, reverse=False):
    """Converts between NCHW <-> NHWC layouts for tensor t by inserting a transpose.
    If reverse=False, t is assumed NCHW and we insert transpose to convert NCHW -> NHWC
    If reverse=True, t is assumed NHWC and we insert transpose to convert NHWC -> NCHW.
    """
    graph = model.graph
    # create new NHWC tensor
    t_shape = model.get_tensor_shape(t)
    bs = t_shape[0]
    ch = t_shape[1]
    height = t_shape[2]
    width = t_shape[3]
    t_trans = onnx.helper.make_tensor_value_info(
        model.make_new_valueinfo_name(),
        onnx.TensorProto.FLOAT,
        (bs, height, width, ch),  # NHWC
    )
    graph.value_info.append(t_trans)
    dt = model.get_tensor_datatype(t)
    t_trans = t_trans.name
    model.set_tensor_datatype(t_trans, dt)
    model.set_tensor_layout(t_trans, DataLayout.NHWC)
    # NCHW <-> NHWC transpose
    if reverse:
        t_trans_node = onnx.helper.make_node("Transpose", [t_trans], [t], perm=[0, 3, 1, 2])
    else:
        t_trans_node = onnx.helper.make_node("Transpose", [t], [t_trans], perm=[0, 2, 3, 1])
    graph.node.insert(idx, t_trans_node)
    return t_trans
