# Copyright (c) 2023 Advanced Micro Devices, Inc.
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
import onnx.parser as oprs

import qonnx.custom_op.general as general
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.base import CustomOp
from qonnx.custom_op.registry import getCustomOp


class AttrTestOp(CustomOp):
    def get_nodeattr_types(self):
        my_attrs = {"tensor_attr": ("t", True, np.asarray([])), "strings_attr": ("strings", True, [""])}
        return my_attrs

    def make_shape_compatible_op(self, model):
        param_tensor = self.get_nodeattr("tensor_attr")
        return super().make_const_shape_op(param_tensor.shape)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # data type stays the same
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def execute_node(self, context, graph):
        node = self.onnx_node
        param_tensor = self.get_nodeattr("tensor_attr")
        context[node.output[0]] = param_tensor

    def verify_node(self):
        pass


def test_attr():
    general.custom_op["AttrTestOp"] = AttrTestOp
    ishp = (1, 10)
    wshp = (1, 3)
    oshp = wshp
    ishp_str = str(list(ishp))
    oshp_str = str(list(oshp))
    wshp_str = str(list(wshp))
    w = np.asarray([1, -2, 3], dtype=np.int8)
    strarr = np.array2string(w, separator=", ")
    w_str = strarr.replace("[", "{").replace("]", "}").replace(" ", "")
    tensor_attr_str = f"int8{wshp_str} {w_str}"
    strings_attr = ["a", "bc", "def"]

    input = f"""
    <
        ir_version: 7,
        opset_import: ["" : 9]
    >
    agraph (float{ishp_str} in0) => (int8{oshp_str} out0)
    {{
        out0 = qonnx.custom_op.general.AttrTestOp<
            tensor_attr={tensor_attr_str}
        >(in0)
    }}
    """
    model = oprs.parse_model(input)
    model = ModelWrapper(model)
    inst = getCustomOp(model.graph.node[0])

    w_prod = inst.get_nodeattr("tensor_attr")
    assert (w_prod == w).all()
    w = w - 1
    inst.set_nodeattr("tensor_attr", w)
    w_prod = inst.get_nodeattr("tensor_attr")
    assert (w_prod == w).all()

    inst.set_nodeattr("strings_attr", strings_attr)
    strings_attr_prod = inst.get_nodeattr("strings_attr")
    assert strings_attr_prod == strings_attr
    strings_attr_prod[0] = "test"
    inst.set_nodeattr("strings_attr", strings_attr_prod)
    assert inst.get_nodeattr("strings_attr") == ["test"] + strings_attr[1:]
