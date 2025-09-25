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

import onnx.parser as oprs

import qonnx.custom_op.general as general
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.base import CustomOp
from qonnx.custom_op.registry import getCustomOp


class VerTestOp_v1(CustomOp):
    def get_nodeattr_types(self):
        my_attrs = {"v1_attr": ("i", True, 0)}
        return my_attrs

    def make_shape_compatible_op(self, model):
        ishape = model.get_tensor_shape(self.onnx_node.input[0])
        return super().make_const_shape_op(ishape)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # data type stays the same
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def execute_node(self, context, graph):
        node = self.onnx_node
        context[node.output[0]] = context[node.input[0]]

    def verify_node(self):
        pass


class VerTestOp_v2(VerTestOp_v1):
    def get_nodeattr_types(self):
        my_attrs = {"v2_attr": ("i", True, 0)}
        return my_attrs


class VerTestOp_v3(VerTestOp_v2):
    def get_nodeattr_types(self):
        my_attrs = {"v3_attr": ("i", True, 0)}
        return my_attrs


def make_vertest_model(vertest_ver):
    ishp = (1, 10)
    oshp = ishp
    ishp_str = str(list(ishp))
    oshp_str = str(list(oshp))
    input = f"""
    <
        ir_version: 7,
        opset_import: ["" : 9, "qonnx.custom_op.general" : {vertest_ver}]
    >
    agraph (float{ishp_str} in0) => (float{oshp_str} out0)
    {{
        out0 = qonnx.custom_op.general.VerTestOp<
            v{vertest_ver}_attr={vertest_ver}
        >(in0)
    }}
    """
    model = oprs.parse_model(input)
    model = ModelWrapper(model)
    return model


def test_customop_version():
    general.custom_op["VerTestOp"] = VerTestOp_v1
    general.custom_op["VerTestOp_v2"] = VerTestOp_v2
    general.custom_op["VerTestOp_v3"] = VerTestOp_v3
    for ver in [1, 2, 3]:
        model = make_vertest_model(ver)
        # explicitly specify onnx_opset_version in getCustomOp
        inst = getCustomOp(model.graph.node[0], onnx_opset_version=ver)
        assert inst.get_nodeattr(f"v{ver}_attr") == ver
        # now use ModelWrapper.get_customop_wrapper for implicit
        # fetching of op version
        inst = model.get_customop_wrapper(model.graph.node[0])
        assert inst.get_nodeattr(f"v{ver}_attr") == ver
