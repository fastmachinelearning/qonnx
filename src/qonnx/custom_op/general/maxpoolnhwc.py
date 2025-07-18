# Copyright (c) 2020 Xilinx, Inc.
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

# TODO deprecate in favor of the channels_last MaxPool

import numpy as np
from onnx import TensorProto, helper

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import qonnx_make_model


def compute_pool_output_dim(ifm_dim, k, stride, pad=0, ceil_mode=0):
    "Return spatial output dimension size for pooling with given params."
    if ceil_mode:
        return int(np.ceil(((ifm_dim + 2 * pad - k) / stride) + 1))
    else:
        return int(np.floor(((ifm_dim + 2 * pad - k) / stride) + 1))


class MaxPoolNHWC(CustomOp):
    # a MaxPool node, but using the NHWC data layout

    def get_nodeattr_types(self):
        # no specific attributes for MaxPoolNHWC
        # attributes below are identical to the standard ONNX MaxPool op:
        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool
        return {
            "kernel_shape": ("ints", True, []),
            "pads": ("ints", True, []),
            "strides": ("ints", True, []),
            "ceil_mode": ("i", False, 0),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node
        iname = node.input[0]
        ishape = model.get_tensor_shape(iname)
        kernel_shape = self.get_nodeattr("kernel_shape")
        pads = self.get_nodeattr("pads")
        strides = self.get_nodeattr("strides")
        ceil_mode = self.get_nodeattr("ceil_mode")
        assert len(kernel_shape) == 2, "Non-2D MaxPoolNHWC not supported"
        assert pads[0] == pads[2], "Uneven padding not supported"
        assert pads[1] == pads[3], "Uneven padding not supported"
        (n, hi, wi, c) = ishape
        ho = compute_pool_output_dim(hi, kernel_shape[0], strides[0], pads[0], ceil_mode)
        wo = compute_pool_output_dim(wi, kernel_shape[1], strides[1], pads[1], ceil_mode)
        oshape = (n, ho, wo, c)
        return super().make_const_shape_op(oshape)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # data type stays the same
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def execute_node(self, context, graph):
        node = self.onnx_node
        inp_name = node.input[0]
        out_name = node.output[0]
        inp = context[inp_name]
        dummy_out = context[out_name]
        # convert i/o NHWC -> NCHW
        inp = np.transpose(inp, (0, 3, 1, 2))
        dummy_out = np.transpose(dummy_out, (0, 3, 1, 2))
        # execute as regular MaxPool
        orig_domain = node.domain
        node.domain = ""
        node.op_type = "MaxPool"
        inp_vi = helper.make_tensor_value_info(inp_name, TensorProto.FLOAT, inp.shape)
        out_vi = helper.make_tensor_value_info(out_name, TensorProto.FLOAT, dummy_out.shape)
        tmp_graph = helper.make_graph(nodes=[node], name="tmp_graph", inputs=[inp_vi], outputs=[out_vi])
        opset_version = self.onnx_opset_version
        opset_imports = [helper.make_opsetid("", opset_version)]
        onnx_kwargs = {"opset_imports": opset_imports}
        tmp_model = qonnx_make_model(tmp_graph, producer_name="finn", **onnx_kwargs)
        tmp_model = ModelWrapper(tmp_model)
        new_ctx = {inp_name: inp}
        from qonnx.core.onnx_exec import execute_onnx

        ret = execute_onnx(tmp_model, new_ctx)
        # restore original node props
        node.domain = orig_domain
        node.op_type = "MaxPoolNHWC"
        outp = ret[out_name]
        # convert output NCHW -> NHWC
        outp = np.transpose(outp, (0, 2, 3, 1))
        context[out_name] = outp

    def verify_node(self):
        info_messages = []
        return info_messages
