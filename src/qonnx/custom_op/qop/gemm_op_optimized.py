########################################################################
#
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
#########################################################################

import onnx
from .helper import helper
import numpy as np

class Gemm_optimized:

    def __init__(self, node):

        gemm_node = node

        x1 = gemm_node.inputs[0]
        x2 = gemm_node.inputs[1]
        x3 = gemm_node.inputs[2]
        y = gemm_node.outputs[0]

        bias_node = gemm_node.i(2);
        bias_tensor = bias_node.inputs[0]
        bias_scale_tensor = bias_node.inputs[1]
        bias_zero_point = bias_node.inputs[2]
        bias_scale_tensor = bias_scale_tensor.values * np.ones(bias_tensor.shape)
        a = bias_tensor.values * bias_scale_tensor
        b = bias_zero_point.values * np.ones(bias_tensor.shape)
        fp32_bias_tensor = a + b
        fp32_bias_tensor = fp32_bias_tensor.astype(np.float32)

        weight_node = gemm_node.i(1).i()
        if gemm_node.i(1).i().op == "Clip":
            weight_node = gemm_node.i(1).i().i()
        weight_tensor = weight_node.inputs[0]
        weight_scale_tensor = weight_node.inputs[1]
        weight_zero_point = weight_node.inputs[2]
        weight_scale_tensor = weight_scale_tensor.values * np.ones(weight_tensor.shape)
        a = weight_tensor.values * weight_scale_tensor
        b = weight_zero_point.values * np.ones(weight_tensor.shape)
        int8_weight = a + b
        int8_weight = np.clip(int8_weight, -127, 127)
        dq_weight_scale_tensor = gemm_node.i(1).inputs[1]
        dq_weight_zero_point = gemm_node.i(1).inputs[2]
        fp32_weight = (int8_weight / (dq_weight_scale_tensor.values * np.ones(int8_weight.shape)) + dq_weight_zero_point.values * np.ones(int8_weight.shape))

        bias_name = x1.name + ".1"
        weight_name = x1.name + ".2"
        bias_tensor_1 = helper.create_initializer_tensor(name=bias_name,
                                                            tensor_array=fp32_bias_tensor,
                                                            data_type=onnx.TensorProto.FLOAT)
        weight_tensor_1 = helper.create_initializer_tensor(name=weight_name,
                                                            tensor_array=fp32_weight,
                                                            data_type=onnx.TensorProto.FLOAT)

        new_gemm_node = onnx.helper.make_node(name = gemm_node.name, op_type = "Gemm",
                                                inputs= [x1.name, weight_name, bias_name],
                                                outputs = [y.name],
                                                alpha = gemm_node.attrs["alpha"],
                                                beta = gemm_node.attrs["beta"],
                                                transB = gemm_node.attrs["transB"])


        node.i(1).i(0).inputs.clear()
        node.i(1).i(0).outputs.clear()
        node.i(1).inputs.clear()
        node.i(1).outputs.clear()

        self.node = new_gemm_node
        intializer_list = []
        intializer_list.append(weight_tensor_1)
        intializer_list.append(bias_tensor_1)
        self.intializer_list = intializer_list

    def get_node(self):
        return self.node

    def get_intializers(self):
        return self.intializer_list
