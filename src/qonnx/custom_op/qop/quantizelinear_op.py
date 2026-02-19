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

class QuantizeLinear:

    def __init__(self, node):
        ql_node = node

        x_name = ql_node.inputs[0].name
        flag = False
        if helper.is_child_present(node, 0, 0) and node.o().op == "DequantizeLinear":
            if  helper.is_child_present(node.o(), 0, 0) and node.o().o().op == "Conv":
                if  helper.is_child_present(node.o().o(), 0, 0) and node.o().o().o().op == "Reshape":
                    flag = True
                    x_tensor = helper.create_initializer_tensor(name = x_name,tensor_array = ql_node.inputs[0].values, data_type = onnx.TensorProto.FLOAT)
            elif helper.is_child_present(node.o(), 0, 0) and node.o().o().op == "Gemm":
                flag = True
                x_tensor = helper.create_initializer_tensor(name = x_name,tensor_array = ql_node.inputs[0].values, data_type = onnx.TensorProto.FLOAT)

        y_scale_name = ql_node.inputs[1].name
        y_scale_value = ql_node.inputs[1].values
        y_scale_tensor = helper.create_initializer_tensor(name = y_scale_name,tensor_array = y_scale_value, data_type = onnx.TensorProto.FLOAT)

        y_zp_name = ql_node.inputs[2].name
        y_zp_value = ql_node.inputs[2].values
        if ql_node.inputs[2].dtype == np.int8:
            y_zp_tensor = helper.create_initializer_tensor(name=y_zp_name,
                                                            tensor_array=y_zp_value,
                                                            data_type = onnx.TensorProto.INT8)
        elif ql_node.inputs[2].dtype == np.uint8:
            y_zp_tensor = helper.create_initializer_tensor(name=y_zp_name,
                                                            tensor_array=y_zp_value,
                                                            data_type = onnx.TensorProto.UINT8)

        y_name = ql_node.outputs[0].name

        quantizelinear_node = onnx.helper.make_node(name = ql_node.name, op_type = "QuantizeLinear", inputs = [x_name, y_scale_name, y_zp_name], outputs = [y_name])

        self.node = quantizelinear_node

        intializer_list = []
        if flag:
            intializer_list.append(x_tensor)
        intializer_list.append(y_scale_tensor)
        intializer_list.append(y_zp_tensor)
        self.intializer_list = intializer_list

    def get_node(self):
        return self.node

    def get_intializers(self):
        return self.intializer_list