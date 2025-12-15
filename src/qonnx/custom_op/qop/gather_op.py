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

class Gather:

    def __init__(self, node):

        gather_node = node
        # --------------------------------
        # For QCDQ / QDQ model, this case:
        # QuantizeLinear
        #       | (0)
        #     Gather ---------- (1) Input
        #       |
        # --------------------------------
        gather_parent_node = node
        quantized_data_tensor = node
        if helper.is_parent_exist(gather_node, 0, 0):
            gather_parent_node = node.i(0)
            if len(gather_parent_node.inputs) > 1 and helper.is_constant_tensor(gather_parent_node.inputs[1]):
                quantized_data_tensor = gather_parent_node.inputs[1].values

        if helper.is_constant_tensor(gather_parent_node.inputs[0]):
            if gather_parent_node.op == "QuantizeLinear":
                X_DQL_node = gather_parent_node
                dequantized_data_tensor = X_DQL_node.inputs[0]
                data_scale_tensor = X_DQL_node.inputs[1]
                data_zero_point_tensor = X_DQL_node.inputs[2]

                data_scale_tensor = data_scale_tensor.values * np.ones(dequantized_data_tensor.shape)
                a = dequantized_data_tensor.values / data_scale_tensor
                b = data_zero_point_tensor.values * np.ones(dequantized_data_tensor.shape)
                quantized_data_tensor = a + b
                quantized_data_tensor = quantized_data_tensor.astype(np.int8)

        else:
            if gather_parent_node.op == "QuantizeLinear":
                X_QL_node = gather_parent_node
                quantized_data_tensor = X_QL_node.inputs[1].values

        data_tensor = helper.create_initializer_tensor(name=gather_node.inputs[0].name,
                                                            tensor_array=quantized_data_tensor,
                                                            data_type=onnx.TensorProto.INT8)

        if helper.is_constant_tensor(gather_parent_node.inputs[0]):
            data_tensor = helper.create_initializer_tensor(name=gather_node.inputs[0].name,
                                                            tensor_array=quantized_data_tensor,
                                                            data_type=onnx.TensorProto.INT8)
        if helper.is_constant_tensor(gather_node.inputs[1]):
            if gather_node.inputs[1].dtype == "int64":
                indices_tensor = helper.create_initializer_tensor(name=gather_node.inputs[1].name,
                                                                tensor_array=gather_node.inputs[1].values,
                                                                data_type=onnx.TensorProto.INT64)
            else:
                print("ERROR check data type in Gather node ", gather_node.name)

        new_gather_node = onnx.helper.make_node(name = gather_node.name, op_type = "Gather",
                                                inputs= [data_tensor.name, gather_node.inputs[1].name],
                                                 outputs = [gather_node.outputs[0].name],
                                                 axis = 0)
        if helper.is_constant_tensor(gather_parent_node.inputs[0]):
            new_gather_node = onnx.helper.make_node(name = gather_node.name, op_type = "Gather",
                                                inputs= [data_tensor.name, gather_node.inputs[1].name],
                                                 outputs = [gather_node.outputs[0].name],
                                                 axis = 0)
        elif helper.is_constant_tensor(gather_node.inputs[1]):
            new_gather_node = onnx.helper.make_node(name = gather_node.name, op_type = "Gather",
                                                inputs= [gather_node.inputs[0].name,indices_tensor.name],
                                                 outputs = [gather_node.outputs[0].name],
                                                 axis = gather_node.attrs['axis'])

        self.node = new_gather_node

        intializer_list = []
        if helper.is_constant_tensor(gather_parent_node.inputs[0]):
            intializer_list.append(data_tensor)
        elif helper.is_constant_tensor(gather_node.inputs[1]):
            intializer_list.append(indices_tensor)
        self.intializer_list = intializer_list

    def get_node(self):
        return self.node

    def get_intializers(self):
        return self.intializer_list

