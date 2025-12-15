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

class DequantizeLinear:

    def __init__(self, node, remove_relu):

        dql_node = node

        x_name = dql_node.inputs[0].name

        if helper.is_parent_exist(dql_node, 0, 0):
            if dql_node.i().op == "QuantizeLinear":
                ql_node = dql_node.i()
                if helper.is_parent_exist(ql_node,0, 0):
                    if ql_node.i().op == "Relu":
                        relu_node = ql_node.i()
                        if remove_relu:
                            x_name = ql_node.outputs[0].name
                        else:
                            x_name = relu_node.outputs[0].name
                else:
                    print("*************** WARNING *********************** Please check parent of QL node", ql_node.name, " ignore if pattern is correct")
        else:
            print("*************** WARNING *********************** Please check parent of DQL node", dql_node.name, " ignore if pattern is correct")
        self.initializers = []

        if len(dql_node.inputs[0].inputs) == 0:
            if dql_node.inputs[0].dtype == np.uint8:
                input_tensor = helper.create_initializer_tensor(name= dql_node.inputs[0].name,
                                                                tensor_array=dql_node.inputs[0].values,
                                                                data_type=onnx.TensorProto.UINT8)
            elif dql_node.inputs[0].dtype == np.int8:
                input_tensor = helper.create_initializer_tensor(name= dql_node.inputs[0].name,
                                                                tensor_array=dql_node.inputs[0].values,
                                                                data_type=onnx.TensorProto.INT8)
            elif dql_node.inputs[0].dtype == np.int32:
                input_tensor = helper.create_initializer_tensor(name= dql_node.inputs[0].name,
                                                                tensor_array=dql_node.inputs[0].values,
                                                                data_type=onnx.TensorProto.INT32)
            self.initializers.append(input_tensor)

        x_scale_name = dql_node.inputs[1].name
        x_scale_value = dql_node.inputs[1].values
        x_scale_tensor = helper.create_initializer_tensor(name=x_scale_name,tensor_array=x_scale_value,data_type=onnx.TensorProto.FLOAT)

        x_zp_name = dql_node.inputs[2].name
        x_zp_value = dql_node.inputs[2].values

        if dql_node.inputs[2].dtype == np.uint8:
            x_zp_tensor = helper.create_initializer_tensor(name=x_zp_name,
                                                        tensor_array=x_zp_value,
                                                        data_type=onnx.TensorProto.UINT8)
        if dql_node.inputs[2].dtype == np.int32:
            x_zp_tensor = helper.create_initializer_tensor(name=x_zp_name,
                                                       tensor_array=x_zp_value,
                                                       data_type=onnx.TensorProto.INT32)
        elif dql_node.inputs[2].dtype == np.int8:
            x_zp_tensor = helper.create_initializer_tensor(name=x_zp_name,
                                                        tensor_array=x_zp_value,
                                                        data_type=onnx.TensorProto.INT8)

        y_name = dql_node.outputs[0].name

        dequantizelinear_node = onnx.helper.make_node(name = dql_node.name,
                                                    op_type = "DequantizeLinear",
                                                    inputs = [x_name, x_scale_name, x_zp_name],
                                                    outputs = [y_name])

        self.node = dequantizelinear_node

        self.initializers.append(x_scale_tensor)
        self.initializers.append(x_zp_tensor)

    def get_node(self):
        return self.node

    def get_intializers(self):
        return self.initializers
