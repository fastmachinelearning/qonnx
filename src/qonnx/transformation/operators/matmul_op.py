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

class MatMul:

    def __init__(self, node):
        matlmul_node = node

        if helper.is_parent_exist(matlmul_node, 0, 0):
            x_DQL_node = matlmul_node.i()
        else:
             print("**************** ERROR *******************  Matmul node ", matlmul_node.name, " input(0,0) DNE")

        if helper.is_parent_exist(matlmul_node, 1, 0):
            w_DQL_node = matlmul_node.inputs[1].inputs[0]
        else:
            print("************* ERROR ************************ Please check the Matmul node ", matlmul_node.name, " the input(1,0) DNE")

        if helper.is_parent_exist(x_DQL_node, 0, 0):
            x_QL_node = x_DQL_node.i()
        else:
             print("**************** ERROR ******************* ", x_DQL_node.name, " input(0,0) DNE Please check")

        x_scale_tensor = x_DQL_node.inputs[1]
        x_scale = x_scale_tensor.values
        x_zp_tensor = x_DQL_node.inputs[2]

        w_scale_tensor = w_DQL_node.inputs[1]
        w_scale = w_scale_tensor.values
        w_zp_tensor = w_DQL_node.inputs[2]

        if helper.is_child_present(matlmul_node, 0, 0):
            if (matlmul_node.o().op == "QuantizeLinear"):
                y_QL_node = matlmul_node.o()
                y_scale_tensor = y_QL_node.inputs[1]
                y_scale = y_scale_tensor.values
                y_zp_tensor = y_QL_node.inputs[2]
            else:
                print("*********************** ERROR output of Matmul node ", matlmul_node.name, " is not QuantizeLinear ***********************")
        else:
            print(matlmul_node.name, " output(0,0) DNE")

        quantized_weight_tensor_original = w_DQL_node.inputs[0].values
        new_shape = quantized_weight_tensor_original.shape + (1,1)
        a1 = np.reshape(quantized_weight_tensor_original, new_shape)
        quantized_weight_tensor = np.transpose(a1, (1,0,2,3))

        if x_QL_node.i().op == "DequantizeLinear" and x_QL_node.i().i().op == "QuantizeLinear":
            x_name = x_QL_node.i().i().outputs[0].name
        else:
            x_name = x_QL_node.outputs[0].name

        y_name = matlmul_node.o().outputs[0].name

        x_scale_name = matlmul_node.name + "_X_SCALE"
        x_scale_value = x_scale
        x_scale_tensor = helper.create_initializer_tensor(name=x_scale_name,
                                                          tensor_array=x_scale_value,
                                                          data_type=onnx.TensorProto.FLOAT)

        x_zp_name = matlmul_node.name + "_X_ZERO_POINT"
        x_zp_value = x_zp_tensor.values

        if (x_QL_node.op == "QuantizeLinear" and x_QL_node.inputs[2].dtype == np.int8):
            x_zp_tensor = helper.create_initializer_tensor(name=x_zp_name,
                                                            tensor_array=x_zp_value,
                                                            data_type=onnx.TensorProto.INT8)
        elif (x_QL_node.op == "QuantizeLinear" and x_QL_node.inputs[2].dtype == np.uint8):
            x_zp_tensor = helper.create_initializer_tensor(name=x_zp_name,
                                                            tensor_array=x_zp_value,
                                                            data_type=onnx.TensorProto.UINT8)

        w_name = matlmul_node.inputs[1].name
        w_value = quantized_weight_tensor
        w_tensor = helper.create_initializer_tensor(name=w_name,
                                                    tensor_array=w_value,
                                                    data_type=onnx.TensorProto.INT8)

        w_scale_name = matlmul_node.name + "_W_SCALE"
        w_scale_value = w_scale
        w_scale_tensor = helper.create_initializer_tensor(name=w_scale_name,
                                                          tensor_array=w_scale_value,
                                                          data_type=onnx.TensorProto.FLOAT)

        w_zp_name = matlmul_node.name + "_W_ZERO_POINT"
        w_zp_value = w_zp_tensor.values
        w_zp_tensor = helper.create_initializer_tensor(name=w_zp_name,
                                                       tensor_array=w_zp_value,
                                                       data_type=onnx.TensorProto.INT8)

        y_scale_name = matlmul_node.name + "_Y_SCALE"
        y_scale_value = y_scale
        y_scale_tensor = helper.create_initializer_tensor(name=y_scale_name,
                                                          tensor_array=y_scale_value,
                                                          data_type=onnx.TensorProto.FLOAT)

        y_zp_name = matlmul_node.name + "_Y_ZERO_POINT"
        y_zp_value = y_zp_tensor.values

        if y_zp_tensor.dtype == np.int8:
                y_zp_tensor = helper.create_initializer_tensor(name=y_zp_name,
                                                            tensor_array=y_zp_value,
                                                            data_type=onnx.TensorProto.INT8)
        elif y_zp_tensor.dtype == np.uint8:
                y_zp_tensor = helper.create_initializer_tensor(name=y_zp_name,
                                                        tensor_array=y_zp_value,
                                                        data_type=onnx.TensorProto.UINT8)

        qlinearconv_node = onnx.helper.make_node(name = matlmul_node.name, op_type = "QLinearConv",
                                                inputs = [x_name, x_scale_name, x_zp_name, w_name, w_scale_name, w_zp_name, y_scale_name, y_zp_name],
                                                outputs = [y_name], dilations = [1,1], group = 1,
                                                kernel_shape = [1,1], pads = [0,0,0,0], strides = [1,1])


        self.node = qlinearconv_node

        intializer_list = []
        intializer_list.append(x_scale_tensor)
        intializer_list.append(x_zp_tensor)
        intializer_list.append(w_tensor)
        intializer_list.append(w_scale_tensor)
        intializer_list.append(w_zp_tensor)
        intializer_list.append(y_scale_tensor)
        intializer_list.append(y_zp_tensor)
        self.intializer_list = intializer_list

    def get_node(self):
        return self.node

    def get_intializers(self):
        return self.intializer_list