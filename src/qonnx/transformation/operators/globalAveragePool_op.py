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

class GlobalAveragePool:

    def __init__(self, node, aecg_zendnn_opt, remove_relu):

        golbal_average_pool_node = node
        x_name = golbal_average_pool_node.inputs[0].name
        y_name = golbal_average_pool_node.outputs[0].name

        if helper.is_parent_exist(golbal_average_pool_node, 0, 0) and golbal_average_pool_node.i().op == "DequantizeLinear":
            if helper.is_parent_exist(golbal_average_pool_node, 0, 0):
                parent_dql_node = golbal_average_pool_node.i()
            else:
                print("************* ERROR ****************** Please check 1st parent of GlobalAveragePool, ", golbal_average_pool_node.name, " parent DNE")

            x_scale_name = node.name + "x_scale"
            x_scale_tensor = helper.create_initializer_tensor(name=x_scale_name,
                                                              tensor_array=parent_dql_node.inputs[1].values,
                                                              data_type=onnx.TensorProto.FLOAT)
            x_zp_name = node.name + "x_zp"

            is_input_s8 = True

            if helper.is_parent_exist(parent_dql_node, 0, 0):
                if aecg_zendnn_opt:
                    x_zp_tensor = helper.create_initializer_tensor(name=x_zp_name,
                                                                    tensor_array=parent_dql_node.inputs[2].values,
                                                                    data_type=onnx.TensorProto.UINT8)
                else:
                    second_parent = parent_dql_node.i()
                    if second_parent.op == "Relu":
                        if helper.is_parent_exist(second_parent, 0, 0) and second_parent.i().op == "QuantizeLinear":
                            third_parent = second_parent.i()
                            if third_parent.inputs[2].values.dtype == np.int8:
                                x_zp_tensor = helper.create_initializer_tensor(name=x_zp_name,
                                                                                tensor_array=third_parent.inputs[2].values,
                                                                                data_type=onnx.TensorProto.INT8)
                                is_input_s8 = True
                            else:
                                x_zp_tensor = helper.create_initializer_tensor(name=x_zp_name,
                                                                                tensor_array=third_parent.inputs[2].values,
                                                                                data_type=onnx.TensorProto.UINT8)
                                is_input_s8 = False
                    else:
                        if parent_dql_node.i().inputs[2].values.dtype == np.int8:
                            x_zp_tensor = helper.create_initializer_tensor(name=x_zp_name,
                                                                            tensor_array=parent_dql_node.inputs[2].values,
                                                                            data_type=onnx.TensorProto.INT8)
                            is_input_s8 = True
                        else:
                            x_zp_tensor = helper.create_initializer_tensor(name=x_zp_name,
                                                                            tensor_array=parent_dql_node.inputs[2].values,
                                                                            data_type=onnx.TensorProto.UINT8)
                            is_input_s8 = False
            else:
                print("************* ERROR ****************** Please check 2nd parent of GlobalAveragePool, ", golbal_average_pool_node.name, " 1st parent of ", parent_dql_node, " parent DNE")

            if parent_dql_node.i().i().op == "Relu" and parent_dql_node.i().i().i().i().inputs[2].values.dtype == np.int8:
                if remove_relu:
                    x_name = parent_dql_node.inputs[0].name
                else:
                    third_parent_relu = parent_dql_node.i().i()
                    if third_parent_relu.i().op == "Conv" or third_parent_relu.i().op == "Add":
                        x_name = third_parent_relu.outputs[0].name
                    else:
                        x_name = (third_parent_relu.o()).outputs[0].name
            else:
                x_name = parent_dql_node.inputs[0].name

        if helper.is_child_present(node, 0, 0) and golbal_average_pool_node.o().op == "QuantizeLinear":
            child_ql_node = golbal_average_pool_node.o()

            y_scale_name = node.name + "y_scale"
            y_scale_tensor = helper.create_initializer_tensor(name=y_scale_name,
                                                              tensor_array=child_ql_node.inputs[1].values,
                                                              data_type=onnx.TensorProto.FLOAT)
            y_zp_name = node.name + "y_zp"

            if aecg_zendnn_opt:
                y_zp_tensor = helper.create_initializer_tensor(name=y_zp_name,
                                                                    tensor_array=child_ql_node.inputs[2].values,
                                                                    data_type=onnx.TensorProto.UINT8)
            else:
                if is_input_s8:
                    y_zp_tensor = helper.create_initializer_tensor(name=y_zp_name,
                                                                    tensor_array=child_ql_node.inputs[2].values,
                                                                    data_type=onnx.TensorProto.INT8)
                else:
                    y_zp_tensor = helper.create_initializer_tensor(name=y_zp_name,
                                                                    tensor_array=child_ql_node.inputs[2].values,
                                                                    data_type=onnx.TensorProto.UINT8)

            y_name = child_ql_node.outputs[0].name

        kwargs = {}
        kwargs["domain"] = 'com.microsoft'
        new_average_pool_node = onnx.helper.make_node(name = golbal_average_pool_node.name, op_type = "QLinearGlobalAveragePool",
                                                        inputs = [x_name, x_scale_name, x_zp_name, y_scale_name, y_zp_name],
                                                        outputs = [y_name],
                                                        channels_last = 0,**kwargs)

        intializer_list = []
        intializer_list.append(x_scale_tensor)
        intializer_list.append(x_zp_tensor)
        intializer_list.append(y_scale_tensor)
        intializer_list.append(y_zp_tensor)
        self.intializer_list = intializer_list

        self.node = new_average_pool_node

    def get_node(self):
        return self.node

    def get_intializers(self):
        return self.intializer_list