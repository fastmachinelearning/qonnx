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

class Concat:

    def __init__(self, node, is_all_concat_input_dql):

        concat_node = node

        number_of_inputs = len(concat_node.inputs)

        zp_value_list = []
        zp_name_list = []
        scale_values_list = []
        scale_name_list = []
        input_tensor_names = []

        intializer_list = []
        input_names = []

        for i in range(number_of_inputs):
            if is_all_concat_input_dql:
                parent_dql_node = concat_node.inputs[i].inputs[0]
                scale_values_list.append(parent_dql_node.inputs[1].values)
                scale_name_list.append(parent_dql_node.inputs[1].name)
                zp_value_list.append(parent_dql_node.inputs[2].values)
                zp_name_list.append(parent_dql_node.inputs[2].name)
                input_tensor_names.append(parent_dql_node.inputs[0].name)
            else:
                input_tensor_names.append(concat_node.inputs[i].name)
                if len(concat_node.inputs[i].inputs) == 0:
                    c_input = helper.create_initializer_tensor(name=concat_node.inputs[i].name,
                                                               tensor_array=concat_node.inputs[i].values,
                                                               data_type=onnx.TensorProto.INT64)
                    intializer_list.append(c_input)
                    self.intializer_list = intializer_list

        if is_all_concat_input_dql:
            for i in range(number_of_inputs):
                scale_tesnor = helper.create_initializer_tensor(name=scale_name_list[i],
                                                                    tensor_array=scale_values_list[i],
                                                                    data_type=onnx.TensorProto.FLOAT)
                zp_tensor = helper.create_initializer_tensor(name=zp_name_list[i],
                                                                    tensor_array=zp_value_list[i],
                                                                    data_type=onnx.TensorProto.UINT8)
                intializer_list.append(scale_tesnor)
                intializer_list.append(zp_tensor)

        if helper.is_child_present(concat_node, 0, 0) and concat_node.o().op == "DequantizeLinear" and is_all_concat_input_dql:
            y_ql_node = concat_node.o()
            y_name = y_ql_node.outputs[0].name
        else:
            y_name = concat_node.outputs[0].name

        if helper.is_child_present(concat_node, 0, 0) and concat_node.o().op == "DequantizeLinear" and is_all_concat_input_dql:
            y_scale_name = y_ql_node.inputs[1].name
            y_scale_value = y_ql_node.inputs[1].values
            y_zp_name = y_ql_node.inputs[2].name
            y_zp_value = y_ql_node.inputs[2].values

            y_scale_tensor = helper.create_initializer_tensor(name=y_scale_name,
                                                            tensor_array=y_scale_value,
                                                            data_type=onnx.TensorProto.FLOAT)
            y_zp_tensor = helper.create_initializer_tensor(name=y_zp_name,
                                                            tensor_array=y_zp_value,
                                                            data_type=onnx.TensorProto.UINT8)

            intializer_list.append(y_scale_tensor)
            intializer_list.append(y_zp_tensor)
            self.intializer_list = intializer_list

            input_names.append(y_scale_tensor.name)
            input_names.append(y_zp_tensor.name)

        for i in range(number_of_inputs):
            input_names.append(input_tensor_names[i])
            if len(scale_name_list)>0 and len(zp_name_list)>0:
                input_names.append(scale_name_list[i])
                input_names.append(zp_name_list[i])

        kwargs = {}
        kwargs["domain"] = 'com.microsoft'

        if is_all_concat_input_dql:
            new_concat_node = onnx.helper.make_node(name = concat_node.name,
                                                        op_type = "QLinearConcat",
                                                        inputs = input_names,
                                                        outputs = [y_name],
                                                        axis = concat_node.attrs["axis"],
                                                        **kwargs)
        else:
            new_concat_node = onnx.helper.make_node(name = concat_node.name,
                                                        op_type = "Concat",
                                                        inputs = input_names,
                                                        outputs = [y_name],
                                                        axis = concat_node.attrs["axis"])

        self.node = new_concat_node

    def get_node(self):
        return self.node

    def get_intializers(self):
        return self.intializer_list