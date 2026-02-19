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

    def __init__(self, node):

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
            input_tensor_names.append(concat_node.inputs[i].name)
            if len(concat_node.inputs[i].inputs) == 0:
                 c_input = helper.create_initializer_tensor(name=concat_node.inputs[i].name,
                                                            tensor_array=concat_node.inputs[i].values,
                                                            data_type=onnx.TensorProto.INT64)
                 intializer_list.append(c_input)
                 self.intializer_list = intializer_list

        y_name = concat_node.outputs[0].name

        for i in range(number_of_inputs):
            input_names.append(input_tensor_names[i])
            if len(scale_name_list)>0 and len(zp_name_list)>0:
                input_names.append(scale_name_list[i])
                input_names.append(zp_name_list[i])

        kwargs = {}
        kwargs["domain"] = 'com.microsoft'

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
