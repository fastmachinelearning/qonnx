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

class Slice:

    def __init__(self, node):

        slice_node = node
        x1_name = slice_node.inputs[0].name

        x2_name = slice_node.inputs[1].name
        x2_value = slice_node.inputs[1].values
        x2_tensor = helper.create_initializer_tensor(x2_name,x2_value,onnx.TensorProto.INT64)

        x3_name = slice_node.inputs[2].name
        x3_value = slice_node.inputs[2].values
        x3_tensor = helper.create_initializer_tensor(x3_name,x3_value,onnx.TensorProto.INT64)

        x4_name = slice_node.inputs[3].name
        x4_value = slice_node.inputs[3].values
        x4_tensor = helper.create_initializer_tensor(x4_name,x4_value,onnx.TensorProto.INT64)

        # x5_name = slice_node.inputs[4].name
        # x5_value = slice_node.inputs[4].values
        # x5_tensor = helper.create_initializer_tensor(x5_name,x5_value,onnx.TensorProto.INT64)

        y_name = slice_node.outputs[0].name

        # new_squeeze_node = onnx.helper.make_node(name = slice_node.name,
        #                                         op_type = "Slice",
        #                                         inputs = [x1_name, x2_name, x3_name, x4_name, x5_name],
        #                                         outputs = [y_name])

        new_squeeze_node = onnx.helper.make_node(name = slice_node.name,
                                                op_type = "Slice",
                                                inputs = [x1_name, x2_name, x3_name, x4_name],
                                                outputs = [y_name])

        self.node = new_squeeze_node

        intializer_list = []
        intializer_list.append(x2_tensor)
        intializer_list.append(x3_tensor)
        intializer_list.append(x4_tensor)
        # intializer_list.append(x5_tensor)
        self.intializer_list = intializer_list

    def get_node(self):
        return self.node

    def get_intializers(self):
        return self.intializer_list