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

class Unsqueeze:

    def __init__(self, node):

        unsq_node = node

        x1_name = unsq_node.inputs[0].name
        y_name = unsq_node.outputs[0].name

        if helper.is_constant_tensor(unsq_node.inputs[1]):
            if unsq_node.inputs[1].dtype == "int64":
                axes_tensor = helper.create_initializer_tensor(name=unsq_node.inputs[1].name,
                                                        tensor_array=unsq_node.inputs[1].values,
                                                        data_type=onnx.TensorProto.INT64)
            else:
                print("ERROR please check axes data type for Unsqueeze Node ", unsq_node.name)


            new_unsq_node = onnx.helper.make_node(name = unsq_node.name, op_type = "Unsqueeze",
                                                inputs = [x1_name, axes_tensor.name],
                                                outputs = [y_name])

        intializer_list = []
        if helper.is_constant_tensor(unsq_node.inputs[1]):
            intializer_list.append(axes_tensor)
        self.intializer_list = intializer_list

        self.node = new_unsq_node

    def get_node(self):
        return self.node
    
    def get_intializers(self):
        return self.intializer_list
