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

class Reshape:

    def __init__(self, node):

        reshape_node = node

        x_name = reshape_node.inputs[0].name

        x2_name = reshape_node.inputs[1].name
        if helper.is_constant_tensor(reshape_node.inputs[1]):
            x2_value = reshape_node.inputs[1].values
            x2_tensor = helper.create_initializer_tensor(x2_name,x2_value,onnx.TensorProto.INT64)

        y_name = reshape_node.outputs[0].name

        try:
            new_reshape_node = onnx.helper.make_node(name = reshape_node.name, op_type = "Reshape",
                                            inputs = [x_name, x2_name],
                                            outputs = [y_name],
                                            allowzero = reshape_node.attrs["allowzero"])
        except:
            new_reshape_node = onnx.helper.make_node(name = reshape_node.name, op_type = "Reshape",
                                            inputs = [x_name, x2_name],
                                            outputs = [y_name])

        self.node = new_reshape_node

        intializer_list = []
        if helper.is_constant_tensor(reshape_node.inputs[1]):
            intializer_list.append(x2_tensor)
        self.intializer_list = intializer_list

    def get_node(self):
        return self.node

    def get_intializers(self):
        return self.intializer_list