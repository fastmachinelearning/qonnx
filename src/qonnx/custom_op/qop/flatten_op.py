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

class Flatten:

    def __init__(self, node):

        flatten_node = node
        x_name = flatten_node.inputs[0].name
        y_name = flatten_node.outputs[0].name

        if flatten_node.i().op == "DequantizeLinear":
            node1 = flatten_node.i()
            x_name = node1.inputs[0].name

        if flatten_node.o().op == "QuantizeLinear":
            node2 = flatten_node.o()
            y_name = node2.outputs[0].name


        new_flatten_node = onnx.helper.make_node(name = flatten_node.name, op_type = "Flatten",
                                                        inputs = [x_name],
                                                        outputs = [y_name])


        self.node = new_flatten_node

    def get_node(self):
        return self.node