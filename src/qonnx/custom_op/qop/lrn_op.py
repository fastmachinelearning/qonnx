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

class LRN:

    def __init__(self, node):

        lrn_node = node

        x_name = lrn_node.inputs[0].name
        y_name = lrn_node.outputs[0].name

        new_lrn_node = onnx.helper.make_node(name = lrn_node.name, op_type = "LRN",
                                            inputs = [x_name],
                                            outputs = [y_name],
                                            alpha = lrn_node.attrs["alpha"],
                                            beta = lrn_node.attrs["beta"],
                                            bias = lrn_node.attrs["bias"],
                                            size = lrn_node.attrs["size"])

        self.node = new_lrn_node

    def get_node(self):
        return self.node
