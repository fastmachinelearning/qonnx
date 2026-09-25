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

class Gemm:

    def __init__(self, node):

        gemm_node = node

        x1 = gemm_node.inputs[0]
        x2 = gemm_node.inputs[1]
        x3 = gemm_node.inputs[2]
        y = gemm_node.outputs[0]

        new_gemm_node = onnx.helper.make_node(name = gemm_node.name, op_type = "Gemm",
                                                inputs= [x1.name, x2.name, x3.name],
                                                outputs = [y.name],
                                                alpha = gemm_node.attrs["alpha"],
                                                beta = gemm_node.attrs["beta"],
                                                transB = gemm_node.attrs["transB"])

        self.node = new_gemm_node

    def get_node(self):
        return self.node