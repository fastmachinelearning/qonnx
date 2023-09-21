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

class Resize:

    def __init__(self, node):

        resize_node = node

        x1_name = resize_node.inputs[0].name
        x2_name = resize_node.inputs[1].name
        x3_name = resize_node.inputs[2].name
        x4_name = resize_node.inputs[3].name

        y_name = resize_node.outputs[0].name

        # Resize has 4 inputs, x, roi, scales, sizes. With later 3 as optional.
        # In the model (retinanet) there are 2 inputs X and sizes thus 2nd input is obtained at 3rd index.
        # 1st and 2nd index i.e x2_name and x3_name come out to be empty
        print("WARNING check inputs of resize node")

        new_resize_node = onnx.helper.make_node(name = resize_node.name, op_type = "Resize",
                                            inputs = [x1_name, x2_name, x3_name, x4_name],
                                            outputs = [y_name],
                                            coordinate_transformation_mode = resize_node.attrs["coordinate_transformation_mode"],
                                            cubic_coeff_a = resize_node.attrs["cubic_coeff_a"],
                                            mode = resize_node.attrs["mode"],
                                            nearest_mode = resize_node.attrs["nearest_mode"])

        self.node = new_resize_node

    def get_node(self):
        return self.node
