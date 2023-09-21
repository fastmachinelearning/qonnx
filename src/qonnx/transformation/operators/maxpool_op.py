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

class MaxPool:

    def __init__(self, node, maxpool_count, remove_relu):

        maxpool_node = node
        x_name = maxpool_node.inputs[0].name
        y_name = maxpool_node.outputs[0].name

        if helper.is_child_present(maxpool_node, 0, 0) and maxpool_node.o().op == "QuantizeLinear":
            if helper.is_parent_exist(maxpool_node, 0, 0) and maxpool_node.i().op == "DequantizeLinear":
                q_node = maxpool_node.o()
                y_name = q_node.outputs[0].name

        if helper.is_parent_exist(maxpool_node, 0, 0):
            found_relu = False
            if maxpool_node.i().op == "Relu":
                relu_node = maxpool_node.i()
                found_relu = True
            elif maxpool_node.i().op == "DequantizeLinear":
                if maxpool_node.i().i().i().op == "Relu":
                    relu_node = maxpool_node.i().i().i()
                    found_relu = True
                elif maxpool_node.i().i().i().op == "Concat":
                    x_name = maxpool_node.i().i().outputs[0].name
                    if maxpool_node.o().op == "QuantizeLinear":
                        y_name = maxpool_node.o().outputs[0].name
                elif maxpool_node.i().i().op == "MaxPool":
                    x_name = maxpool_node.i().i().outputs[0].name

            if found_relu:
                if helper.is_child_present(relu_node, 0, 0) and relu_node.outputs[0].outputs[0].op == "MaxPool":
                    ql_node =  relu_node.outputs[0].outputs[0]
                    x_name = ql_node.outputs[0].name
                elif helper.is_child_present(relu_node, 0, 1) and relu_node.outputs[0].outputs[1].op == "MaxPool":
                    ql_node =  relu_node.outputs[0].outputs[0]
                    x_name = ql_node.outputs[0].name
                elif helper.is_child_present(relu_node, 0, 0) and relu_node.o().o().outputs[0].outputs[0].op == "MaxPool":
                    x_name = relu_node.outputs[0].name
                elif helper.is_child_present(relu_node, 0, 0) and relu_node.o().o().outputs[0].outputs[1].op == "MaxPool":
                    x_name = relu_node.outputs[0].name


            if maxpool_node.i().op == "QuantizeLinear":
                x_ql_node = maxpool_node.i()
                if remove_relu:
                    x_name = x_ql_node.outputs[0].name
                else:
                    if helper.is_parent_exist(x_ql_node, 0, 0) and  x_ql_node.i().op == "Relu" and x_ql_node.i().i().op == "Conv":
                        relu_node = x_ql_node.i()
                        x_name = relu_node.outputs[0].name

        if helper.is_attr_exist(maxpool_node, 'auto_pad'):
            auto_pad_attr = maxpool_node.attrs["auto_pad"]
        else:
            auto_pad_attr = "NOTSET"

        if helper.is_attr_exist(maxpool_node, 'ceil_mode'):
            ceil_mode_attr = maxpool_node.attrs["ceil_mode"]
        else:
            ceil_mode_attr = 0

        if helper.is_attr_exist(maxpool_node, 'dilations'):
            dilations_attr = maxpool_node.attrs["dilations"]
        else:
            dilations_attr =[1,1]

        if helper.is_attr_exist(maxpool_node, 'pads'):
            pads_attr = maxpool_node.attrs["pads"]
        else:
            pads_attr = [0,0,0,0]

        if helper.is_attr_exist(maxpool_node, 'storage_order'):
            storage_order_attr = maxpool_node.attrs["storage_order"]
        else:
            storage_order_attr = 0

        if helper.is_attr_exist(maxpool_node, 'strides'):
            strides_attr = maxpool_node.attrs["strides"]
        else:
            strides_attr = [1,1]

        new_mapool_node = onnx.helper.make_node(name = maxpool_node.name,
                                                op_type = "MaxPool",
                                                inputs = [x_name],
                                                outputs = [y_name],
                                                auto_pad = auto_pad_attr,
                                                ceil_mode = ceil_mode_attr,
                                                dilations = dilations_attr,
                                                pads = pads_attr,
                                                storage_order = storage_order_attr,
                                                strides = strides_attr,
                                                kernel_shape = maxpool_node.attrs["kernel_shape"])

        self.node = new_mapool_node

    def get_node(self):
        return self.node