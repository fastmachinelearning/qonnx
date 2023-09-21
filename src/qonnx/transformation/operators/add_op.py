a########################################################################
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

class QLinearAdd:

    def __init__(self, node, aecg_zendnn_opt, remove_relu):

        add_node = node

        if len(add_node.inputs[1].inputs)==0:
            # if Add node has only 1 input node and other input is constant tensor we cannot change it to QLinearAdd node hence keeping it as is
            x_name = add_node.inputs[0].name
            y_name = add_node.outputs[0].name

            const_val = add_node.inputs[1].values

            const_name = add_node.name + "_const_add_tensor"
            y_scale_tensor = helper.create_initializer_tensor(name=const_name,
                                                            tensor_array=const_val,
                                                            data_type=onnx.TensorProto.FLOAT)

            new_add_node = onnx.helper.make_node(name = add_node.name,
                                                    op_type = "Add",
                                                    inputs = [x_name, const_name],
                                                    outputs = [y_name])
            self.node = new_add_node

            if helper.is_child_present(add_node, 0, 0) and add_node.o().op == "Relu":
                relu_node = add_node.o()
                relu_node1 = onnx.helper.make_node(name = relu_node.name, op_type = "Relu", inputs = [add_node.outputs[0].name], outputs = [relu_node.outputs[0].name])
                self.relu_node = relu_node1

            intializer_list = []
            intializer_list.append(y_scale_tensor)
            self.intializer_list = intializer_list

        else:
            input_node1 = add_node.inputs[0].inputs[0]
            input_node2 = add_node.inputs[1].inputs[0]
            output_node = add_node.o()

            is_relu_present = False
            if output_node.op == "Relu":
                is_relu_present = True
                relu_node = output_node
                # relu_node gets updated in later conditions thus keeping relu_node_name and relu_node_output_tensor to make it simple to keep their track
                relu_node_name = relu_node.name
                relu_node_output_tensor = relu_node.outputs[0].name
                if relu_node.o().op == "QuantizeLinear":
                    output_node = relu_node.o()
                else:
                    print("*********************** ERROR output of Relu node ", relu_node.name, " is not QuantizeLinear ***********************")
            elif not(output_node.op == "QuantizeLinear"):
                print("*********************** ERROR output of Add node ", add_node.name, " is not QuantizeLinear ***********************")


            # in order to get scale and zp for the 2 inputs to Add node, we need 2 DQL nodes.
            if not (input_node1.op == "DequantizeLinear" and  input_node2.op == "DequantizeLinear"):

                """
                case observed in Resnet50v1
                                            Add1
                                            |
                                            |
                                            V
                                            Relu--------------------
                                            |                       |
                                            |                       |
                                            V                       |
                                            QL                      |
                                            |                       |
                                            |                       |
                                            |                       |
                DQL             DQL         DQL2                    |
                |               |           |                       |
                |               |           |                       |
                ----------------------------Conv                    |
                                            |                       |
                                            |                       |
                                            V                       |
                                            QL                      |
                                            |                       |
                                            |                       |
                                            V                       |
                DQL             DQL         DQL                     |
                |               |           |                       |
                |               |           |                       |
                |               |           V                       |
                ---------------------------Conv                     |
                                            |                       |
                                            |                       |
                                            V                       |
                                            QL                      |
                                            |                       |
                                            |                       |
                                            V                       |
                DQL             DQL         DQL                     |
                |               |           |                       |
                |               |           |                       |
                |               |           V                       |
                ----------------------------Conv                    |
                                            |                       |
                                            |                       |
                                            V                       |
                                            QL                      |
                                            |                       |
                                            |                       |
                                            V                       |
                                            DQL1                    |
                                            |                       |
                                            |                       |
                                            V                       |
                                            Add<---------------------


                    here Add doesn't have 1 of the DQL node, so we take DQL2 as the other DQL node.

                    in case both inputs are missing DQL node, haven't encountered this case to this is flagged for now, if needed will be handled later depending on the case
                """
                if not (input_node1.op == "DequantizeLinear") and  not (input_node2.op == "DequantizeLinear"):
                    print("***************************** ERROR No input of Add node is DequantizeLinear ***********************************")
                elif not (input_node1.op == "DequantizeLinear"):
                    # if input_node1 is not DQL
                    if input_node1.op == "Relu":
                        relu_node = input_node1
                        if relu_node.i().op == "Add" and relu_node.o().op == "QuantizeLinear":
                            if (relu_node.o()).o().op == "DequantizeLinear":
                                input_node1 = (relu_node.o()).o()
                                # in the example case, shown input_node1 is now DQL2
                    elif input_node1.op == "MaxPool":
                        # when resnet strides has been implemented there will be a maxpool node between the shown Relu and Add node.
                        maxpool_node = input_node1
                        relu_node = maxpool_node.i()
                        if relu_node.i().op == "Add" and (relu_node.o().op == "QuantizeLinear" or relu_node.output[0].outputs[1].op == "QuantizeLinear"):
                            if (relu_node.o()).o().op == "DequantizeLinear":
                                input_node1 = (relu_node.o()).o()
                                # input_node1 is now DQL2
                            elif (relu_node.outputs[0].outputs[1]).o().op == "DequantizeLinear":
                                input_node2 = (relu_node.outputs[0].outputs[1]).o()
                                # input_node2 is now DQL2
                    elif input_node1.op == "Add":

                        """
                        this case is observed in mobilenetv2-12-qdq.onnx


                        Add2-------------------------
                        |                           |
                        |                           |
                        |                           V
                        |                           QL1
                        |                           |
                        |                           |
                        |                           V
                        |                           DQL1         DQL         DQL
                        |                           |           |           |
                        |                           |           |           |
                        |                           V           |           |
                        |                           Conv<--------------------
                        |                           |
                        |                           |
                        |                           V
                        |                           QL
                        |                           |
                        |                           |
                        |                           V
                        |                           DQL         DQL         DQL
                        |                           |           |           |
                        |                           |           |           |
                        |                           V           |           |
                        |                           Conv<--------------------
                        |                           |
                        |                           |
                        |                           V
                        |                           QL
                        |                           |
                        |                           |
                        |                           V
                        |                           DQL         DQL         DQL
                        |                           |           |           |
                        |                           |           |           |
                        |                           V           |           |
                        |                           Conv<-------------------
                        |                           |
                        |                           |
                        |                           V
                        |                           QL
                        |                           |
                        |                           |
                        |                           V
                        |                           DQL
                        |                           |
                        |                           |
                        Add1-------------------------

                        Add2 = parent_add_node
                        QL1 = parent_add_node_ql_node
                        input_node1 = DQL1

                        """
                        parent_add_node = input_node1
                        parent_add_node_ql_node = parent_add_node.o()
                        input_node1 = parent_add_node_ql_node.o()
                elif not (input_node2.op == "DequantizeLinear"):
                    # if input_node2 is not DQL
                    if input_node2.op == "Relu":
                        relu_node = input_node2
                        if relu_node.i().op == "Add" and relu_node.o().op == "QuantizeLinear":
                            if (relu_node.o()).o().op == "DequantizeLinear":
                                input_node2 = (relu_node.o()).o()
                                # input_node2 is now the DQL node from which we need to take scale and zp

                    elif input_node2.op == "MaxPool":
                            maxpool_node = input_node2
                            if maxpool_node.i().op == "Relu":
                                relu_node = maxpool_node.i()
                            elif maxpool_node.i().op == "DequantizeLinear":
                                if maxpool_node.i().i().op == "QuantizeLinear":
                                    if maxpool_node.i().i().i().op == "Relu":
                                        relu_node = maxpool_node.i().i().i()
                            if relu_node.i().op == "Add" and (relu_node.o().op == "QuantizeLinear" or (len(relu_node.outputs[0].outputs)>1 and  relu_node.output[0].outputs[1].op == "QuantizeLinear")):
                                if (relu_node.o()).o().op == "DequantizeLinear":
                                    input_node2 = (relu_node.o()).o()
                                elif len(relu_node.outputs[0].outputs)>1 and (relu_node.outputs[0].outputs[1]).o().op == "DequantizeLinear":
                                    input_node2 = (relu_node.outputs[0].outputs[1]).o()
                                    # input_node2 is now the DQL node from which we need to take scale and zp

            if input_node1.op == "DequantizeLinear" and  input_node2.op == "DequantizeLinear" and output_node.op == "QuantizeLinear":
                # now we have input_node1 = input_node2 = DQL and output as QL node
                if add_node.inputs[0].inputs[0].op == "MaxPool":
                    # this is strides case now if Maxpool is parent to Add node, maxpool = node1
                    node1 = add_node.i()
                elif add_node.inputs[0].inputs[0].op == "Add":
                    # this is for mobilenet case, so Add2 = node1
                    node1 = add_node.i()
                else:
                    """
                    if above 2 cases not there lets assume following case now from Resnet50v1 model

                    |       DQL         DQL                     |           DQL         DQL
                    |       |           |                       |           |           |
                    |       |           |                       |           |           |
                    Conv<---------------                        Conv---------------------
                    |                                           |
                    |                                           |
                    QL1                                         QL2
                    |                                           |
                    |                                           |
                    DQL                                         DQL
                    |                                           |
                    |                                           |
                    Add<-----------------------------------------

                    now node1 is QL1/QL2

                    """
                    node1 = add_node.inputs[0].inputs[0].i()

                if add_node.inputs[1].inputs[0].op == "MaxPool":
                    # same as above but for other input, node2 = maxpool node
                    node2 = add_node.inputs[1].inputs[0]
                else:
                    # same as the above general case discussed, node2 = QL1/QL2
                    node2 =input_node2.i()

                if node1.op == "Add":
                    # this is mobilenet case explained abaove, node1 will be converted to QLinearAdd node and it wiil act as input to current add node
                    # this a_name = QL1 output tensor name (please refer above mobilenet case)
                    a_name = node1.o().outputs[0].name
                else:
                    # refering to general case taken above from resnet50v1 model, a_name = QL1/QL2's output tensor name
                    a_name = node1.outputs[0].name

                a_scale_name = add_node.name + "_A_SCALE"
                a_scale_value = input_node1.inputs[1].values
                a_scale_tensor = helper.create_initializer_tensor(name=a_scale_name,
                                                                tensor_array=a_scale_value,
                                                                data_type=onnx.TensorProto.FLOAT)

                a_zp_name = add_node.name + "_A_ZP"
                a_zp_value = input_node1.inputs[2].values

                if aecg_zendnn_opt:
                    a_zp_tensor = helper.create_initializer_tensor(name=a_zp_name,
                                                                    tensor_array=a_zp_value,
                                                                    data_type=onnx.TensorProto.UINT8)
                else:
                    if node1.i().op == "QuantizeLinear" and node1.i().i() == "Relu":
                        a_zp_tensor = helper.create_initializer_tensor(name=a_zp_name,
                                                                    tensor_array=a_zp_value,
                                                                    data_type=onnx.TensorProto.UINT8)
                    else:
                        if input_node1.inputs[2].dtype == np.int8:
                            a_zp_tensor = helper.create_initializer_tensor(name=a_zp_name,
                                                                        tensor_array=a_zp_value,
                                                                        data_type=onnx.TensorProto.INT8)
                        elif input_node1.inputs[2].dtype == np.uint8:
                            a_zp_tensor = helper.create_initializer_tensor(name=a_zp_name,
                                                                        tensor_array=a_zp_value,
                                                                        data_type=onnx.TensorProto.UINT8)

                # TODO: Only 1 condition is handled here that Add Node's 1st parent is DQL<--QL and 2nd parent can be Relu. Vice Versa and other cases are not encountered yet thus not handled.
                if helper.is_parent_exist(node2, 0, 0):
                    if remove_relu:
                        # b_name = the QL's output tensor
                        b_name = node2.outputs[0].name
                    else:
                        # check Relu and input of Add node is s8, any 1 input can be checked, thus we check for node1
                        if node2.i().op == "Relu" and node1.inputs[2].values.dtype == np.int8:
                            """
                            this case is observed in renset50v1.5

                                                    DQL                 DQL
                                                    |                   |
                                                    |                   |
                                                    V                   |
                                                    Add<-----------------
                                                    |
                                                    |
                                                    V
                                                    Relu1
                                                    |
                                                    |
                                                    V
                                                    QL1
                                                    |
                                                    |
                                                    V
                ------------------------------------DQL1        DQL         DQL
                |                                    |           |           |
                |                                    |           |           |
                |                                    V           |           |
                |                                   Conv4<--------------------
                |                                    |
                |                                    |
                |                                    V
                |                                    Relu
                |                                    |
                |                                    |
                |                                    V
                |                                   QL
                |                                    |
                |                                    |
                |                                    V
                |                                   DQL         DQL         DQL
                |                                    |           |           |
                |                                    |           |           |
                |                                    V           |           |
                |                                   Conv3<--------------------
                |                                    |
                |                                    |
                |                                    V
                |                                   Relu
                |                                    |
                |                                    |
                |                                    V
                |                                   QL
                |                                    |
                |                                    |
                |                                    V
                |                                    DQL         DQL         DQL
                |                                    |           |           |
                |                                    |           |           |
                |                                    V           |           |
                |                                   Conv2<--------------------
                |                                    |
                |                                    |
                |                                    V
                |                                   QL
                |                                    |
                |                                    |
                |                                    V
                |                                  DQL
                |                                    |
                |                                    |
                |                                    V
                ---------------------------------->Add1


                in this case node2 is QL1
                node2_relu_node = Relu1
                thus b_name = Relu1's output as abotve top Add node is converted as follows-

                QLinearAdd
                |
                |
                V
                Relu1

                thus relu1 output is set to b_name


                            """
                            node2_relu_node = node2.i()
                            if node2_relu_node.i().op == "Conv" or node2_relu_node.i().op == "Add":
                                b_name = node2_relu_node.outputs[0].name
                            else:
                                b_name = node2.outputs[0].name
                        else:
                            b_name = node2.outputs[0].name
                else:
                    print("************* ERROR ****************** Please check parent of Add Node's parent, ", node2.name)

                b_scale_name = add_node.name + "_B_SCALE"
                b_scale_value = input_node2.inputs[1].values
                b_scale_tensor = helper.create_initializer_tensor(name=b_scale_name,
                                                                tensor_array=b_scale_value,
                                                                data_type=onnx.TensorProto.FLOAT)

                b_zp_name = add_node.name + "_B_ZP"
                b_zp_value = input_node2.inputs[2].values

                if aecg_zendnn_opt:
                    b_zp_tensor = helper.create_initializer_tensor(name=b_zp_name,
                                                                    tensor_array=b_zp_value,
                                                                    data_type=onnx.TensorProto.UINT8)
                else:
                    if node2.i().op == "QuantizeLinear" and node2.i().i().op == "Relu":
                        b_zp_tensor = helper.create_initializer_tensor(name=b_zp_name,
                                                                    tensor_array=b_zp_value,
                                                                    data_type=onnx.TensorProto.UINT8)
                    else:
                        if input_node2.inputs[2].dtype == np.int8:
                            b_zp_tensor = helper.create_initializer_tensor(name=b_zp_name,
                                                                        tensor_array=b_zp_value,
                                                                        data_type=onnx.TensorProto.INT8)
                        elif input_node2.inputs[2].dtype == np.uint8:
                            b_zp_tensor = helper.create_initializer_tensor(name=b_zp_name,
                                                                        tensor_array=b_zp_value,
                                                                        data_type=onnx.TensorProto.UINT8)

                y_scale_name = add_node.name + "_Y_SCALE"
                y_scale_value = output_node.inputs[1].values
                y_scale_tensor = helper.create_initializer_tensor(name=y_scale_name,
                                                                tensor_array=y_scale_value,
                                                                data_type=onnx.TensorProto.FLOAT)

                y_zp_name = add_node.name + "_Y_ZP"
                y_zp_value = output_node.inputs[2].values

                if aecg_zendnn_opt:
                    y_zp_tensor = helper.create_initializer_tensor(name=y_zp_name,
                                                                    tensor_array=y_zp_value,
                                                                    data_type=onnx.TensorProto.UINT8)
                    y_name = output_node.outputs[0].name
                else:
                    if output_node.inputs[2].dtype == np.int8:
                        y_zp_tensor = helper.create_initializer_tensor(name=y_zp_name,
                                                                    tensor_array=y_zp_value,
                                                                    data_type=onnx.TensorProto.INT8)
                    elif output_node.inputs[2].dtype == np.uint8:
                        y_zp_tensor = helper.create_initializer_tensor(name=y_zp_name,
                                                                    tensor_array=y_zp_value,
                                                                    data_type=onnx.TensorProto.UINT8)

                    if is_relu_present and not remove_relu and node1.inputs[2].values.dtype == np.int8:
                        y_name = add_node.outputs[0].name
                    else:
                        y_name = output_node.outputs[0].name

                kwargs = {}
                kwargs["domain"] = 'com.microsoft'


                new_add_node = onnx.helper.make_node(name = add_node.name,
                                                    op_type = "QLinearAdd",
                                                    inputs = [a_name, a_scale_name, a_zp_name, b_name, b_scale_name, b_zp_name, y_scale_name, y_zp_name],
                                                    outputs = [y_name],
                                                    **kwargs)

                self.node = new_add_node

                if is_relu_present:
                    relu_node = onnx.helper.make_node(name = relu_node_name, op_type = "Relu", inputs = [add_node.outputs[0].name], outputs = [relu_node_output_tensor])
                    self.relu_node = relu_node

                intializer_list = []
                intializer_list.append(a_scale_tensor)
                intializer_list.append(a_zp_tensor)
                intializer_list.append(b_scale_tensor)
                intializer_list.append(b_zp_tensor)
                intializer_list.append(y_scale_tensor)
                intializer_list.append(y_zp_tensor)
                self.intializer_list = intializer_list

    def get_node(self):
        return self.node

    def get_intializers(self):
        return self.intializer_list

    def get_relu_node(self):
        return self.relu_node
