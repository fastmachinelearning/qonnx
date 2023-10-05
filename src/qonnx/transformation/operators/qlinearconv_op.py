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
import numpy as np
from .helper import helper

class QLinearConv:

    def __init__(self, node, aecg_zendnn_opt, remove_relu, conv_count):
        x_DQL_node = node.i()

        conv_node = node

        has_bias = True if len(conv_node.inputs) == 3 else False

        w_DQL_node = conv_node.inputs[1].inputs[0]
        QCDQ_model_detected=False
        clip_max = np.iinfo(np.int8).min
        clip_min = np.iinfo(np.int8).max
        if (helper.is_constant_tensor(w_DQL_node.i())==False and w_DQL_node.i().op == "Clip"):
            QCDQ_model_detected=True
            clip_min = w_DQL_node.i().inputs[1].values
            clip_max = w_DQL_node.i().inputs[2].values

        # b_DQL_node = (3)
        # ------------------------------------------------------------------------
        #  (1) (2)  DequantizeLinear          (1)   (2)
        #   \   |    /  (3) for bias    OR     \    /
        #    \  |   /                           \  /
        #      Conv (QDQ model)        Conv (3 - FP32 bias embedded) (QCDQ model)
        #       |                                 |
        # ------------------------------------------------------------------------
        # Initialization
        b_DQL_node = conv_node
        b_DQL_tensor = conv_node
        if has_bias:
            b_DQL_node = conv_node.inputs[2]     # For QDQ
            b_DQL_tensor = conv_node.inputs[2]   # For QCDQ
        if has_bias and QCDQ_model_detected==False:
            b_DQL_node = conv_node.inputs[2].inputs[0]
        is_fp32_bias_embedded = False
        if QCDQ_model_detected:
            if helper.is_constant_tensor(b_DQL_tensor) and b_DQL_tensor.dtype == "float32":
                is_fp32_bias_embedded = True
            b_QL_tensor = b_DQL_tensor
            if is_fp32_bias_embedded:
                if not helper.is_parent_exist(b_DQL_tensor, 0, 0):
                    b_QL_tensor = b_DQL_tensor

        is_weight_tensor_quantized = False
        if len(w_DQL_node.inputs[0].inputs) == 0:
            is_weight_tensor_quantized = True
        is_bias_tensor_quantized = False
        if QCDQ_model_detected and has_bias and not is_fp32_bias_embedded and not helper.is_parent_exist(b_DQL_tensor, 0, 0) and b_DQL_tensor.dtype == "int32":
            is_bias_tensor_quantized = True
        elif QCDQ_model_detected==False and has_bias and len(b_DQL_node.inputs[0].inputs) == 0:
            is_bias_tensor_quantized = True

        if not is_weight_tensor_quantized:
            w_QL_node = w_DQL_node.i()

        if QCDQ_model_detected==False and has_bias and (not is_bias_tensor_quantized):
            b_QL_node = b_DQL_node.i()

        x_scale_tensor = x_DQL_node.inputs[1]
        x_scale = x_scale_tensor.values
        x_zp_tensor = x_DQL_node.inputs[2]

        w_scale_tensor = w_DQL_node.inputs[1]
        w_scale = w_scale_tensor.values
        w_zp_tensor = w_DQL_node.inputs[2]

        is_relu_present = False
        if conv_node.o().op == "Relu":
            relu_node = conv_node.o()
            is_relu_present = True
            if relu_node.o().op == "QuantizeLinear":
                y_QL_node = relu_node.o()
                y_scale_tensor = y_QL_node.inputs[1]
                y_scale = y_scale_tensor.values
                y_zp_tensor = y_QL_node.inputs[2]
            else:
                print("*********************** ERROR output of Relu node ", relu_node.name, " is not QuantizeLinear ***********************")
        elif (conv_node.o().op == "QuantizeLinear"):
            y_QL_node = conv_node.o()
            y_scale_tensor = y_QL_node.inputs[1]
            y_scale = y_scale_tensor.values
            y_zp_tensor = y_QL_node.inputs[2]
        else:
            print("*********************** ERROR output of Conv node ", conv_node.name, " is not QuantizeLinear ***********************")

        S8_MIN = np.iinfo(np.int8).min
        S8_MAX = np.iinfo(np.int8).max
        if clip_min != np.iinfo(np.int8).max and clip_max != np.iinfo(np.int8).min:
            S8_MIN = clip_min
            S8_MAX = clip_max
        U8_MIN = np.iinfo(np.uint8).min
        U8_MAX = np.iinfo(np.uint8).max
        S32_MIN = np.iinfo(np.int32).min
        S32_MAX = np.iinfo(np.int32).max

        if (QCDQ_model_detected and helper.is_parent_exist(w_DQL_node, 0, 0) and helper.is_parent_exist(w_DQL_node.i(0), 0, 0) and w_DQL_node.i(0).i(0).op == "QuantizeLinear"):
            w_QL_node = w_DQL_node.i(0).i(0)

        if QCDQ_model_detected==False and has_bias and (not is_bias_tensor_quantized) and helper.is_parent_exist(b_DQL_node, 0, 0):
            b_QL_node = b_DQL_node.i()

        # --------------------------------------------------------------------------
        #       QuantizeLinear (w_QL_node set to this in first if condition)
        #            |
        #          Clip
        #            |
        #      DequantizeLinear (for weight)
        # (0)   / (1)
        #  |   /
        #  Conv
        # --------------------------------------------------------------------------
        if QCDQ_model_detected and helper.is_parent_exist(w_DQL_node, 0, 0) and helper.is_parent_exist(w_DQL_node.i(0), 0, 0):
            w_QL_node = w_DQL_node.i().i()
            quantized_weight_tensor = w_QL_node.inputs[0]
        #if is_weight_tensor_quantized and QCDQ_model_detected:
        #    quantized_weight_tensor = w_DQL_node.inputs[1].values
        if is_weight_tensor_quantized and not QCDQ_model_detected:
            quantized_weight_tensor = w_DQL_node.inputs[0].values
        elif helper.is_constant_tensor(w_QL_node):
            quantized_weight_tensor = w_QL_node.values
            quantized_weight_tensor = np.clip(quantized_weight_tensor, S8_MIN, S8_MAX)
            quantized_weight_tensor = np.round(quantized_weight_tensor)
            quantized_weight_tensor = quantized_weight_tensor.astype(np.int8)
        elif not helper.is_constant_tensor(w_QL_node):
            weight_tensor = w_QL_node.inputs[0]
            weight_scale_tensor = w_QL_node.inputs[1]
            weight_zp_tensor = w_QL_node.inputs[2]

            weight_scaled_tensor = weight_scale_tensor.values * np.ones(weight_tensor.shape)
            if QCDQ_model_detected:
                weight_scaled_tensor = np.ones(weight_tensor.shape) * weight_scale_tensor.values[:, np.newaxis, np.newaxis, np.newaxis]
            b = weight_tensor.values / weight_scaled_tensor
            c = weight_zp_tensor.values * np.ones(weight_tensor.shape)
            if QCDQ_model_detected:
                c = weight_zp_tensor.values[:, np.newaxis, np.newaxis, np.newaxis] * np.ones(weight_tensor.shape)
            quantized_weight_tensor = b + c
            if weight_zp_tensor.dtype == "int8":
                quantized_weight_tensor = np.clip(quantized_weight_tensor, S8_MIN, S8_MAX)
            elif weight_zp_tensor.dtype == "uint8":
                quantized_weight_tensor = np.clip(quantized_weight_tensor, U8_MIN, U8_MAX)
            quantized_weight_tensor = np.round(quantized_weight_tensor)
            quantized_weight_tensor = quantized_weight_tensor.astype(np.int8)
            if QCDQ_model_detected:
                clip_node = w_DQL_node.i()
                clip_node.inputs.clear()
                clip_node.outputs.clear()

        if has_bias and is_bias_tensor_quantized:
            quantized_bias_tensor = b_DQL_node.inputs[0].values
        elif is_fp32_bias_embedded and has_bias:
            bias_tensor = b_QL_tensor
            bias_scale_tensor1 = w_QL_node.inputs[1]
            bias_zp_tensor = w_QL_node.inputs[2]

            # satutration after QL node
            a = x_scale * bias_scale_tensor1.values
            b = bias_tensor.values / a
            # Zero point is set to 0 for quantizing bias
            d = b
            d = np.round(d)
            quantized_bias_tensor = d
            quantized_bias_tensor = np.clip(quantized_bias_tensor, S32_MIN, S32_MAX)
            quantized_bias_tensor = np.round(quantized_bias_tensor)
            quantized_bias_tensor = quantized_bias_tensor.astype(np.int32)
        elif has_bias:
            bias_tensor = b_QL_node.inputs[0]
            bias_scale_tensor1 = b_QL_node.inputs[1]
            bias_zp_tensor = b_QL_node.inputs[2]

            # satutration after QL node
            a = bias_scale_tensor1.values * np.ones(bias_tensor.shape)
            b = bias_tensor.values / a
            c = bias_zp_tensor.values * np.ones(bias_tensor.shape)
            d = b + c
            if bias_zp_tensor.dtype == "int8":
                d = np.clip(d, S8_MIN, S8_MAX)
            elif bias_zp_tensor.dtype == "uint8":
                d = np.clip(d, U8_MIN, U8_MAX)
            d = np.round(d)

            # now again dequantize it
            e = d * a
            f = e - c
            # f is now fp32 tensor

            bias_scale = x_scale * w_scale
            bias_scale_tensor = bias_scale * np.ones(bias_tensor.shape)
            quantized_bias_tensor = (f / bias_scale_tensor)
            quantized_bias_tensor = np.clip(quantized_bias_tensor, S32_MIN, S32_MAX)
            quantized_bias_tensor = np.round(quantized_bias_tensor)
            quantized_bias_tensor = quantized_bias_tensor.astype(np.int32)

        x_QL_node = x_DQL_node.i()
        is_x_QL_maxpool = False
        is_X_QL_transpose = True if x_QL_node.op == "Transpose" else False
        maxpool_input_s8 = False # True means s8 False means u8
        if x_QL_node.op == "MaxPool":
            is_x_QL_maxpool = True

        if helper.is_parent_exist(x_QL_node, 0, 0):
            if x_QL_node.i().op == "Relu":
                if remove_relu:
                    # if this flag is enabled, then relu will not be added thus x_name will be x_QL's output tensor name
                    x_name = x_QL_node.outputs[0].name
                else:
                    if (x_QL_node.i().i().op == "Conv") or (x_QL_node.i().i().op == "Add" and x_QL_node.i().i().i().inputs[2].values.dtype == np.int8):

                        """
                        these are 2 condtions
                        one in resnet50v1

                        DQL             DQL
                        |               |
                        |               |
                        V               |
                        Add<-------------
                        |
                        |
                        V
                        Relu------------------------------
                        |
                        |
                        QL (x_QL_node)
                        |
                        |
                        DQL     DQL     DQL
                        |       |       |
                        |       |       |
                        Conv<------------

                        if Add input is s8
                        x_relu_node = Relu
                        relu will be maintained due to s8 data type thus
                        x_name = relu's output

                        other case is in Resnet50v1.5

                        Conv
                        |
                        |
                        Relu
                        |
                        |
                        QL
                        |
                        |
                        DQL     DQL     DQL
                        |       |       |
                        |       |       |
                        Conv<------------

                        we maintain relu node here thus x_name = relu's output

                        """
                        x_relu_node = x_QL_node.i()
                        x_name = x_relu_node.outputs[0].name
                    else:
                        x_name = x_QL_node.outputs[0].name
            elif x_QL_node.op == "MaxPool":
                """
                this is resnet50v1 case

                QL
                |
                |
                V
                Maxpool
                |
                |
                V
                DQL     DQL     DQL
                |       |       |
                |       |       |
                V       |       |
                Conv<------------

                """
                x_name = x_QL_node.outputs[0].name
                if x_QL_node.i().op == "QuantizeLinear":
                    if (x_QL_node.i()).inputs[2].dtype == np.int8:
                        maxpool_input_s8 = True
                    elif (x_QL_node.i()).inputs[2].dtype == np.uint8:
                        maxpool_input_s8 = False
            else:
                x_name = x_QL_node.outputs[0].name
                if x_QL_node.op == "Clip":
                    x_name = str(int(x_QL_node.o().outputs[0].name)-3)
        else:
            x_name = x_QL_node.outputs[0].name

        if is_relu_present and not(remove_relu):
            y_name = conv_node.outputs[0].name
        else:
            y_name = y_QL_node.outputs[0].name

        x_scale_name = conv_node.name + "_X_SCALE"
        x_scale_value = x_scale
        x_scale_tensor = helper.create_initializer_tensor(name=x_scale_name,
                                                          tensor_array=x_scale_value,
                                                          data_type=onnx.TensorProto.FLOAT)

        x_zp_name = conv_node.name + "_X_ZERO_POINT"
        x_zp_value = x_zp_tensor.values

        if aecg_zendnn_opt and conv_count > 0:
            x_zp_tensor = helper.create_initializer_tensor(name=x_zp_name,
                                                                tensor_array=x_zp_value,
                                                                data_type=onnx.TensorProto.UINT8)
        else:
            if is_x_QL_maxpool:
                if maxpool_input_s8:
                    x_zp_tensor = helper.create_initializer_tensor(name=x_zp_name,
                                                                    tensor_array=x_zp_value,
                                                                    data_type=onnx.TensorProto.INT8)
                else:
                    x_zp_tensor = helper.create_initializer_tensor(name=x_zp_name,
                                                                    tensor_array=x_zp_value,
                                                                    data_type=onnx.TensorProto.UINT8)
            elif is_X_QL_transpose:
                x_zp_tensor = helper.create_initializer_tensor(name=x_zp_name,
                                                                    tensor_array=x_zp_value,
                                                                    data_type=onnx.TensorProto.INT8)
            else:
                if (x_QL_node.op == "QuantizeLinear" and x_QL_node.inputs[2].dtype == np.int8):
                    x_zp_tensor = helper.create_initializer_tensor(name=x_zp_name,
                                                            tensor_array=x_zp_value,
                                                            data_type=onnx.TensorProto.INT8)
                elif (x_QL_node.op == "QuantizeLinear" and x_QL_node.inputs[2].dtype == np.uint8):
                    x_zp_tensor = helper.create_initializer_tensor(name=x_zp_name,
                                                                tensor_array=x_zp_value,
                                                                data_type=onnx.TensorProto.UINT8)
                elif x_QL_node.op == "Relu" or x_QL_node.op == "Clip":
                    if (x_QL_node.i().op == "QuantizeLinear" and x_QL_node.i().inputs[2].dtype == np.int8):
                        x_zp_tensor = helper.create_initializer_tensor(name=x_zp_name,
                                                            tensor_array=x_zp_value,
                                                            data_type=onnx.TensorProto.INT8)
                    elif (x_QL_node.i().op == "QuantizeLinear" and x_QL_node.i().inputs[2].dtype == np.uint8):
                        x_zp_tensor = helper.create_initializer_tensor(name=x_zp_name,
                                                                tensor_array=x_zp_value,
                                                                data_type=onnx.TensorProto.UINT8)
                else:
                    print("ERROR Please check x_zp_tensor of ", conv_node.name)

        w_name = conv_node.inputs[1].name
        w_value = quantized_weight_tensor
        w_tensor = helper.create_initializer_tensor(name=w_name,
                                                    tensor_array=w_value,
                                                    data_type=onnx.TensorProto.INT8)

        w_scale_name = conv_node.name + "_W_SCALE"
        w_scale_value = w_scale
        w_scale_tensor = helper.create_initializer_tensor(name=w_scale_name,
                                                          tensor_array=w_scale_value,
                                                          data_type=onnx.TensorProto.FLOAT)

        w_zp_name = conv_node.name + "_W_ZERO_POINT"
        w_zp_value = w_zp_tensor.values
        w_zp_tensor = helper.create_initializer_tensor(name=w_zp_name,
                                                       tensor_array=w_zp_value,
                                                       data_type=onnx.TensorProto.INT8)

        y_scale_name = conv_node.name + "_Y_SCALE"
        y_scale_value = y_scale
        y_scale_tensor = helper.create_initializer_tensor(name=y_scale_name,
                                                          tensor_array=y_scale_value,
                                                          data_type=onnx.TensorProto.FLOAT)

        y_zp_name = conv_node.name + "_Y_ZERO_POINT"
        y_zp_value = y_zp_tensor.values

        if aecg_zendnn_opt:
            # if this opt is enabled then y_zp has be to set to u8 type
            y_zp_tensor = helper.create_initializer_tensor(name=y_zp_name,
                                                    tensor_array=y_zp_value,
                                                    data_type=onnx.TensorProto.UINT8)
        else:
            if y_zp_tensor.dtype == np.int8:
                y_zp_tensor = helper.create_initializer_tensor(name=y_zp_name,
                                                            tensor_array=y_zp_value,
                                                            data_type=onnx.TensorProto.INT8)
            elif y_zp_tensor.dtype == np.uint8:
                y_zp_tensor = helper.create_initializer_tensor(name=y_zp_name,
                                                        tensor_array=y_zp_value,
                                                        data_type=onnx.TensorProto.UINT8)

        if has_bias:
            b_name = conv_node.inputs[2].name
            b_value = quantized_bias_tensor
            b_tensor = helper.create_initializer_tensor(name=b_name,
                                                        tensor_array=b_value,
                                                        data_type=onnx.TensorProto.INT32)

        if helper.is_attr_exist(conv_node, 'auto_pad'):
            auto_pad_attr = conv_node.attrs["auto_pad"]
        else:
            auto_pad_attr = "NOTSET"

        if helper.is_attr_exist(conv_node, 'dilations'):
            dilations_attr = conv_node.attrs["dilations"]
        else:
            dilations_attr = 1

        if helper.is_attr_exist(conv_node, 'group'):
            group_attr = conv_node.attrs["group"]
        else:
            group_attr = 1

        if helper.is_attr_exist(conv_node, 'pads'):
            pads_attr = conv_node.attrs["pads"]
        else:
            pads_attr = [0,0,0,0]

        if helper.is_attr_exist(conv_node, 'strides'):
            strides_attr = conv_node.attrs["strides"]
        else:
            strides_attr = 1

        qlinearconv_node = onnx.helper.make_node(name = conv_node.name, op_type = "QLinearConv", inputs = [x_name, x_scale_name, x_zp_name, w_name, w_scale_name, w_zp_name, y_scale_name, y_zp_name], outputs = [y_name], auto_pad = auto_pad_attr, group = group_attr, dilations = dilations_attr, kernel_shape = conv_node.attrs["kernel_shape"], pads = pads_attr, strides = strides_attr)
        if has_bias:
            qlinearconv_node = onnx.helper.make_node(name = conv_node.name, op_type = "QLinearConv", inputs = [x_name, x_scale_name, x_zp_name, w_name, w_scale_name, w_zp_name, y_scale_name, y_zp_name, b_name], outputs = [y_name], auto_pad = auto_pad_attr, group = group_attr, dilations = dilations_attr, kernel_shape = conv_node.attrs["kernel_shape"], pads = pads_attr, strides = strides_attr)

        if is_relu_present:
            relu_node = onnx.helper.make_node(name = relu_node.name, op_type = "Relu", inputs = [conv_node.outputs[0].name], outputs = [relu_node.outputs[0].name])
            self.relu_node = relu_node

        self.node = qlinearconv_node

        intializer_list = []
        intializer_list.append(x_scale_tensor)
        intializer_list.append(x_zp_tensor)
        intializer_list.append(w_tensor)
        intializer_list.append(w_scale_tensor)
        intializer_list.append(w_zp_tensor)
        intializer_list.append(y_scale_tensor)
        intializer_list.append(y_zp_tensor)
        if has_bias:
            intializer_list.append(b_tensor)
        self.intializer_list = intializer_list

    def get_node(self):
        return self.node

    def get_intializers(self):
        return self.intializer_list

    def get_relu_node(self):
        return self.relu_node
