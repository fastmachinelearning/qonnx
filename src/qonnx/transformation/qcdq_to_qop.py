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

import onnx_graphsurgeon as gs
import numpy as np
import onnx
import os
import argparse
from onnx import TensorProto
import sys

import math
import onnx.numpy_helper
from typing import Tuple
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name

from qonnx.custom_op.qop.qlinearconv_op import *
from qonnx.custom_op.qop.quantizelinear_op import *
from qonnx.custom_op.qop.dequantizelinear_op import *
from qonnx.custom_op.qop.maxpool_op import *
from qonnx.custom_op.qop.add_op import *
from qonnx.custom_op.qop.averagepool_op import *
from qonnx.custom_op.qop.squeeze_op import *
from qonnx.custom_op.qop.globalAveragePool_op import *
from qonnx.custom_op.qop.flatten_op import *
from qonnx.custom_op.qop.matmul_op import *
from qonnx.custom_op.qop.lrn_op import *
from qonnx.custom_op.qop.concat_op import *
from qonnx.custom_op.qop.softmax_op import *
from qonnx.custom_op.qop.matmul_retained_op import *
from qonnx.custom_op.qop.cast_op import *
from qonnx.custom_op.qop.gather_op import *
from qonnx.custom_op.qop.gemm_op import *
from qonnx.custom_op.qop.gemm_op_optimized import *
from qonnx.custom_op.qop.greater_op import *
from qonnx.custom_op.qop.less_op import *
from qonnx.custom_op.qop.slice_op import *
from qonnx.custom_op.qop.transpose_op import *
from qonnx.custom_op.qop.relu_op import *
from qonnx.custom_op.qop.reshape_op import *
from qonnx.custom_op.qop.identity_op import *
from qonnx.custom_op.qop.shape_op import *
from qonnx.custom_op.qop.resize_op import *
from qonnx.custom_op.qop.unsqueeze_op import *
from qonnx.custom_op.qop.clip_op import *

class CustomEnv():
    imp_strides_opt=False
    save_opt_qdq=False
    change_avgpool=False
    aecg_zendnn_opt=False
    remove_relu=True
    retain_matmul=False
    is_ryzenai_model=False
    is_retinanet=False

    def __init__(self):
        pass

class QCDQToQOp(Transformation):

    def __init__(self) -> None:
        super().__init__()

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        args = CustomEnv()
        graph = gs.import_onnx(model.model)

        graph.fold_constants()

        aecg_zendnn_opt = args.aecg_zendnn_opt
        retain_matmul = args.retain_matmul

        def is_parent_conv(index, add_node):
            if len(add_node.inputs[index].inputs)==1 and add_node.inputs[index].inputs[0].op == "DequantizeLinear":
                dql_node = add_node.inputs[index].inputs[0]
                if len(dql_node.inputs)>0 and len(dql_node.inputs[0].inputs)==1 and dql_node.i().op == "QuantizeLinear":
                    ql_node = dql_node.i()
                    if len(ql_node.inputs)>0 and len(ql_node.inputs[0].inputs)==1 and ql_node.i().op == "Conv":
                        return True
            return False

        def is_relu_input_s8_or_fp32(node):
            if node.op == "Add" and (len(node.inputs[1].inputs)==0):
                return True
            elif helper.is_parent_exist(node, 0, 0) and node.i().op == "DequantizeLinear":
                if helper.is_parent_exist(node.i(), 0, 0):
                    add_node_ql_parent = node.i().i()
                    if add_node_ql_parent.inputs[2].values.dtype == np.int8:
                        return True
                else:
                    print("Please check Add node, ", add_node.name)
            elif helper.is_parent_exist(node, 1, 0) and node.inputs[1].inputs[0].op == "DequantizeLinear":
                if helper.is_parent_exist(node.inputs[1].inputs[0], 1, 0):
                    add_node_ql_parent = (node.inputs[1].inputs[0]).i()
                    if add_node_ql_parent.inputs[2].values.dtype == np.int8:
                        return True
                else:
                    print("Please check Add node, ", add_node.name)
            else:
                return False

        def is_any_output_tensor_graph_output(node):
            for i in range(len(graph.outputs)):
                output_tensor_name = graph.outputs[i].name
                if node.outputs[0].name == output_tensor_name:
                    return True
            return False

        # return 6/7/8th child name depending on Relu is present or not
        def get_child_name(node):
            if helper.is_child_present(node, 0, 0):
                c1 = node.o()
                if c1.op == "Relu": #C1 is relu
                    if helper.is_child_present(c1, 0, 0):
                        c1 = c1.o() # c1 is QL node
                # c1 is QL now
                if helper.is_child_present(c1, 0, 0):
                    c2 = c1.o() # c2 is DQL
                if helper.is_child_present(c2, 0, 0):
                    c3 = c2.o() # c3 is Conv

                if helper.is_child_present(c3, 0, 0):
                    c4 = c3.o()

                    if c4.op == "Relu":
                        if helper.is_child_present(c4, 0, 0):
                            c4 = c4.o()
                    # c4 is QL now

                    if helper.is_child_present(c4, 0, 0):
                        c5 = c4.o() # c5 is DQL
                    if helper.is_child_present(c5, 0, 0):
                        c6 = c5.o() # c6 is conv

                        return c6.name

            print("************************* ERROR ************************* get_child_name() returned empty string")
            return ""

        def get_child_conv(node):
            if helper.is_child_present(node, 0, 0):
                c1 = node.o()
                if c1.op == "Relu" and helper.is_child_present(c1, 0, 0):
                    c1 = c1.o()
                if helper.is_child_present(c1, 0, 0):
                    c2 = c1.o()
                if helper.is_child_present(c2, 0, 0):
                    c3 = c2.o()
                return c3

        supported_op = ["Conv", "QuantizeLinear", "DequantizeLinear", "MaxPool", "Add", "AveragePool", "Squeeze", "GlobalAveragePool", "Flatten", "MatMul", "LRN", "Concat", "Softmax", "Cast", "Gather", "Gemm", "Greater", "Less", "Slice", "Transpose", "Relu", "Reshape", "Shape", "Resize", "Unsqueeze", "Clip"]

        '''
        for node in graph.nodes:
            if not node.op in supported_op:
                print(node.op, " op is currently not supported in the converter. Exiting model converter")
                sys.exit()
        '''

        maxpool_count = 0
        ctr = 0
        cast_count = 0
        clip_num = 0
        retinanet_end_pattern_found = False
        squeeze_output = False
        for node in graph.nodes:

            if node.op == "Flatten":
                squeeze_output = True
            # Resnet strides optimization for Resnet50v1

            """
                                        |--------->Relu2
                                        |           |
                                        V           |
                                       QL           |
                                        |           |
                                        |           |
                                        V           |
                DQL       DQL          DQL          |
                |          |            |           |
                |          |            |           |
                |          |            V           |
                ---------------------->Conv7        |
                                        |           |
                                        |           |
                                        V           |
                                       QL           |
                                        |           |
                                        |           |
                                        V           |
                DQL      DQL          DQL           |
                |          |            |           |
                |          |            |           |
                |          |            V           |
                --------------------->Conv6         |
                                        |           |
                                        |           |
                                        V           |
                                       QL           |
                                        |           |
                                        |           |
                                        V           |
                DQL     DQL           DQL           |
                |          |            |           |
                |          |            |           |
                |          |            V           |
                --------------------->Conv5         |
                                        |           |
                                        |           |
                                        V           |
                                        QL          |
                                        |           |
                                        |           |
                                        V           |
                                        DQL         |
                                        |           |
                                        |           V
                                        |--------->Add2
                                                    |
                                                    |
                                                    V
                                                   Relu1
                                                    |
                                                    |
                                                    V
                                                    QL
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
                |                                   QL
                |                                    |
                |                                    |
                |                                    V
                |         DQL       DQL             DQL         DQL         DQL
                |          |        |                |           |           |
                |          |        |                |           |           |
                V          |        |                V           |           |
            Conv1<--------------------              Conv2<--------------------
              |                                      |
              |                                      |
              V                                      V
              QL                                    QL
              |                                      |
              |                                      |
              V                                      V
             DQL                                   DQL
              |                                      |
              |                                      |
              V                                      V
              ------------------------------------->Add1


                    Add1 = add_node
                    Relu1 = relu_node
                    DQL1 = relu_dql_node

                    Conv1 = conv_node1
                    Conv2 = conv_node2
                    make sure conv_node1 has strides [2,2] and conv_node2 has strides [1,1] and conv_node1 will be the shortcut path

                    Conv4 and Conv1 are child1_node and child2_node (not necessary conv4 is child1_node and conv1 is child2_node)
                    but we are sure conv4's 6th child is conv2 thus get_child_name() gives 6th child name of child1_node and child2_name and check if the 6th child name = conv2's name when it is found 
                    make child1_node = conv_node1 that is Conv1 = conv_node1 and child1_node and conv4 = child2_node
                    conv1 and conv4 should have strides = [2,2]

                    Add2 = upper_add_node
                    Conv5 = upper_conv_node it should have strides = [1,1]
                    Relu2 = upper_relu_node

                    Now add Maxpool between Relu2 and Add2
            """
            if args.imp_strides_opt and node.op == "Add" and len(node.inputs)==2:
                add_node = node
                if is_parent_conv(0, add_node) and is_parent_conv(1, add_node):
                    conv_node1 = add_node.inputs[0].inputs[0].i().i()
                    conv_node2 = add_node.inputs[1].inputs[0].i().i()

                    strides1 = conv_node1.attrs["strides"]
                    strides2 = conv_node2.attrs["strides"]

                    if (strides1==[1,1] and strides2==[2,2]) or (strides1==[2,2] and strides2==[1,1]):
                        if strides1==[1,1] and strides2==[2,2]:
                            temp_node = conv_node1
                            conv_node1 = conv_node2
                            conv_node2 = temp_node
                        # conv_node1 has stride [2,2]
                        relu_node = conv_node1.i().i().i()
                        # due to retinanet cases discussed below, instead of taking dql_node at relu_node.o().o() we take QL node at relu_node.o(), please check below case for more clarity
                        relu_ql_node = relu_node.o()

                        if (len(relu_ql_node.outputs[0].outputs)==2):

                            child1_node = relu_ql_node.outputs[0].outputs[0].o()
                            child2_node = relu_ql_node.outputs[0].outputs[1].o()

                            if child1_node.op == "Conv" and child2_node.op == "Conv":

                                if (child1_node.name == conv_node1.name and get_child_name(child2_node) == conv_node2.name) or (child2_node.name == conv_node1.name and get_child_name(child1_node) == conv_node2.name):

                                    if not(child1_node.name == conv_node1.name):
                                        tem = child1_node
                                        child1_node = child2_node
                                        child2_node = tem

                                    if child1_node.attrs["strides"] == [2,2] and child2_node.attrs["strides"] == [2,2]:

                                        upper_add_node = relu_node.i()

                                        if upper_add_node.inputs[0].inputs[0].op == "Relu":

                                            upper_conv_node = upper_add_node.inputs[1].inputs[0].i().i()
                                            upper_relu_node = upper_add_node.inputs[0].inputs[0]

                                        elif upper_add_node.inputs[1].inputs[0].op == "Relu":

                                            upper_conv_node = upper_add_node.inputs[0].inputs[0].i().i()
                                            upper_relu_node = upper_add_node.inputs[1].inputs[0]

                                        else:
                                            continue
                                        if not (upper_conv_node.attrs["strides"] == [1,1]):
                                            continue
                                        else:
                                            #all conditions satisfied
                                            child1_node.attrs["strides"] = [1,1]
                                            child2_node.attrs["strides"] = [1,1]
                                            upper_conv_node.attrs["strides"] = [2,2]

                                            #now add maxpool between upper_relu and upper add
                                            maxpool_attrs = {
                                                "strides":[2,2],
                                                "kernel_shape":[1,1]
                                            }
                                            maxpool_output = gs.Variable(name = "maxpool_output_" + child1_node.name, dtype = np.uint8)

                                            if len(upper_relu_node.outputs[0].outputs) == 1:
                                                maxpool_node = gs.Node(op="MaxPool", name = "maxpool_" + child1_node.name, attrs=maxpool_attrs, inputs = [upper_relu_node.o().o().outputs[0]], outputs = [maxpool_output])
                                            else:
                                                maxpool_node = gs.Node(op="MaxPool", name = "maxpool_" + child1_node.name, attrs=maxpool_attrs, inputs = [upper_relu_node.outputs[0]], outputs = [maxpool_output])

                                            # conv_x_dql_node = child1_node.i()
                                            list2 = [upper_add_node.inputs[0], upper_add_node.inputs[1]]

                                            if upper_relu_node.outputs[0].name == upper_add_node.inputs[0].name:
                                                list2 = [upper_add_node.inputs[1]]
                                                upper_add_node.inputs.clear()
                                                upper_add_node.inputs = [maxpool_output, list2[0]]
                                            elif upper_relu_node.outputs[0].name == upper_add_node.inputs[1].name:
                                                list2 = [upper_add_node.inputs[0]]
                                                upper_add_node.inputs.clear()
                                                upper_add_node.inputs = [list2[0], maxpool_output]
                                            else:
                                                if upper_relu_node.o().o().outputs[0].name == upper_add_node.inputs[0].name:
                                                    list2 = [upper_add_node.inputs[1]]
                                                    upper_add_node.inputs.clear()
                                                    upper_add_node.inputs = [maxpool_output, list2[0]]
                                                elif upper_relu_node.o().o().outputs[0].name == upper_add_node.inputs[1].name:
                                                    list2 = [upper_add_node.inputs[0]]
                                                    upper_add_node.inputs.clear()
                                                    upper_add_node.inputs = [list2[0], maxpool_output]
                                                else:
                                                    print("ERROR in strides optimization")
                                            graph.nodes.append(maxpool_node)

            if node.op == "Gemm":
                gemm_node = node
                if helper.is_child_present(gemm_node, 0, 0) and gemm_node.o().op == "Softmax":
                    continue
                gemm_input_node = gemm_node.i()
                if gemm_input_node.op == "DequantizeLinear":

                    if gemm_node.inputs[1].inputs[0].op == "DequantizeLinear":
                        w_dql_node = gemm_node.inputs[1].inputs[0]
                        is_weight_quantized = True if len(w_dql_node.inputs[0].inputs) == 0 else False
                        if is_weight_quantized:
                            wt_tensor = w_dql_node.inputs[0]
                        else:
                            w_ql_node = w_dql_node.i()
                            wt_tensor = w_ql_node.inputs[0]
                        org = wt_tensor.values
                        new_shape = org.shape + (1,1)
                        new = np.reshape(org, new_shape)
                        if is_weight_quantized:
                            w_dql_node.inputs[0] = gs.Constant(name=w_dql_node.inputs[0].name, values = new.astype(np.int8))
                        else:
                            w_ql_node.inputs[0] = gs.Constant(name=w_ql_node.inputs[0].name, values = new.astype(np.float32))

                    gemm_node.op = "Conv"
                    new_attrs = {
                        "dilations":[1,1],
                        "group":1,
                        "kernel_shape":[1,1],
                        "pads":[0,0,0,0],
                        "strides":[1,1]
                    }
                    gemm_node.attrs = new_attrs
                elif gemm_input_node.op == "Flatten":
                    flatten_node = gemm_input_node
                    flatten_dql_node = flatten_node.i()
                    flatten_dql_node.outputs = flatten_node.outputs
                    flatten_node.outputs.clear()
                    gemm_ql_node = node.o().o()

                    w_dql_node = gemm_node.inputs[1].inputs[0]
                    is_weight_quantized = True if len(w_dql_node.inputs[0].inputs) == 0 else False
                    wt_tensor = w_dql_node.i().inputs[0]
                    if is_weight_quantized:
                        wt_tensor = w_dql_node.i().inputs[0]
                    else:
                        wt_tensor = w_dql_node.i().inputs[0]
                        w_ql_node = w_dql_node.inputs[0]
                        if w_dql_node.i().op == "Clip":
                            w_ql_node = w_dql_node.i(0).i(0)
                            wt_tensor = w_ql_node.inputs[0]
                    org = wt_tensor.values
                    new_shape = org.shape + (1,1)
                    new = np.reshape(org, new_shape)
                    if is_weight_quantized:
                        w_dql_node.inputs[0] = gs.Constant(name=w_dql_node.inputs[0].name, values = new.astype(np.int8))
                    else:
                        w_ql_node.inputs[0] = gs.Constant(name=w_ql_node.inputs[0].name, values = new.astype(np.float32))

                    gemm_node.op = "Conv"
                    new_attrs = {
                        "dilations":[1,1],
                        "group":1,
                        "kernel_shape":[1,1],
                        "pads":[0,0,0,0],
                        "strides":[1,1]
                    }
                    gemm_node.attrs = new_attrs

                    squeeze_dim = [2, 3]
                    Y1 = gs.Variable(name="sq_output_" + node.name, dtype=np.uint8)
                    parent_node = gemm_ql_node if node.o().op == "Relu" else node

                    X1 = parent_node.outputs[0]
                    X2 = gs.Constant(name="axes" + node.name, values=(np.array(squeeze_dim)).astype(np.int64))

                    squeeze_node = gs.Node(op="Squeeze", name="squeeze_node_" + node.name, inputs=[X1, X2], outputs=[Y1])

                    gemm_ql_node.o().inputs[0] = squeeze_node.outputs[0]

                    graph.nodes.append(squeeze_node)

            if node.op == "Reshape":
                reshape_node = node
                reshape_child_node = reshape_node.o()

                if reshape_child_node.op == "Gemm":

                    """
                    Removing a pattern in Resent50v1.5 model

                               DQL--------------
                                |               |
                                |               |
                                |               V
                                |               Shape
                                |               |
                                |               |
                                |               V
                                |               Gather
                                |               |
                                |               |
                                |               V
                                |               Unsqueeze
                                |               |
                                |               |
                                |               V
                                |               Concat
                                |               |
                                |               |
                                |               V
                                Reshape<--------
                                |
                                |                           DQL                     DQL
                                |                           |                       |
                                |                           |                       |
                                V                           |                       |
                                Gemm<------------------------------------------------
                                |
                                |
                                |
                                QL


                                Connect DQL directly to Gemm and change Gemm to Conv node

                    """
                    gemm_node = reshape_child_node
                    DQL_node = reshape_node.i()

                    DQL_node.outputs = reshape_node.outputs
                    reshape_node.outputs.clear()

                    gemm_DQL_node = gemm_node.inputs[1].inputs[0]
                    gemm_QL_node = gemm_DQL_node.i()

                    w_tensor = gemm_QL_node.inputs[0]
                    original = w_tensor.values
                    new_shape = original.shape + (1,1)
                    new = np.reshape(original, new_shape)
                    gemm_QL_node.inputs[0] = gs.Constant(name= gemm_QL_node.inputs[0].name , values=new.astype(np.float32))

                    new_attrs = {
                        "dilations":[1,1],
                        "group":1,
                        "kernel_shape":[1,1],
                        "pads":[0,0,0,0],
                        "strides":[1,1]
                    }
                    gemm_node.attrs = new_attrs
                    gemm_node.op = "Conv"

                elif reshape_child_node.op == "QuantizeLinear":
                    reshape_parent_node = reshape_node.i()
                    if reshape_parent_node.op == "DequantizeLinear":
                        if len(reshape_parent_node.inputs[0].inputs) == 1 and len(reshape_child_node.outputs[0].outputs) == 1: # 1 parent and 1 child

                            """
                            Node1-------->QL------>Reshape----->DQL---------->Node2

                            is changed to

                            Node1------>Node2
                            """
                            pp = reshape_parent_node.i()
                            cc = reshape_child_node.o()
                            pp.outputs = reshape_child_node.outputs
                            reshape_child_node.outputs.clear()
                        else:
                            # if there is any other connection to the QL or DQL node, remove only the reshape node, let QL and DQL as is
                            reshape_parent_node.outputs = reshape_node.outputs
                            reshape_node.outputs.clear()

                elif reshape_child_node.op == "Transpose":
                    if helper.is_parent_exist(reshape_node, 0, 0) and reshape_node.i().op == "DequantizeLinear":
                        if helper.is_parent_exist(reshape_node.i(), 0, 0) and reshape_node.i().i().op == "QuantizeLinear":
                            new_shape = reshape_node.inputs[1].values
                            p1 = reshape_node.i()
                            p2 = p1.i()
                            if helper.is_constant_tensor(p2.inputs[0]):
                                p2.inputs[0].values = np.reshape(p2.inputs[0].values, new_shape)
                                p1.outputs = reshape_node.outputs
                                reshape_node.outputs.clear()

                elif reshape_child_node.op == "Add":
                    if reshape_child_node.i().op == "Conv":
                        conv_node = reshape_child_node.i()
                        conv_node.inputs = [conv_node.inputs[0], conv_node.inputs[1], reshape_node.inputs[0]]
                        reshape_node.inputs.clear

                        conv_node.outputs = reshape_child_node.outputs
                        reshape_child_node.outputs.clear()
                elif reshape_child_node.op == "Conv":
                    reshape_node.i().outputs = reshape_node.outputs
                    reshape_node.outputs.clear()
                elif reshape_node.i().i().op == "Conv":
                    reshape_node.i().outputs = reshape_node.outputs
                    reshape_node.outputs.clear()

            if node.op == "Clip":
                clip_num = clip_num + 1
                if helper.is_parent_exist(node, 0, 0) and (node.i().op == "Conv" or node.i().op == "Add"):
                    if helper.is_child_present(node, 0, 0) and node.o().op == "QuantizeLinear":
                        clip_node = node
                        clip_max = clip_node.inputs[2].values

                        p1 = clip_node.i()
                        c1 = clip_node.o()

                        scale = c1.inputs[1].values
                        new_clip_max_tensor = gs.Constant(name=clip_node.inputs[2].name+"_"+str(clip_num), values=(np.asarray(clip_max/scale)).astype(np.int8))
                        new_clip_min_tensor = gs.Constant(name=clip_node.inputs[1].name+"_"+str(clip_num), values=clip_node.inputs[1].values.astype(np.int8))
                        clip_node.inputs[2] = new_clip_max_tensor
                        clip_node.inputs[1] = new_clip_min_tensor

                        # p1---->Clip ------>c1----->c2
                        # becomes
                        # p1---->c1-----> Clip----->c2
                        # p1 = conv/add, c1 = QL, c2 = anything
                        if helper.is_child_present(c1, 0, 0):
                            c2 = c1.o()
                            c1.inputs = [p1.outputs[0], c1.inputs[1], c1.inputs[2]]
                            clip_node.inputs = [c1.outputs[0], clip_node.inputs[1], clip_node.inputs[2]]
                            c2.inputs = [clip_node.outputs[0], c2.inputs[1], c2.inputs[2]]
                        else:
                            # p1---->Clip ------>c1---->graph.outputs
                            # becomes
                            # p1---->c1-----> Clip---->graph.outputs
                            c1.inputs = [p1.outputs[0], c1.inputs[1], c1.inputs[2]]
                            clip_node.inputs = [c1.outputs[0], clip_node.inputs[1], clip_node.inputs[2]]

                            clip_node.outputs[0].dtype = "int8"
                            graph.outputs[0] = clip_node.outputs[0]

            if node.op == "Transpose":
                tranpose_node = node
                if helper.is_parent_exist(tranpose_node, 0, 0) and tranpose_node.i().op == "DequantizeLinear":
                    if helper.is_parent_exist(tranpose_node.i(), 0, 0) and tranpose_node.i().i().op == "QuantizeLinear":
                        td = tranpose_node.i()
                        tq = td.i()

                        if helper.is_constant_tensor(tq.inputs[0]):
                            tq.inputs[0].values = np.transpose(tq.inputs[0].values, (3,2,0,1))
                        else:
                            tq.inputs[0].shape = [None, 3, 224, 224]
                        td.outputs = tranpose_node.outputs
                        tranpose_node.outputs.clear()

            if node.op == "Squeeze":
                if helper.is_parent_exist(node,0,0) and node.i().op == "GlobalAveragePool":
                    squeeze_node = node
                    p1 = squeeze_node.i()

                    if helper.is_child_present(squeeze_node, 0, 0) and squeeze_node.o().op == "Mul":
                        mul_node = squeeze_node.o()

                        if helper.is_child_present(mul_node, 0, 0) and mul_node.o().op == "QuantizeLinear":
                            ql_node = mul_node.o()

                            if helper.is_child_present(ql_node, 0, 0) and ql_node.o().op == "DequantizeLinear":
                                dql_node = ql_node.o()

                                # GlobalAveragPool ---> Squeeze ---> Mul ---> QL ---> DQL
                                # becomes
                                # GlobalAveragPool --->QL ---> DQL
                                ql_node.inputs[0] = p1.outputs[0]
                                mul_node.outputs.clear()

            if node.op == "Mul":
                # Remove Mul node
                mul_node = node
                if helper.is_parent_exist(mul_node, 0, 0) and helper.is_child_present(mul_node, 0, 0):
                    average_pool_node = mul_node.i()
                    average_pool_node.outputs = mul_node.outputs
                    mul_node.outputs.clear()

            if node.op == "Pad":
                # Remove Pad node
                pad_node = node
                if len(pad_node.inputs) == 2:
                    nl,cl,hl,wl,nr,cr,hr,wr = pad_node.inputs[1].values

                    if helper.is_child_present(pad_node, 0 ,0) and pad_node.o().op == "Conv":
                        conv_child_node = pad_node.o()
                        conv_child_node.attrs['pads'] = hl,wl,hr,wr

                DQL_node = pad_node.i()
                DQL_node.outputs = pad_node.outputs
                pad_node.outputs.clear()

            # TODO: Add a condition, if input size == Averagepool's kernel shape, only then change it to GlobalAveragePool not all, for time being adding a flag for it
            if node.op == "AveragePool" and args.change_avgpool:
                # Change AveragePool node toGlobalAveragePool Node.
                node.op = "GlobalAveragePool"

            if node.op == "Flatten":
                flatten_node = node
                if helper.is_parent_exist(flatten_node, 0, 0) and flatten_node.i().op == "DequantizeLinear":
                    dql_node = flatten_node.i()
                    if helper.is_child_present(flatten_node, 0, 0) and flatten_node.o().op == "QuantizeLinear":
                        ql_node = flatten_node.o()
                        # don't remove this Flatten node in VGG model. as input is Maxpool (producing 4d tensor) and Child is Matmul expecting 2d tensor.
                        # this child matmul is retained (not converted to Conv) due to Add node after this Matmul which has 2nd input as 2d tensor. thus two 2d tensors will get added.
                        if helper.is_child_present(dql_node, 0, 0) and dql_node.i().op == "MaxPool":
                            continue
                        # node1--->DQL--->Flatten---->QL----->node2
                        #           becomes
                        # node1 ----> node2
                        node1 =  dql_node.i()
                        node1.outputs = ql_node.outputs
                        ql_node.outputs.clear()

                if helper.is_parent_exist(flatten_node, 0, 0) and flatten_node.i().op == "Relu":
                    relu_node1 = flatten_node.i()
                    if helper.is_child_present(flatten_node, 0, 0) and flatten_node.o().op == "QuantizeLinear":
                        relu_node1.outputs = flatten_node.outputs
                        flatten_node.outputs.clear()

            if node.op == "Relu":
                relu_node = node
                if helper.is_parent_exist(relu_node, 0, 0) and relu_node.i().op == "DequantizeLinear":
                    dql_node = relu_node.i()
                    if helper.is_child_present(relu_node, 0, 0) and relu_node.o().op == "QuantizeLinear":
                        ql_node = relu_node.o()
                        node1 =  dql_node.i()
                        node2 = ql_node.o()
                        if node1.op == "QuantizeLinear" and (not (node1.i()).op == "Add"):
                            #if node1 produces u8 output then Relu can also be removed, but if it creates s8 output then Relu should be retained
                            if node1.inputs[2].values.dtype == np.uint8:
                                # node1--->DQL--->Relu---->QL----->node2
                                #           becomes
                                # node1 ----> node2
                                for i in range(len(node2.inputs)):
                                    if node2.inputs[i].name == ql_node.ouputs[0].name:
                                        node2.inputs[i] = node1.outputs[0]
                                        ql_node.outputs.clear()
                            else:
                                # node1--->DQL--->Relu---->QL----->node2
                                #           becomes
                                # node1 ----> Relu ---> node2

                                #relu has single input
                                relu_node.inputs = node1.outputs
                                relu_node.outputs = ql_node.outputs
                                ql_node.outputs.clear()

                        if node1.op == "QuantizeLinear" and (node1.i()).op == "Add":

                            # Add -----> node1 ------> DQL -----> Relu ------> QL -------> node2
                            # becomes
                            # Add ----> node1 ------> Relu ------> node2
                            node1.outputs = dql_node.outputs
                            dql_node.outputs.clear()

                            relu_node.outputs = ql_node.outputs
                            ql_node.outputs.clear()

            if node.op == "MaxPool":
                if helper.is_parent_exist(node, 0, 0) and helper.is_child_present(node, 0, 0):
                    parent_node = node.i()
                    child_node = node.o()
                    if len(parent_node.outputs[0].outputs) == 1 and parent_node.op == "DequantizeLinear" and child_node.op == "QuantizeLinear":
                        dql_node = parent_node
                        dql_parent = dql_node.i()
                        dql_parent.outputs = dql_node.outputs
                        dql_node.outputs.clear()

                        ql_node = child_node
                        node.outputs = ql_node.outputs
                        ql_node.outputs.clear()
                    elif len(parent_node.outputs[0].outputs) == 1 and parent_node.op == "DequantizeLinear" and child_node.op == "Conv":
                        dql_node = parent_node
                        dql_parent = dql_node.i()
                        node.inputs[0] = dql_parent.outputs[0]

                        conv_node1 = child_node
                        dql_node.inputs[0] = node.outputs[0]
                        conv_node1.inputs[0] = dql_node.outputs[0]

            # add Squeeze as input to last DequantizeLinear node
            if squeeze_output and (not args.is_retinanet) and node.op == "DequantizeLinear" and ((len(node.outputs[0].outputs) == 0) or (len(node.outputs[0].outputs)==1 and (node.o().op == "Add" or node.o().op == "Softmax") and len(node.o().outputs[0].outputs)==0)):

                # no need to add Squeeze node if DQL is already getting 2d tensor
                # TODO: add a check if input is 2d then don't add Squeeze node
                # retain_matmul condition is sufficient to ensure Matmul will be present (not converted to conv) and it will give 2d tensor
                if (retain_matmul):
                    continue

                squeeze_dim = [2, 3]

                Y1 = gs.Variable(name="sq_output" + node.name, dtype=np.int8)
                parent_node = node.i()

                X1 = parent_node.outputs[0]
                X2 = gs.Constant(name="axes" + node.name, values=(np.array(squeeze_dim)).astype(np.int64))

                squeeze_node = gs.Node(op="Squeeze", name="squeeze_node" + node.name, inputs=[X1, X2], outputs=[Y1])

                node.inputs[0] = squeeze_node.outputs[0]
                graph.nodes.append(squeeze_node)

            # Retinanet case
            if node.op == "DequantizeLinear":
                if helper.is_parent_exist(node, 0, 0):
                    dql_parent = node.i()
                if len(node.outputs) > 0  and len(node.outputs[0].outputs) > 1:
                    for i in range(len(node.outputs[0].outputs)):
                        # node.outputs[0].outputs[0].op is used instead of node.outputs[0].outputs[i].op because in each pass 1 child is removed
                        if node.outputs[0].outputs[0].op == "Shape" or node.outputs[0].outputs[0].op == "Add" or is_any_output_tensor_graph_output(node) or node.outputs[0].outputs[0].op == "Conv" or node.outputs[0].outputs[0].op == "Relu" or node.outputs[0].outputs[0].op == "Resize":
                            child_node = node.outputs[0].outputs[0]
                            s = gs.Constant(name=node.inputs[1].name + "_" + str(i), values=(node.inputs[1].values).astype(np.float32))
                            zp = gs.Constant(name=node.inputs[2].name + "_" + str(i), values=(node.inputs[2].values).astype(node.inputs[2].dtype))
                            y = gs.Variable(name=node.outputs[0].name + "_" + str(i), dtype=node.inputs[2].dtype)
                            new_dql_node = gs.Node(op = "DequantizeLinear", name = node.name + "_" + str(i),  inputs = [node.i().outputs[0], s, zp], outputs = [y])

                            for j in range(len(child_node.inputs)):
                                if child_node.inputs[j].name == node.outputs[0].name:
                                    child_node.inputs[j] = new_dql_node.outputs[0]
                            graph.nodes.append(new_dql_node)

                    # QL                                        QL-------DQL-------Conv
                    # |                                         | \
                    # |                                         |   \
                    # DQL---------conv gets converted to        DQL  \
                    # |                                         |    DQL
                    # |                                         |
                    # Conv                                      Conv
                    # this extra DQL needs to be removed, when later we do graph.cleanup() this node gets removed but before cleanup if any case needs QL childs it will reflect 3 childs

                    for i in range(len(dql_parent.outputs[0].outputs)):
                        child_node = dql_parent.outputs[0].outputs[i]
                        if not helper.is_child_present(child_node, 0, 0) and not is_any_output_tensor_graph_output(child_node):
                            child_node.inputs.clear()
                            break

            if node.op == "Gather" and node.o().op == "Transpose":
                gather_node = node
                transpose_node = gather_node.o()
                gather_dql_node = gather_node.i()
                gather_ql_node = gather_dql_node.i()
                if gather_ql_node.op == "Clip" and gather_dql_node.i().i().op == "QuantizeLinear":
                    gather_ql_node = gather_dql_node.i().i()
                transpose_conv_node = transpose_node.o()
                #                   QL                           QL
                #                   |                            |
                #                   |                            Clip
                #                   |                            |
                #                   DQL                          DQL
                #                   |                            |
                #                   |                            |
                #         ---------Gather       OR     ---------Gather
                #                   |                            |
                #                   |                            |
                #                   Transpose                    Transpose
                #                   |                            |
                #                   |                            |
                #                   Conv                         Conv

                # is changed to


                #                   QL
                #                   |
                #                   |
                #      ------------Gather
                #                   |
                #                   |
                #                   Transpose
                #                   |
                #                   |
                #                   DQL
                #                   |
                #                   |
                #                   Conv
                gather_dql_node_inputs = gather_dql_node.inputs
                gather_node.inputs[0] = gather_ql_node.outputs[0]

                gather_dql_node_inputs[0] = transpose_node.outputs[0]
                transpose_conv_node.inputs[0] = gather_dql_node.outputs[0]
                gather_dql_node.inputs = gather_dql_node_inputs


            if node.op == "Conv":
                ctr = ctr + 1
                conv_node = node
                if len(conv_node.outputs[0].outputs) == 4:
                    if not (conv_node.outputs[0].outputs[0].op == "Shape" and conv_node.outputs[0].outputs[1].op == "Shape" and conv_node.outputs[0].outputs[2].op == "Shape" and conv_node.outputs[0].outputs[3].op == "Reshape"):
                        continue
                    shape_node1 = conv_node.outputs[0].outputs[0]
                    shape_node2 = conv_node.outputs[0].outputs[1]
                    shape_node3 = conv_node.outputs[0].outputs[2]
                    reshape_node = conv_node.outputs[0].outputs[3]
                    if helper.is_child_present(reshape_node, 0, 0) and reshape_node.o().op == "Transpose":
                        if helper.is_child_present(reshape_node.o(), 0, 0) and reshape_node.o().o().op == "Reshape":
                            if helper.is_child_present(reshape_node.o().o(), 0, 0) and reshape_node.o().o().o().op == "Concat":
                                if helper.is_child_present(reshape_node.o().o().o(), 0, 0) and reshape_node.o().o().o().o().op == "QuantizeLinear":
                                    ret_ql_node = reshape_node.o().o().o().o()
                                    # this is retinaNet pattern at the end
                                    # Conv--->Reshape--->Transpose--->Reshape--->Concat--->QL---->node2
                                    # will be made as Conv--->QL---->Rehsape---->Transpose---->Reshape---->Concat---->QL----->node2
                                    # later QL at end will also be removed
                                    s_ql = gs.Constant(name=ret_ql_node.inputs[1].name + "_" + str(ctr), values=(ret_ql_node.inputs[1].values).astype(np.float32))
                                    zp_ql = gs.Constant(name=ret_ql_node.inputs[2].name + "_" + str(ctr), values=(ret_ql_node.inputs[2].values).astype(np.int8))
                                    y_ql = gs.Variable(name=ret_ql_node.outputs[0].name + "_" + str(ctr), dtype=np.int8)
                                    new_ql_node = gs.Node(op = "QuantizeLinear", name = ret_ql_node.name + "_" + str(ctr),  inputs = [conv_node.outputs[0], s_ql, zp_ql], outputs = [y_ql])
                                    reshape_node.inputs[0] = new_ql_node.outputs[0]
                                    shape_node1.inputs[0] = new_ql_node.outputs[0]
                                    shape_node2.inputs[0] = new_ql_node.outputs[0]
                                    shape_node3.inputs[0] = new_ql_node.outputs[0]
                                    graph.nodes.append(new_ql_node)
                                    retinanet_end_pattern_found = True

            if node.op == "QuantizeLinear" and retinanet_end_pattern_found:
                if helper.is_parent_exist(node, 0, 0) and node.i().op == "Concat":
                    if helper.is_child_present(node, 0, 0) and node.o().op == "DequantizeLinear":
                        # remove the QL node as mentioned in the above condition. (Part of retinaNet model)
                        # Concat------>QL -------> DQL is changed to
                        # Concat------>DQL
                        node.i().outputs = node.outputs
                        node.outputs.clear()


            if node.op == "Unsqueeze":
                unsqueeze_node = node
                if helper.is_parent_exist(unsqueeze_node, 0, 0) and unsqueeze_node.i().op == "Gather":
                    if helper.is_parent_exist(unsqueeze_node.i(), 0, 0) and unsqueeze_node.i().i().op == "Shape":
                        if helper.is_parent_exist(unsqueeze_node.i().i(), 0, 0) and unsqueeze_node.i().i().i().op == "QuantizeLinear":
                            if helper.is_child_present(unsqueeze_node, 0, 0) and unsqueeze_node.o().op == "Concat":

                                # QL-------> Shape------> Gather ------>Unsqueeze------> Concat is changed to
                                # QL-------> Shape------> Gather ------>Unsqueeze------> Cast----->Concat

                                concat_node = unsqueeze_node.o()
                                cast_count += 1
                                cast_node_name = node.name + "_" + str(cast_count)
                                cast_output_tensor = gs.Variable(name=cast_node_name + "_output", dtype=np.int64)
                                new_cast_node = gs.Node(op = "Cast", name = cast_node_name,  attrs = {"to":getattr(TensorProto, "INT64")}, inputs = [unsqueeze_node.outputs[0]], outputs = [cast_output_tensor])

                                for i in range(len(concat_node.inputs)):
                                    if concat_node.inputs[i].name == node.outputs[0].name:
                                        concat_node.inputs[i] = new_cast_node.outputs[0]
                                        break
                                graph.nodes.append(new_cast_node)

        graph.cleanup()

        if args.save_opt_qdq:
            onnx.save(gs.export_onnx(graph), "optimized_qdq_" + onnx_model_name)
            print("Optimized QDQ model has been saved")

        node_list = []
        initializer_list = []
        node_count = 0
        maxpool_count = 0
        conv_count = 0

        def is_all_concat_input_dql(node):
            for i in range(len(node.inputs)):
                if helper.is_parent_exist(node, i, 0) and  node.inputs[i].inputs[0].op != "DequantizeLinear":
                    return False
            return True

        def concat_input_not_constant(node):
            for i in range(len(node.inputs)):
                if len(node.inputs[i].inputs) == 0:
                    return True
            return False


        def all_dql_conditions_satisfy(node):
            has_output_ternsor = len(node.outputs) > 0
            has_no_child = has_output_ternsor and len(node.outputs[0].outputs)==0
            has_child = helper.is_child_present(node, 0, 0)
            child_is_add_node = False
            child_has_no_child = False
            child_is_averagepool_node = False
            child_add_node_has_no_2nd_input = False

            if has_child:
                child_is_add_node = node.o().op == "Add"
                child_is_softmax_node = node.o().op == "Softmax"
                child_has_no_child = len(node.o().outputs[0].outputs)==0
                child_is_averagepool_node = node.o().op == "AveragePool"
                child_is_lrn_node = node.o().op == "LRN"
                child_is_gemm_node = node.o().op == "Gemm"
                child_is_relu_node = node.o().op == "Relu"
                child_is_shape_node = node.o().op == "Shape"
                child_is_slice_node = node.o().op == "Slice"
                child_is_resize_node = node.o().op == "Resize"
                child_is_reshape_node = node.o().op == "Reshape"

                if child_is_add_node:
                    child_add_node = node.o()
                    if len(child_add_node.inputs[1].inputs) == 0:
                        child_add_node_has_no_2nd_input = True

            if not has_output_ternsor:
                return False

            if has_output_ternsor and is_any_output_tensor_graph_output(node):
                return True

            if has_no_child:
                return True

            if child_is_add_node and child_add_node_has_no_2nd_input:
                return True

            if child_is_softmax_node and child_has_no_child:
                return True

            if child_is_averagepool_node:
                return True

            if child_is_lrn_node:
                return True

            if child_is_gemm_node:
                return True

            if child_is_relu_node:
                return True

            if child_is_shape_node or child_is_slice_node or child_is_resize_node:
                return True

            if helper.is_child_present(node, 0, 1):
                c2 = node.outputs[0].outputs[1]
                if c2.op == "Shape" or c2.op == "Resize":
                    return True

            if helper.is_child_present(node, 0, 2):
                c2 = node.outputs[0].outputs[2]
                if c2.op == "Shape" or c2.op == "Resize":
                    return True

            if child_is_reshape_node:
                if helper.is_child_present(node.o(), 0, 0) and node.o().o().op == "Softmax":
                    return True

            return False

        def all_ql_conditions_satify(count, node):
            if args.is_ryzenai_model and count == 2:
                return True
            if helper.is_child_present(node, 0, 0):
                if node.o().op == "Gather":
                    return False
                if helper.is_child_present(node.o(), 0, 0) and node.o().o().op == "Gemm" and len(node.inputs[0].inputs) == 0:
                    return True
            if count == 0:
                if args.is_ryzenai_model and helper.is_child_present(node, 0, 0) and node.o().op == "DequantizeLinear":
                    if helper.is_child_present(node.o(), 0,0) and node.o().o().op == "Conv":
                        if helper.is_child_present(node.o().o(), 0, 0) and node.o().o().o().op == "QuantizeLinear":
                            return False
                return True
            has_parent = helper.is_parent_exist(node, 0, 0)

            if has_parent:
                is_parent_averagepool = node.i().op == "AveragePool"
                is_parent_lrn_node = node.i().op == "LRN"
                is_parent_maxpool_node = node.i().op == "MaxPool"
                is_parent_relu_node = node.i().op == "Relu"
                is_parent_resize_node = node.i().op == "Resize"
                is_parent_concat = node.i().op == "Concat"

                if is_parent_averagepool or is_parent_lrn_node:
                    return True

                if is_parent_maxpool_node:
                    # (Non DQL)--->MaxPool----->QL (keep this QL)
                    if not (node.i().i().op == "DequantizeLinear"):
                        return True
                if is_parent_relu_node:
                    parent_relu_node = node.i()
                    if parent_relu_node.i().op == "Add":
                        parent_add_node = parent_relu_node.i()
                        if len(parent_add_node.inputs[1].inputs)==0:
                            return True

                if is_parent_resize_node:
                    return True
                #if is_parent_concat:
                #    return True

            if helper.is_child_present(node, 0, 0):
                if helper.is_parent_exist(node, 0, 0):
                    if node.i().op == "Relu":
                        return False

            return False

        for node in graph.nodes:

            if node.op == "Conv":
                QLinearConv_node = QLinearConv(node, aecg_zendnn_opt, args.remove_relu, conv_count)
                node_list.append(QLinearConv_node.get_node())
                initializer_list.append(QLinearConv_node.get_intializers())
                conv_count = conv_count + 1
            elif node.op == "QuantizeLinear" and all_ql_conditions_satify(node_count, node):
                QuantizeLinear_node = QuantizeLinear(node)
                node_list.append(QuantizeLinear_node.get_node())
                initializer_list.append(QuantizeLinear_node.get_intializers())
            elif node.op == "DequantizeLinear" and all_dql_conditions_satisfy(node):
                DequantizeLinear_node = DequantizeLinear(node, aecg_zendnn_opt, args.remove_relu)
                node_list.append(DequantizeLinear_node.get_node())
                initializer_list.append(DequantizeLinear_node.get_intializers())
            elif node.op == "MaxPool":
                maxpool_node = MaxPool(node, maxpool_count, args.remove_relu)
                node_list.append(maxpool_node.get_node())
                maxpool_count = maxpool_count + 1
            elif node.op == "Add":
                add_node = QLinearAdd(node, aecg_zendnn_opt, args.remove_relu)
                node_list.append(add_node.get_node())
                initializer_list.append(add_node.get_intializers())
            elif node.op == "AveragePool":
                average_pool_node = AveragePool(node)
                node_list.append(average_pool_node.get_node())
            elif node.op == "Squeeze":
                squeeze_node = Squeeze(node)
                node_list.append(squeeze_node.get_node())
                initializer_list.append(squeeze_node.get_intializers())
            elif node.op == "GlobalAveragePool":
                global_average_pool_node = GlobalAveragePool(node, aecg_zendnn_opt, args.remove_relu)
                node_list.append(global_average_pool_node.get_node())
                initializer_list.append(global_average_pool_node.get_intializers())
            elif node.op == "Flatten":
                flatten_node = Flatten(node)
                node_list.append(flatten_node.get_node())
            elif node.op == "MatMul":
                if retain_matmul:
                    matmul_node = MatMul_Retained(node)
                    node_list.append(matmul_node.get_node())
                    initializer_list.append(matmul_node.get_intializers())
                else:
                    matmul_node = MatMul(node)
                    node_list.append(matmul_node.get_node())
                    initializer_list.append(matmul_node.get_intializers())
            elif node.op == "LRN":
                lrn_node = LRN(node)
                node_list.append(lrn_node.get_node())
            elif node.op == "Concat":
                concat_node = Concat(node, is_all_concat_input_dql(node))
                node_list.append(concat_node.get_node())
                if (is_all_concat_input_dql(node) or concat_input_not_constant(node)):
                    initializer_list.append(concat_node.get_intializers())
            elif node.op == "Softmax":
                softmax_node = Softmax(node)
                node_list.append(softmax_node.get_node())
            elif node.op == "Cast":
                cast_node = Cast(node)
                node_list.append(cast_node.get_node())
            elif node.op == "Gather":
                gather_node = Gather(node)
                node_list.append(gather_node.get_node())
                initializer_list.append(gather_node.get_intializers())
            elif node.op == "Gemm":
                # If weights and bias are dequantized, embed it in Gemm
                if node.i(0).op == "DequantizeLinear" and node.i(1).op == "DequantizeLinear" and node.i(2).op == "DequantizeLinear":
                    dql_node1 = node.i(1).name
                    dql_node2 = node.i(2).name
                    ql_node1 = node.i(1).i(0).name
                    dql_list = [dql_node1, dql_node2, ql_node1]
                    dql_found = []
                    gemm_node = Gemm_optimized(node)
                    for node_current in node_list:
                        if node_current.name in dql_list:
                            dql_found.append(node_current)
                    for node_dql in dql_found:
                        node_list.remove(node_dql)
                    node_list.append(gemm_node.get_node())
                    initializer_list.append(gemm_node.get_intializers())
                else:
                    gemm_node = Gemm(node)
                    node_list.append(gemm_node.get_node())
            elif node.op == "Greater":
                greater_node = Greater(node)
                node_list.append(greater_node.get_node())
                initializer_list.append(greater_node.get_intializers())
            elif node.op == "Less":
                less_node = Less(node)
                node_list.append(less_node.get_node())
                initializer_list.append(less_node.get_intializers())
            elif node.op == "Slice":
                slice_node = Slice(node)
                node_list.append(slice_node.get_node())
                initializer_list.append(slice_node.get_intializers())
            elif node.op == "Transpose":
                transpose_node = Transpose(node)
                node_list.append(transpose_node.get_node())
            elif node.op == "Relu":
                if not args.remove_relu:
                    relu_node = Relu(node)
                    node_list.append(relu_node.get_node())
            elif node.op == "Reshape":
                reshape_node = Reshape(node)
                node_list.append(reshape_node.get_node())
                initializer_list.append(reshape_node.get_intializers())
            elif node.op == "Shape":
                shape_node = Shape(node)
                node_list.append(shape_node.get_node())
            elif node.op == "Resize":
                resize_node = Resize(node)
                node_list.append(resize_node.get_node())
                initializer_list.append(resize_node.get_intializers())
            elif node.op == "Unsqueeze":
                unsq_node = Unsqueeze(node)
                node_list.append(unsq_node.get_node())
                initializer_list.append(unsq_node.get_intializers())
            elif node.op == "Clip":
                found = False
                for node_current in node_list:
                    if node_current.name == node.i(0).name:
                        found = True
                if found == False:
                    continue
                clip_node = Clip(node)
                node_list.append(clip_node.get_node())
                initializer_list.append(clip_node.get_intializers())

            if node.op in supported_op:
                node_count = node_count + 1

        new_list = []
        for list1 in initializer_list:
            for i in list1:
                new_list.append(i)

        graph_input_shape = graph.inputs[0].shape
        graph_input_shape[0] = None

        if graph.inputs[0].dtype == "float32":
            grapth_input_tensor_dtype = onnx.TensorProto.FLOAT
        elif graph.inputs[0].dtype == "int8":
            grapth_input_tensor_dtype = onnx.TensorProto.INT8
        elif graph.inputs[0].dtype == "int64":
            grapth_input_tensor_dtype = onnx.TensorProto.INT64
        X = onnx.helper.make_tensor_value_info(graph.inputs[0].name,
                                            grapth_input_tensor_dtype,
                                            graph_input_shape)
        graph_output_tensor_list = []
        for i in range(len(graph.outputs)):
            if graph.outputs[i].dtype == "float32":
                grapth_output_tensor_dtype = onnx.TensorProto.FLOAT
            elif graph.outputs[i].dtype == "int8":
                grapth_output_tensor_dtype = onnx.TensorProto.INT8
            elif graph.outputs[i].dtype == "bool":
                grapth_output_tensor_dtype = onnx.TensorProto.BOOL

            graph_output_shape = graph.outputs[i].shape

            Y = onnx.helper.make_tensor_value_info(graph.outputs[i].name,
                                            grapth_output_tensor_dtype,
                                            graph_output_shape)
            graph_output_tensor_list.append(Y)

        graph_def = onnx.helper.make_graph(nodes=node_list, name=graph.name,
                                            inputs=[X],
                                            outputs=graph_output_tensor_list,
                                            initializer=new_list)

        model_def = onnx.helper.make_model(graph_def, producer_name="onnx-example")
        model_def.opset_import[0].version = 16
        model_qop = ModelWrapper(model_def)
        return (model_qop, False)
