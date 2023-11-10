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
from qonnx.custom_op.qop.squeeze_op import *
from qonnx.custom_op.qop.flatten_op import *
from qonnx.custom_op.qop.concat_op import *
from qonnx.custom_op.qop.softmax_op import *
from qonnx.custom_op.qop.cast_op import *
from qonnx.custom_op.qop.gather_op import *
from qonnx.custom_op.qop.gemm_op import *
from qonnx.custom_op.qop.gemm_op_optimized import *
from qonnx.custom_op.qop.greater_op import *
from qonnx.custom_op.qop.less_op import *
from qonnx.custom_op.qop.slice_op import *
from qonnx.custom_op.qop.transpose_op import *
from qonnx.custom_op.qop.relu_op import *
from qonnx.custom_op.qop.clip_op import *

class CustomEnv():
    remove_relu=True
    def __init__(self):
        pass

class QCDQToQOp(Transformation):

    def __init__(self) -> None:
        super().__init__()

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        args = CustomEnv()
        graph = gs.import_onnx(model.model)

        graph.fold_constants()

        aecg_zendnn_opt = False

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

        supported_op = ["Conv", "QuantizeLinear", "DequantizeLinear", "MaxPool", "Squeeze", "Flatten", "Concat", "Softmax", "Cast", "Gather", "Gemm", "Greater", "Less", "Slice", "Transpose", "Relu", "Clip"]

        maxpool_count = 0
        ctr = 0
        cast_count = 0
        clip_num = 0
        squeeze_output = False
        for node in graph.nodes:

            if node.op == "Flatten":
                squeeze_output = True
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

            if node.op == "Clip":
                clip_num = clip_num + 1
                if helper.is_parent_exist(node, 0, 0) and (node.i().op == "Conv"):
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
                        # p1 = conv, c1 = QL, c2 = anything
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

            if node.op == "Flatten":
                flatten_node = node
                if helper.is_parent_exist(flatten_node, 0, 0) and flatten_node.i().op == "DequantizeLinear":
                    dql_node = flatten_node.i()
                    if helper.is_child_present(flatten_node, 0, 0) and flatten_node.o().op == "QuantizeLinear":
                        ql_node = flatten_node.o()
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
                        if node1.op == "QuantizeLinear":
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
            if squeeze_output and node.op == "DequantizeLinear" and ((len(node.outputs[0].outputs) == 0) or (len(node.outputs[0].outputs)==1 and (node.o().op == "Softmax") and len(node.o().outputs[0].outputs)==0)):

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
                        if is_any_output_tensor_graph_output(node) or node.outputs[0].outputs[0].op == "Conv" or node.outputs[0].outputs[0].op == "Relu":
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

        graph.cleanup()

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
            child_has_no_child = False

            if has_child:
                child_is_softmax_node = node.o().op == "Softmax"
                child_has_no_child = len(node.o().outputs[0].outputs)==0
                child_is_gemm_node = node.o().op == "Gemm"
                child_is_relu_node = node.o().op == "Relu"
                child_is_slice_node = node.o().op == "Slice"

            if not has_output_ternsor:
                return False

            if has_output_ternsor and is_any_output_tensor_graph_output(node):
                return True

            if has_no_child:
                return True

            if child_is_softmax_node and child_has_no_child:
                return True

            if child_is_gemm_node:
                return True

            if child_is_relu_node:
                return True

            if child_is_slice_node:
                return True

            return False

        def all_ql_conditions_satify(count, node):
            if helper.is_child_present(node, 0, 0):
                if node.o().op == "Gather":
                    return False
                if helper.is_child_present(node.o(), 0, 0) and node.o().o().op == "Gemm" and len(node.inputs[0].inputs) == 0:
                    return True
            if count == 0:
                return True
            has_parent = helper.is_parent_exist(node, 0, 0)

            if has_parent:
                is_parent_maxpool_node = node.i().op == "MaxPool"
                is_parent_relu_node = node.i().op == "Relu"
                is_parent_concat = node.i().op == "Concat"

                if is_parent_maxpool_node:
                    # (Non DQL)--->MaxPool----->QL (keep this QL)
                    if not (node.i().i().op == "DequantizeLinear"):
                        return True
                if is_parent_relu_node:
                    parent_relu_node = node.i()

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
            elif node.op == "Squeeze":
                squeeze_node = Squeeze(node)
                node_list.append(squeeze_node.get_node())
                initializer_list.append(squeeze_node.get_intializers())
            elif node.op == "Flatten":
                flatten_node = Flatten(node)
                node_list.append(flatten_node.get_node())
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
