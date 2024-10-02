# Copyright (c) 2020 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import warnings
from onnx import TensorProto, helper

from qonnx.transformation.base import Transformation
from qonnx.transformation.extract_conv_bias import ExtractBiasFromConv
from qonnx.util.basic import auto_pad_to_explicit_padding, get_by_name


class LowerConvsToMatMul(Transformation):
    """Replace Conv layers with pairs of Im2Col-MatMul layers, plus Transpose
    layers to keep the original data layout."""

    def apply(self, model):
        model = model.transform(ExtractBiasFromConv())
        graph = model.graph
        graph_modified = False
        for node_ind, node in enumerate(graph.node, start=1):
            if node.op_type != "Conv":
                continue

            if len(node.input) == 3:
                warnings.warn("Found Conv node with bias, skipping")
                continue

            # extract parameters of node
            (
                cnv_input,
                cnv_output,
                cnv_input_datatype,
                cnv_output_datatype,
                k_h,
                k_w,
                stride_h,
                stride_w,
                group,
                weight_name,
                conv_weight_inp_name,
                conv_weight_q_scale_name,
                W_conv,
                ifm_ch,
                ofm_ch,
                ifm_dim_h,
                ifm_dim_w,
                ofm_dim_h,
                ofm_dim_w,
                dilation,
                pad,
            ) = self.extract_conv_params(model, node)

            if W_conv is None:
                warnings.warn("Found Conv node with non-initialized weight, skipping")
                continue

            # if depthwise conv create sparse matrix and variable "dw"
            # to store as attribute in Im2Col that indicates that the created
            # Im2Col node belongs to a depthwise convolution
            dw = False
            if group == ifm_ch and ofm_ch == ifm_ch:
                W_sparse = np.zeros((ofm_ch, ifm_ch, k_h, k_w))  # (OFM, IFM, k_H, k_W)
                # TODO: if the convolution is quantized with a non-zero zeropoint we
                # should be using the zeropoint value here instead of np.zeros
                for ch in range(ifm_ch):
                    W_sparse[ch][ch] = W_conv[ch][0]  # W_conv = [OFM, IFM, k_H, k_W]
                W_conv = W_sparse.astype(np.float32)
                # we need to store information of the
                # sparsity of the weight matrix. For this
                # we use the sparsity annotation of the
                # weight tensor
                sparsity = {"dw": {"kernel_shape": [k_h, k_w]}}
                model.set_tensor_sparsity(weight_name, sparsity)
                # additionally create variable "dw" to store
                # as attribute in Im2Col that indicates that the created
                # Im2Col node belongs to a depthwise convolution
                dw = True

            # reuse conv weights for new matmul weights
            # conv weights are [OFM][IFM][k][k]
            # first convert to [OFM][k_h][k_w][IFM] (to remain compatible with
            # finn-hlslib and how it does im2col/sliding window)
            W_matmul = W_conv.transpose(0, 2, 3, 1)  # W_conv = [OFM, IFM, k_H, k_W]
            # reshape into [OFM][k_h*k_w*IFM] matrix
            W_matmul = W_matmul.reshape(ofm_ch, ifm_ch * k_h * k_w)
            # transpose to get ONNX-compatible [k_h*k_w*IFM][OFM] matrix
            W_matmul = W_matmul.T
            model.set_initializer(weight_name, W_matmul)
            if weight_name != conv_weight_inp_name:
                # required for convs with quantized weights
                model.set_tensor_shape(conv_weight_inp_name, W_matmul.shape)
            if conv_weight_q_scale_name is not None:
                # required for convs with quantized weights
                scale_weight_q = model.get_initializer(conv_weight_q_scale_name)
                if scale_weight_q.ndim > 0:
                    # scale shape is originally [OFM, IFM, k_H, k_W]
                    # transpose into [OFM, k_H, k_W, IFM]
                    scale_weight_q = scale_weight_q.transpose(0, 2, 3, 1)
                    # reshape into [OFM][k_h*k_w*IFM] matrix
                    scale_weight_q = scale_weight_q.reshape(ofm_ch, -1)
                    # transpose to be shape-compatible with weight matrix
                    scale_weight_q = scale_weight_q.T
                    model.set_initializer(conv_weight_q_scale_name, scale_weight_q)

            # create new intermediate values
            inp_trans_out = helper.make_tensor_value_info(
                model.make_new_valueinfo_name(),
                TensorProto.FLOAT,
                (1, ifm_dim_h, ifm_dim_w, ifm_ch),  # NHWC
            )
            graph.value_info.append(inp_trans_out)
            inp_trans_out = inp_trans_out.name
            model.set_tensor_datatype(inp_trans_out, cnv_input_datatype)

            # k_h=k_w==1: pointwise convolution, thus no im2col needed
            need_im2col = any(p != 0 for p in pad) or k_h != 1 or k_w != 1 or stride_h != 1 or stride_w != 1

            # create new intermediate values
            matmul_out = helper.make_tensor_value_info(
                model.make_new_valueinfo_name(), TensorProto.FLOAT, (1, ofm_dim_h, ofm_dim_w, ofm_ch)
            )
            graph.value_info.append(matmul_out)
            matmul_out = matmul_out.name
            model.set_tensor_datatype(matmul_out, cnv_output_datatype)

            # create new nodes
            # NCHW -> NHWC
            inp_trans_node = helper.make_node("Transpose", [cnv_input], [inp_trans_out], perm=[0, 2, 3, 1])
            nodes_to_insert = [inp_trans_node]

            if need_im2col:
                im2col_out = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, (1, ofm_dim_h, ofm_dim_w, ifm_ch * k_h * k_w)
                )
                graph.value_info.append(im2col_out)
                im2col_out = im2col_out.name
                model.set_tensor_datatype(im2col_out, cnv_input_datatype)
                im2col_node = helper.make_node(
                    "Im2Col",
                    [inp_trans_out],
                    [im2col_out],
                    domain="qonnx.custom_op.general",
                    stride=[stride_h, stride_w],
                    kernel_size=[k_h, k_w],
                    pad_amount=pad,
                    input_shape="(1,{},{},{})".format(ifm_dim_h, ifm_dim_w, ifm_ch),
                    depthwise=dw,
                    dilations=dilation,
                )
                nodes_to_insert.append(im2col_node)

            matmul_input = im2col_out if need_im2col else inp_trans_out
            # do matmul
            matmul_node = helper.make_node("MatMul", [matmul_input, conv_weight_inp_name], [matmul_out])
            # NHWC -> NCHW
            out_trans_node = helper.make_node("Transpose", [matmul_out], [cnv_output], perm=[0, 3, 1, 2])

            nodes_to_insert.extend([matmul_node, out_trans_node])

            # insert nodes where the conv is to preserve topological ordering
            for i, insert_node in enumerate(nodes_to_insert):
                graph.node.insert(node_ind + i, insert_node)
            graph.node.remove(node)

        return (model, graph_modified)

    def extract_conv_params(self, model, node):
        cnv_input = node.input[0]
        cnv_output = node.output[0]
        cnv_input_datatype = model.get_tensor_datatype(cnv_input)
        cnv_output_datatype = model.get_tensor_datatype(cnv_output)
        k_h = get_by_name(node.attribute, "kernel_shape").ints[0]
        k_w = get_by_name(node.attribute, "kernel_shape").ints[1]
        stride_h = get_by_name(node.attribute, "strides").ints[0]
        stride_w = get_by_name(node.attribute, "strides").ints[1]
        group = get_by_name(node.attribute, "group").i
        weight_name = node.input[1]
        conv_weight_inp_name = node.input[1]
        conv_weight_q_scale_name = None
        W_conv = model.get_initializer(weight_name)
        if W_conv is None:
            # check to see if there is an immediate quantizer node feeding the weight input
            w_producer = model.find_producer(weight_name)
            if not (w_producer is None) and w_producer.op_type == "Quant":
                W_conv = model.get_initializer(w_producer.input[0])
                weight_name = w_producer.input[0]
                conv_weight_q_scale_name = w_producer.input[1]
        ifm_ch = model.get_tensor_shape(cnv_input)[1]  # assume NCHW
        ofm_ch = model.get_tensor_shape(cnv_output)[1]  # assume NCHW
        ifm_dim_h = model.get_tensor_shape(cnv_input)[2]  # assume NCHW
        ifm_dim_w = model.get_tensor_shape(cnv_input)[3]  # assume NCHW
        ofm_dim_h = model.get_tensor_shape(cnv_output)[2]  # assume NCHW
        ofm_dim_w = model.get_tensor_shape(cnv_output)[3]  # assume NCHW
        dilation_attr = get_by_name(node.attribute, "dilations")
        dilation = dilation_attr.ints if dilation_attr is not None else [1, 1]  # default value
        auto_pad = get_by_name(node.attribute, "auto_pad")
        if auto_pad is not None:
            auto_pad = auto_pad.s.decode("utf-8")
            if auto_pad == "NOTSET":
                pad = get_by_name(node.attribute, "pads").ints
            else:
                pad = auto_pad_to_explicit_padding(
                    auto_pad, ifm_dim_h, ifm_dim_w, k_h, k_w, stride_h, stride_w, len(model.get_tensor_shape(cnv_input)) - 2
                )
        else:
            pad = get_by_name(node.attribute, "pads").ints

        if len(pad) == 2:  # only one dimension should be padded
            assert ifm_dim_h == 1 or ifm_dim_w == 1, "Padding is assumed to be 1D, image is 2D"

        return (
            cnv_input,
            cnv_output,
            cnv_input_datatype,
            cnv_output_datatype,
            k_h,
            k_w,
            stride_h,
            stride_w,
            group,
            weight_name,
            conv_weight_inp_name,
            conv_weight_q_scale_name,
            W_conv,
            ifm_ch,
            ofm_ch,
            ifm_dim_h,
            ifm_dim_w,
            ofm_dim_h,
            ofm_dim_w,
            dilation,
            pad,
        )
