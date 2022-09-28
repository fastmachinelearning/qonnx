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
from qonnx.util.basic import get_by_name


def _auto_pad_to_explicit_padding(autopad_str, idim_h, idim_w, k_h, k_w, stride_h, stride_w, n_dims):
    pad_total_h = (stride_h - 1) * idim_h - stride_h + k_h
    pad_total_w = (stride_w - 1) * idim_w - stride_w + k_w
    pad_half_small_h = int((pad_total_h / 2))
    pad_half_small_w = int((pad_total_w / 2))
    pad_half_large_h = pad_total_h - pad_half_small_h
    pad_half_large_w = pad_total_w - pad_half_small_w
    if autopad_str == "VALID":
        return [0 for i in range(2 * n_dims)]
    elif autopad_str == "SAME_UPPER":
        return [pad_half_small_h, pad_half_small_w, pad_half_large_h, pad_half_large_w]
    elif autopad_str == "SAME_LOWER":
        return [pad_half_large_h, pad_half_large_w, pad_half_small_h, pad_half_small_w]
    else:
        raise Exception("Unsupported auto_pad: " + autopad_str)


class LowerConvsToMatMul(Transformation):
    """Replace Conv layers with pairs of Im2Col-MatMul layers, plus Transpose
    layers to keep the original data layout."""

    def apply(self, model):
        model = model.transform(ExtractBiasFromConv())
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Conv":
                if len(n.input) == 3:
                    warnings.warn("Found Conv node with bias, skipping")
                    continue
                cnv_input = n.input[0]
                cnv_output = n.output[0]
                idt = model.get_tensor_datatype(cnv_input)
                odt = model.get_tensor_datatype(cnv_output)
                # extract conv parameters
                k = get_by_name(n.attribute, "kernel_shape").ints
                k_h = k[0]
                k_w = k[1]
                stride_h = get_by_name(n.attribute, "strides").ints[0]
                stride_w = get_by_name(n.attribute, "strides").ints[1]
                group = get_by_name(n.attribute, "group").i
                weight_name = n.input[1]
                W_conv = model.get_initializer(weight_name)
                ifm_ch = model.get_tensor_shape(n.input[0])[1]  # assume NCHW
                ofm_ch = model.get_tensor_shape(n.output[0])[1]  # assume NCHW
                ifm_dim_h = model.get_tensor_shape(n.input[0])[2]  # assume NCHW
                ifm_dim_w = model.get_tensor_shape(n.input[0])[3]
                ofm_dim_h = model.get_tensor_shape(n.output[0])[2]  # assume NCHW
                ofm_dim_w = model.get_tensor_shape(n.output[0])[3]
                dilation_attr = get_by_name(n.attribute, "dilations")
                if dilation_attr is not None:
                    dilation = dilation_attr.ints
                else:
                    dilation = [1, 1]  # default value
                # handle both auto_pad and explicit padding
                auto_pad = get_by_name(n.attribute, "auto_pad")
                if auto_pad is not None:
                    # find equivalent specified padding
                    auto_pad = auto_pad.s.decode("utf-8")
                    if auto_pad == "NOTSET":
                        # use specified padding
                        pad = get_by_name(n.attribute, "pads").ints
                    else:
                        pad = _auto_pad_to_explicit_padding(
                            auto_pad,
                            ifm_dim_h,
                            ifm_dim_w,
                            k_h,
                            k_w,
                            stride_h,
                            stride_w,
                            len(model.get_tensor_shape(n.input[0])) - 2,
                        )
                else:
                    # use specified padding
                    pad = get_by_name(n.attribute, "pads").ints

                # If len(pad) == 2, assume no padding for other dimension
                if len(pad) == 2:  # only one dimension should be padded
                    assert ifm_dim_h == 1 or ifm_dim_w == 1, "Padding is assumed to be 1D, image is 2D"

                # if depthwise conv create sparse matrix and variable "dw"
                # to store as attribute in Im2Col that indicates that the created
                # Im2Col node belongs to a depthwise convolution
                dw = False
                if group == ifm_ch and ofm_ch == ifm_ch:
                    W_sparse = np.zeros((ofm_ch, ifm_ch, k_h, k_w))  # (OFM, IFM, k_H, k_W)
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
                # first convert to [OFM][k][k][IFM] (to remain compatible with
                # finn-hlslib and how it does im2col/sliding window)
                W_matmul = W_conv.transpose(0, 2, 3, 1)  # W_conv = [OFM, IFM, k_H, k_W]
                # reshape into [OFM][k*k*IFM] matrix
                W_matmul = W_matmul.reshape(ofm_ch, ifm_ch * k_h * k_w)
                # transpose to get ONNX-compatible [k*k*IFM][OFM] matrix
                W_matmul = W_matmul.T
                model.set_initializer(weight_name, W_matmul)

                # create new intermediate values
                inp_trans_out = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    (1, ifm_dim_h, ifm_dim_w, ifm_ch),  # NHWC
                )
                graph.value_info.append(inp_trans_out)
                inp_trans_out = inp_trans_out.name
                model.set_tensor_datatype(inp_trans_out, idt)

                need_im2col = True
                if all(p == 0 for p in pad):
                    padding = 0

                # k_h=k_w==1: pointwise convolution, thus no im2col needed
                if k_h == 1 and k_w == 1 and padding == 0 and stride_h == 1 and stride_w == 1:
                    need_im2col = False

                if need_im2col:
                    im2col_out = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        (1, ofm_dim_h, ofm_dim_w, ifm_ch * k_h * k_w),
                    )
                    graph.value_info.append(im2col_out)
                    im2col_out = im2col_out.name
                    model.set_tensor_datatype(im2col_out, idt)

                matmul_out = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    (1, ofm_dim_h, ofm_dim_w, ofm_ch),
                )
                graph.value_info.append(matmul_out)
                matmul_out = matmul_out.name
                model.set_tensor_datatype(matmul_out, odt)

                # create new nodes
                # NCHW -> NHWC
                inp_trans_node = helper.make_node("Transpose", [cnv_input], [inp_trans_out], perm=[0, 2, 3, 1])
                # lower input tensor
                matmul_input = inp_trans_out
                if need_im2col:
                    matmul_input = im2col_out
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

                # do matmul
                matmul_node = helper.make_node("MatMul", [matmul_input, weight_name], [matmul_out])
                # NHWC -> NCHW
                out_trans_node = helper.make_node("Transpose", [matmul_out], [cnv_output], perm=[0, 3, 1, 2])
                # insert nodes where the conv is to preserve topological ordering
                graph.node.insert(node_ind, inp_trans_node)
                if need_im2col:
                    graph.node.insert(node_ind + 1, im2col_node)
                    graph.node.insert(node_ind + 2, matmul_node)
                    graph.node.insert(node_ind + 3, out_trans_node)
                else:
                    graph.node.insert(node_ind + 1, matmul_node)
                    graph.node.insert(node_ind + 2, out_trans_node)
                # remove old nodes
                graph.node.remove(n)

        return (model, graph_modified)
