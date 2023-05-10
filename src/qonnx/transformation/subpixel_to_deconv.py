# Copyright (c) 2023, Advanced Micro Devices, Inc.
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
# * Neither the name of QONNX nor the names of its
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
from onnx import helper

from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name


def _weight_shuffle(cnv_weights: np.ndarray, block_size: int) -> np.ndarray:
    """Adaptation of the weight shuffle algorithm as proposed in Colbert et al. (2021) - `An
    Energy-Efficient Edge Computing Paradigm for Convolution-Based Image Upsampling`"""
    ofm_ch = cnv_weights.shape[0]
    ifm_ch = cnv_weights.shape[1]
    assert ofm_ch % block_size == 0, "Out channels need to be evenly divisible by block size"
    ofm_ch = ofm_ch // pow(block_size, 2)
    kh_size = cnv_weights.shape[2]
    kw_size = cnv_weights.shape[3]
    assert kh_size == kw_size, "Only square channels supported currently."
    # NOTE - this is different than the convolution kernels, which are OC x IC x KH x KW
    #        rather than IC x OC x KH x KW
    dcnv_weights = np.zeros((ifm_ch, ofm_ch, kh_size * block_size, kw_size * block_size))
    cnv_weights = np.moveaxis(cnv_weights, 0, 1)  # change conv to match deconv data format
    for oc_d in range(ofm_ch):
        for kh_d in range(kh_size * block_size):
            for kw_d in range(kw_size * block_size):
                kh_c = kh_d // block_size
                kw_c = kw_d // block_size
                _a = kh_d % block_size
                _b = kw_d % block_size
                _c = oc_d
                oc_c = (pow(block_size, 2) * _c) + (block_size * _a) + _b
                # shuffling and transposing the kernels
                dcnv_weights[:, oc_d, kh_d, kw_d] = cnv_weights[:, oc_c, kh_size - kh_c - 1, kw_size - kw_c - 1]
    return dcnv_weights


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


class SubPixelToDeconvolution(Transformation):
    """Replaces sub-pixel convolution layers (i.e., same-padded convolution + depth2space)
    with deconvolution layers using the weight shuffle algorithm. Currently does not support
    same-padded convolutions with biases."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Conv":
                cnv_input = n.input[0]
                cnv_output = n.output[0]
                consumers = model.find_consumers(cnv_output)
                if len(consumers) > 1 and any([c.op_type == "DepthToSpace" for c in consumers]):
                    warnings.warn(
                        "Skipping sub-pixel conv that has same-padded conv with multiple consumers. Not yet supported."
                    )
                    continue
                consumer = consumers[0]
                # NOTE - currently supports onnx opset 11+, which introduced the
                # pixel shuffle operator as a depth2space op_type
                if consumer is not None and consumer.op_type == "DepthToSpace":
                    # TODO - converting sub-pixel convolution with bias requires a non-trivial
                    # elementwise addition node since the bias needs to be shuffled also
                    if len(n.input) == 3:
                        warnings.warn("Skipping sub-pixel conv with bias. Not yet supported.")
                        continue
                    group = get_by_name(n.attribute, "group").i
                    if group != 1:
                        warnings.warn("Skipping sub-pixel conv with group > 1. Not yet supported.")
                    weight_name = n.input[1]
                    W_conv = model.get_initializer(weight_name)  # (OC, IC, KH, KW)
                    kshape = get_by_name(n.attribute, "kernel_shape").ints
                    ifm_ch = model.get_tensor_shape(n.input[0])[1]  # assume NCHW
                    ofm_ch = model.get_tensor_shape(n.output[0])[1]  # assume NCHW
                    ifm_dim_h = model.get_tensor_shape(n.input[0])[2]  # assume NCHW
                    ifm_dim_w = model.get_tensor_shape(n.input[0])[3]  # assume NCHW
                    ofm_dim_h = model.get_tensor_shape(n.output[0])[2]  # assume NCHW
                    ofm_dim_w = model.get_tensor_shape(n.output[0])[3]
                    if (ifm_dim_h != ofm_dim_h) or (ifm_dim_w != ofm_dim_w):
                        warnings.warn("Skipping sub-pixel conv, only same-padded convs supported.")
                    dilation_attr = get_by_name(n.attribute, "dilations")
                    if dilation_attr is not None:
                        dilation = dilation_attr.ints
                    else:
                        dilation = [1, 1]  # default value
                    if dilation != [1, 1]:
                        warnings.warn("Skipping sub-pixel conv, only supporting dilation=[1,1].")
                    # get depth-to-space (i.e., pixel shuffle) op attributes
                    block_size = get_by_name(consumer.attribute, "blocksize").i
                    if ofm_ch % block_size != 0:
                        warnings.warn(
                            "Skipping sub-pixel conv, the output channels and block size need to be evenly divisible."
                        )
                    W_deconv = _weight_shuffle(W_conv, block_size).astype(np.float32)
                    kh_size_deconv = kshape[0] * block_size
                    kw_size_deconv = kshape[1] * block_size
                    ofm_ch_deconv = ofm_ch // pow(block_size, 2)
                    assert W_deconv.shape == (
                        ifm_ch,
                        ofm_ch_deconv,
                        kh_size_deconv,
                        kw_size_deconv,
                    ), "The resulting deconvolution weight shape is incorrect."
                    stride_h = get_by_name(n.attribute, "strides").ints[0]
                    stride_w = get_by_name(n.attribute, "strides").ints[1]
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
                                kshape[0],
                                kshape[1],
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
                    deconv_inp = cnv_input
                    deconv_out = consumer.output[0]
                    deconv_pad = [p * block_size for p in pad]
                    deconv_node = helper.make_node(
                        "ConvTranspose",
                        [deconv_inp, weight_name],
                        [deconv_out],
                        kernel_shape=[kh_size_deconv, kw_size_deconv],
                        strides=[block_size, block_size],
                        pads=deconv_pad,
                    )
                    model.set_initializer(weight_name, W_deconv)
                    graph.node.insert(node_ind, deconv_node)
                    # remove old nodes
                    graph.node.remove(n)
                    graph.node.remove(consumer)
                    graph_modified = True
        return (model, graph_modified)
