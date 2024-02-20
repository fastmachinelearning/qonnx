# Copyright (c) 2024, Advanced Micro Devices, Inc.
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

from qonnx.core.datatype import DataType
from qonnx.custom_op.general.quant import quant
from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name


def _weight_convolution(cnv_weights: np.ndarray, scale: int) -> np.ndarray:
    """Adaptation of the weight convolution algorithm as proposed in Colbert et al. (2021) - `An
    Energy-Efficient Edge Computing Paradigm for Convolution-Based Image Upsampling`"""
    ofm_ch = cnv_weights.shape[0]
    ifm_ch = cnv_weights.shape[1]
    kh_size = cnv_weights.shape[2]
    kw_size = cnv_weights.shape[3]
    assert kh_size == kw_size, "Only square channels supported currently."
    # NOTE - this is different than the convolution kernels, which are OC x IC x KH x KW
    #        rather than IC x OC x KH x KW
    dcnv_weights = np.zeros((ifm_ch, ofm_ch, kh_size + scale - 1, kw_size + scale - 1))
    for oc in range(ofm_ch):
        for ic in range(ifm_ch):
            for i in range(scale):
                for j in range(scale):
                    dcnv_weights[ic, oc, i : i + kh_size, j : j + kw_size] += np.rot90(cnv_weights[oc, ic], 2, [0, 1])
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


class ResizeConvolutionToDeconvolution(Transformation):
    """Replaces resize convolution layers (e.g., nearest neighbor upsample + same-padded convolution)
    with deconvolution layers using the weight convolution algorithm. Currently does not support
    resize convolutions that use bilinear or bicubic upsampling"""

    def __init__(self, maintain_bit_width: bool = False):
        super().__init__()
        self.maintain_bit_width = maintain_bit_width

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Resize":
                resize_input = n.input[0]
                resize_output = n.output[0]
                consumers = model.find_consumers(resize_output)

                if len(consumers) == 0:
                    continue

                if len(consumers) > 1 and any([c.op_type == "Conv" for c in consumers]):
                    warnings.warn("Skipping resize conv that has resize with multiple consumers. Not yet supported.")
                    continue

                conv = consumers[0]
                if conv is not None and conv.op_type == "Conv":
                    # TODO: extend support to other resize convolutions
                    resize_mode = get_by_name(n.attribute, "mode").s.decode()
                    if resize_mode != "nearest":
                        warnings.warn(f"Skipping resize conv with resize_mode={resize_mode}. Not yet supported.")
                        continue

                    group = get_by_name(conv.attribute, "group").i
                    if group != 1:
                        warnings.warn("Skipping resize conv with group > 1. Not yet supported.")
                        continue

                    # The weights of the convolution can be generated by another input op if the model is
                    # quantized. Preliminary support for quantization focuses on QONNX ops (i.e., Quant)
                    weight_name = conv.input[1]
                    weight_prod = model.find_producer(weight_name)

                    # If the producer is None, then it is initialized by the Conv node
                    if weight_prod is None:
                        W_conv = model.get_initializer(weight_name)  # (OC, IC, KH, KW)

                    # If the convolution weights are not initialized by the convolution, then we need to
                    # find the node is producing the weights
                    else:
                        if weight_prod.op_type == "Quant":
                            [q_w_name, q_s_name, q_zp_name, q_bw_name] = weight_prod.input
                            W_conv = model.get_initializer(q_w_name)
                            W_scale = model.get_initializer(q_s_name)
                            W_scale = np.moveaxis(W_scale, 0, 1)
                            W_zeropt = model.get_initializer(q_zp_name)
                            W_bitwidth = model.get_initializer(q_bw_name)
                            W_signed = get_by_name(weight_prod.attribute, "signed").i
                            W_narrow = get_by_name(weight_prod.attribute, "narrow").i
                            W_rounding_mode = get_by_name(weight_prod.attribute, "rounding_mode").s.decode()
                        else:
                            warnings.warn(
                                f"Weight producer is {weight_prod.op_type}, not a QONNX Quant node. Not yet supported."
                            )
                            continue

                    kshape = get_by_name(conv.attribute, "kernel_shape").ints
                    ifm_ch = model.get_tensor_shape(conv.input[0])[1]  # assume NCHW
                    ofm_ch = model.get_tensor_shape(conv.output[0])[1]  # assume NCHW
                    ifm_dim_h = model.get_tensor_shape(conv.input[0])[2]  # assume NCHW
                    ifm_dim_w = model.get_tensor_shape(conv.input[0])[3]  # assume NCHW
                    ofm_dim_h = model.get_tensor_shape(conv.output[0])[2]  # assume NCHW
                    ofm_dim_w = model.get_tensor_shape(conv.output[0])[3]
                    if (ifm_dim_h != ofm_dim_h) or (ifm_dim_w != ofm_dim_w):
                        warnings.warn("Skipping resize conv, only same-padded convs supported.")
                        continue
                    dilation_attr = get_by_name(conv.attribute, "dilations")
                    if dilation_attr is not None:
                        dilation = dilation_attr.ints
                    else:
                        dilation = [1, 1]  # default value
                    if dilation != [1, 1]:
                        warnings.warn("Skipping resize conv, only supporting dilation=[1,1].")
                        continue
                    # get resize scaling attribute
                    resize_scales = model.get_initializer(n.input[2])  # assume NCHW
                    if not (resize_scales[0] == resize_scales[1] == 1):
                        warnings.warn("Skipping resize conv, scaling along batch or channel dimension not supported.")
                        continue
                    if resize_scales[2] != resize_scales[3]:
                        warnings.warn("Skipping resize conv, non-square scaling not yet supported.")
                        continue
                    resize_scale = int(resize_scales[2])  # TODO: extend to vector once non-square scaling supported

                    W_deconv = _weight_convolution(W_conv, resize_scale).astype(np.float32)
                    kh_size_deconv = kshape[0] + resize_scale - 1
                    kw_size_deconv = kshape[1] + resize_scale - 1
                    assert W_deconv.shape == (
                        ifm_ch,
                        ofm_ch,
                        kh_size_deconv,
                        kw_size_deconv,
                    ), "The resulting deconvolution weight shape is incorrect."

                    stride_h = get_by_name(conv.attribute, "strides").ints[0]
                    stride_w = get_by_name(conv.attribute, "strides").ints[1]
                    # handle both auto_pad and explicit padding
                    auto_pad = get_by_name(conv.attribute, "auto_pad")
                    if auto_pad is not None:
                        # find equivalent specified padding
                        auto_pad = auto_pad.s.decode("utf-8")
                        if auto_pad == "NOTSET":
                            # use specified padding
                            pad = get_by_name(conv.attribute, "pads").ints
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
                        pad = get_by_name(conv.attribute, "pads").ints

                    # if `maintain_bit_width`, then we use the quant parameters to
                    # re-quantize the weights after the weight convolution
                    if self.maintain_bit_width and (weight_prod is not None):
                        W_deconv_quant = quant(W_deconv, W_scale, W_zeropt, W_bitwidth, W_signed, W_narrow, W_rounding_mode)
                        if not np.allclose(W_deconv, W_deconv_quant):
                            warnings.warn("Clipping error introduced, consider `maintain_bit_width=False`.")

                    # if not `maintain_bit_width`, then we adjust the bit width to
                    # account for the clipping errors.
                    elif weight_prod is not None:
                        W_int = (W_deconv / W_scale) + W_zeropt
                        W_int = W_int.round()  # handling rounding errors
                        if W_int.min() < 0:
                            if np.abs(W_int).min() > W_int.max():
                                tdt = DataType.get_smallest_possible(W_int.min())
                            else:
                                tdt = DataType.get_smallest_possible(-W_int.max() - 1)
                        else:
                            tdt = DataType.get_smallest_possible(W_int.max())
                        assert np.vectorize(tdt.allowed)(W_int).all(), "Error: issue finding data type to support."
                        if W_bitwidth != tdt.bitwidth():
                            W_bitwidth = np.array(tdt.bitwidth(), dtype=np.float32)
                        assert tdt.signed() == W_signed, "Error: should maintain sign of the weights."

                    deconv_inps = [resize_input, weight_name]
                    # Make sure to keep the biases from the convolution
                    if len(conv.input) == 3:
                        bias_name = conv.input[2]
                        B_conv = model.get_initializer(bias_name)  # (OC,)
                        deconv_inps.append(bias_name)  # add to the inputs
                        model.set_initializer(bias_name, B_conv)
                    deconv_outs = conv.output
                    deconv_pad = pad
                    deconv_node = helper.make_node(
                        "ConvTranspose",
                        deconv_inps,
                        deconv_outs,
                        kernel_shape=[kh_size_deconv, kw_size_deconv],
                        strides=[resize_scale, resize_scale],
                        pads=deconv_pad,
                        group=group,
                        dilations=dilation,
                    )
                    W_deconv_init = weight_name
                    if weight_prod is not None:
                        W_deconv_init = q_w_name
                        model.set_initializer(q_s_name, W_scale)
                        model.set_initializer(q_bw_name, W_bitwidth)
                    model.set_initializer(W_deconv_init, W_deconv)
                    model.set_tensor_shape(weight_name, list(W_deconv.shape))
                    graph.node.insert(node_ind, deconv_node)
                    # remove old nodes
                    graph.node.remove(n)
                    graph.node.remove(conv)
                    graph_modified = True

        return (model, graph_modified)