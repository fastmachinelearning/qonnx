# Copyright (c) 2023 Advanced Micro Devices, Inc.
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
# * Neither the name of qonnx nor the names of its
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

import qonnx.util.basic as util
from qonnx.core.datatype import DataType
from qonnx.custom_op.base import CustomOp

# adapted from A. Karpathy's CS231 im2col code
# utilities to generate a patch matrix from a multichannel image
# of shape (batches, channels, height, width)
# note: the spatial dimensions can be set to 1 to indicate
# a dummy dimension (e.g. 1D convs represented as 2D)


def compute_conv_output_dim(ifm_dim, k, stride, total_pad=0, dilation=1):
    """Returns spatial output dimension size for convolution with given params.
    total_pad gives the total amount of padding along the entire axis
    (both sides included).
    """

    out_dim = int(((ifm_dim + total_pad - dilation * (k - 1) - 1) / stride) + 1)
    return out_dim


def get_im2col_indices_nchw(
    x_shape,
    field_height,
    field_width,
    padding=0,
    stride_h=1,
    stride_w=1,
    dilation_h=1,
    dilation_w=1,
):
    """Returns im2col indices."""
    # First figure out what the size of the output should be
    n, c, h, w = x_shape
    pad_h = padding[0] + padding[2]
    pad_w = padding[1] + padding[3]
    out_height = compute_conv_output_dim(h, field_height, stride_h, pad_h, dilation_h)
    out_width = compute_conv_output_dim(w, field_width, stride_w, pad_w, dilation_w)

    i0 = dilation_h * np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, c)
    i1 = stride_h * np.repeat(np.arange(out_height), out_width)
    j0 = dilation_w * np.tile(np.arange(field_width), field_height * c)
    j1 = stride_w * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(c), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices_nchw(
    x,
    ifm_h,
    ifm_w,
    field_height,
    field_width,
    padding=[0, 0, 0, 0],
    stride_h=1,
    stride_w=1,
    pad_val=0,
    dilation_h=1,
    dilation_w=1,
):
    """Performs im2col on image (2D tensor, possibly with 1-length dummy dimensions) x with
    given field height and width, as well as values for padding and stride size.
    Returns result of im2col."""
    # Zero-pad the input
    p = padding

    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (p[0], p[2]), (p[1], p[3])),
        mode="constant",
        constant_values=pad_val,
    )

    k, i, j = get_im2col_indices_nchw(
        x.shape,
        field_height,
        field_width,
        padding,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
    )

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


# ONNX i/o tensor shape assumptions for Im2Col:
# input 0 is the input vector, shape (1, ih, iw, ifm)
# output 0 is the output vector, shape (1, oh, ow, kh*kw*ifm)
# where:
# * ih, iw are the height and width of the input image
# * oh, ow are the height and width of the output (lowered) image
# * ifm is the number of input channels
# * kh, kw is the convolutional kernel size

# note: for the innermost (dot product) dimension of k*k*ifm, we
# assume an internal ordering (k, k, ifm)

# note2: it's possible to set one of ih, iw to be 1 to indicate a
# dummy dimension, e.g. for representing 1D convs as 2D. the corresponding
# oh/ow and kh/kw will also be 1 in this case


class Im2Col(CustomOp):
    def get_nodeattr_types(self):
        return {
            # stride and shape of convolution kernel
            "stride": ("ints", True, []),
            "kernel_size": ("ints", True, []),
            # input tensor shape
            "input_shape": ("s", True, ""),
            # amount of padding to be inserted before/after each non-dummy spatial dim
            # i.e. [H_begin, W_begin, H_end, W_end]
            "pad_amount": ("ints", False, [0, 0, 0, 0]),  # default: no padding
            # value of padding pixels to be inserted
            "pad_value": ("i", False, 0),
            # depthwise: if 1, infer ConvolutionInputGenerator with depthwise == 1
            "depthwise": ("i", False, 0, {0, 1}),
            # dilation factor applied to the conv kernel
            "dilations": ("ints", False, [1, 1]),
        }

    def make_shape_compatible_op(self, model):
        k_h, k_w = self.get_nodeattr("kernel_size")  # Assumption: Height x Width
        stride_h, stride_w = self.get_nodeattr("stride")
        ishape = self.get_nodeattr("input_shape")
        dilation_h, dilation_w = self.get_nodeattr("dilations")
        pad = self.get_nodeattr("pad_amount")  # padding: [H_begin, W_begin, H_end, W_end]
        pad_h = pad[0] + pad[2]
        pad_w = pad[1] + pad[3]

        # convert string into list of integers
        ishape = ishape.strip("(")
        ishape = ishape.strip(")")
        ishape = ishape.split(",")
        for i in range(0, len(ishape)):
            ishape[i] = int(ishape[i])

        # extract all necessary information and determine output dimensions
        ifm_ch = ishape[-1]
        assert len(ishape) == 4, "Unexpected input shape for Im2Col"
        # NHWC (QONNX always converts to NHWC during conv lowering)
        ifm_dim_h = ishape[1]
        ifm_dim_w = ishape[2]

        ofm_dim_h = compute_conv_output_dim(ifm_dim_h, k_h, stride_h, pad_h, dilation_h)
        ofm_dim_w = compute_conv_output_dim(ifm_dim_w, k_w, stride_w, pad_w, dilation_w)

        return super().make_const_shape_op([1, ofm_dim_h, ofm_dim_w, k_h * k_w * ifm_ch])

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # data type stays the same
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def execute_node(self, context, graph):
        node = self.onnx_node
        k_h, k_w = self.get_nodeattr("kernel_size")  # Assumption: Height x Width
        stride_h, stride_w = self.get_nodeattr("stride")
        pad = self.get_nodeattr("pad_amount")
        pad_h = pad[0] + pad[2]
        pad_w = pad[1] + pad[3]
        pad_val = self.get_nodeattr("pad_value")
        dilation_h, dilation_w = self.get_nodeattr("dilations")

        iname = node.input[0]
        x = context[iname]
        qnt_annotations = graph.quantization_annotation
        ret = util.get_by_name(qnt_annotations, iname, "tensor_name")
        ret = util.get_by_name(ret.quant_parameter_tensor_names, "finn_datatype", "key")
        idt = DataType[ret.value]
        if pad != [0, 0, 0, 0]:
            assert idt.allowed(pad_val), "Im2Col dtype must allow pad_val"
        # check that input is NHWC
        assert x.ndim == 4, "Unexpected number of input dims for Im2Col"
        n, h, w, c = x.shape

        # check that kernel tensor also respects any existing dummy dimensions
        if h == 1:
            kernel_1d = k_h == 1
            pad_1d = pad_h == 0
            assert (
                kernel_1d and pad_1d
            ), "Unexpected kernel shape and padding for input image\
             of dimensions (N, 1, W, C)"
        if w == 1:
            kernel_1d = k_w == 1
            pad_1d = pad_w == 0
            assert (
                kernel_1d and pad_1d
            ), "Unexpected kernel shape and padding for input image\
             of dimensions (N, H, 1, C)"

        out_dim_h = compute_conv_output_dim(h, k_h, stride_h, pad_h, dilation_h)
        out_dim_w = compute_conv_output_dim(w, k_w, stride_w, pad_w, dilation_w)
        # internally convert input to NCHW
        x = x.transpose(0, 3, 1, 2)
        # call NCHW im2col implementation
        ret = im2col_indices_nchw(
            x,
            h,
            w,
            k_h,
            k_w,
            pad,
            stride_h,
            stride_w,
            pad_val=pad_val,
            dilation_h=dilation_h,
            dilation_w=dilation_w,
        )
        # result shape is (k_H*k_W*N, out_dim_H*out_dim_W), convert to NCHW
        ret = ret.reshape(n, c, k_h, k_w, out_dim_h, out_dim_w)
        # (N=0,C=1,kh=2,kw=3,H=4,W=5) -> (N=0,H=4,W=5,kh=2,kw=3,C=1)
        ret = ret.transpose(0, 4, 5, 2, 3, 1)
        ret = ret.reshape(n, out_dim_h, out_dim_w, k_h * k_w * c)

        # ret = ret.reshape(N, k * k * C, out_dim, out_dim)
        # convert output back to NHWC
        # ret = ret.transpose(0, 2, 3, 1)
        context[node.output[0]] = ret

    def verify_node(self):
        node = self.onnx_node

        info_messages = []

        # verify number of attributes
        num_of_attr = 3
        if len(node.attribute) == num_of_attr:
            info_messages.append("The number of attributes is correct")
        else:
            info_messages.append(
                """The number of attributes is incorrect,
            {} should have {} attributes""".format(
                    node.op_type, num_of_attr
                )
            )
        # verify that all necessary attributes exist
        try:
            self.get_nodeattr("stride")
            self.get_nodeattr("kernel_size")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append(
                """The necessary attributes do not exist.
                Im2Col needs the following attributes:
                stride, kernel_size"""
            )

        # verify the number of inputs
        if len(node.input) == 1:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append("{} needs 1 data input".format(node.op_type))

        return info_messages
