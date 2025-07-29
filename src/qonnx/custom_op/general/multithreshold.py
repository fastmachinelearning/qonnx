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
import onnx.helper as helper

from qonnx.core.datatype import DataType
from qonnx.custom_op.base import CustomOp


def multithreshold(v, thresholds, out_scale=None, out_bias=None):
    """Given a set of threshold values t={t_0, t_1 ... t_n} the successive
    thresholding maps any real number x to an integer in the interval [0, n],
    where the returned integer is the number of thresholds x is greater than
    or equal to.

    The output tensor will be scaled by out_scale and biased by out_bias."""
    # the inputs are expected to be in the shape (N,C,_) where:
    #  C is the number of channels
    #  _ represents any (including zero) number of spatial dims
    # the thresholds are expected to be in the shape (C, B) where
    # C : Number of channels (must be the same value as C in input tensor
    #     or 1 if all channels use the same threshold value)
    # B : Desired activation steps => i.e. for 4-bit activation,
    #     B=7 (2^(n)-1 and n=4), but can also be fewer
    # the output tensor will be scaled by out_scale and biased by out_bias
    # assert threshold shape - threshold channels must be either equal
    # to input channels, or be a single global scalar
    is_global_threshold = thresholds.shape[0] == 1
    assert (
        v.shape[1] == thresholds.shape[0]
    ) or is_global_threshold, """"Threshold
    shape incorrect"""
    # starting assumption: input tensor is in NC_ layout
    # (where _ can be any number of spatial dims)
    # get the input tensor into right shape to use numpy broadcasting
    # move the channels axis to the last position
    # NC_ -> N_C
    vm = np.moveaxis(v, source=1, destination=-1)
    # add a dummy dimension at the end of the input tensor
    # (for broadcasting against the thresholds)
    # N_C -> N_C1
    vm = np.expand_dims(vm, axis=-1)
    # now perform the comparison against thresholds
    # (N_C1 >= CT) -> N_CT
    cmp = vm >= thresholds
    # replace last axis by count of nonzero values (True)
    # N_CT -> N_C
    ret = np.count_nonzero(cmp, axis=-1)
    # finally, move the channels axis back to index 1
    ret = np.moveaxis(ret, source=-1, destination=1)
    assert ret.shape == v.shape, "Shape changed during thresholding!"

    if out_scale is None:
        out_scale = 1.0
    if out_bias is None:
        out_bias = 0.0
    return out_scale * ret + out_bias


class MultiThreshold(CustomOp):
    """Class that corresponds to a multithresholding node."""

    def get_nodeattr_types(self):
        return {
            "out_dtype": ("s", True, ""),
            "out_scale": ("f", False, 1.0),
            "out_bias": ("f", False, 0.0),
            "data_layout": ("s", False, "NCHW", {"NCHW", "NHWC"}),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node
        return helper.make_node("Relu", [node.input[0]], [node.output[0]])

    def infer_node_datatype(self, model):
        node = self.onnx_node
        odt = self.get_nodeattr("out_dtype")
        is_float = False
        scale = self.get_nodeattr("out_scale")
        bias = self.get_nodeattr("out_bias")
        if scale is not None and (int(scale) != scale):
            is_float = True
        if bias is not None and (int(bias) != bias):
            is_float = True
        if is_float:
            model.set_tensor_datatype(node.output[0], DataType["FLOAT32"])
        else:
            model.set_tensor_datatype(node.output[0], DataType[odt])

    def execute_node(self, context, graph):
        node = self.onnx_node
        # save inputs
        v = context[node.input[0]]
        thresholds = context[node.input[1]]
        # retrieve attributes if output scaling is used
        out_scale = self.get_nodeattr("out_scale")
        out_bias = self.get_nodeattr("out_bias")
        # transpose input if NHWC data layout is chosen
        data_layout = self.get_nodeattr("data_layout")
        if data_layout == "NHWC":
            if v.ndim == 4:
                # NHWC -> NCHW
                v = np.transpose(v, (0, 3, 1, 2))
            elif v.ndim == 2:
                # no HW dimension means NHWC and NCHW layouts are equivalent
                pass
            else:
                raise Exception("Unknown data_layout and input ndim" " combination for MultiThreshold.")

        # Remember whether the shape has been modified to handle 1d or 3d data
        # layouts
        orig_shape = None
        # If the input tensor has dimensions not covered by the NC or NCWH data
        # layouts, the shape needs to be adapted such that it can be handled by
        # multithreshold.
        # TODO: Seems like a rather sketchy solution to support arbitrary data
        #  layouts. This does not even validate the assumption of channel last
        #  layout.
        if v.ndim not in {2, 4}:
            # Remember the original shape to be restored later
            orig_shape = v.shape
            # Assume last dimension to be the channel dimension C and reshape
            # into NC layout which is supported by multithreshold
            v = v.reshape((-1, v.shape[-1]))

        # calculate output
        output = multithreshold(v, thresholds, out_scale, out_bias)
        # setting context according to output
        if data_layout == "NHWC":
            if output.ndim == 4:
                # NCHW -> NHWC
                output = np.transpose(output, (0, 2, 3, 1))
            elif output.ndim == 2:
                # no HW dimension means NHWC and NCHW layouts are equivalent
                pass
            else:
                raise Exception("Unknown data_layout and output ndim" " combination for MultiThreshold.")

        # If the shape has been modified to support arbitrary layouts, restore
        # the original shape
        # TODO: Part of the rather sketchy solution above.
        if orig_shape is not None:
            output = output.reshape(orig_shape)

        context[node.output[0]] = output

    def verify_node(self):
        info_messages = []

        # verify that all necessary attributes exist
        try:
            self.get_nodeattr("out_dtype")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append(
                """The necessary attributes do not exist.
                MultiThreshold needs the following attributes:
                out_scale, out_bias, out_dtype"""
            )

        # verify the number of inputs
        if len(self.onnx_node.input) == 2:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append(
                """MultiThreshold needs 2 inputs
                    (data input and threshold values)"""
            )

        return info_messages
