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
from onnx import TensorProto, helper

from qonnx.custom_op.channels_last.base_wrapped_op import ChannelsLastWrappedOp
from qonnx.custom_op.general.maxpoolnhwc import compute_pool_output_dim


class MaxPool(ChannelsLastWrappedOp):
    def get_nodeattr_types(self):
        """Returns a dict of permitted attributes for node, where:
        ret_dict[attribute_name] = (dtype, require, default_value, <allowed_values>)
        - dtype indicates which member of the ONNX AttributeProto
        will be utilized
        - require indicates whether this attribute is required
        - default_val indicates the default value that will be used if the
        attribute is not set
        - <allowed_values> (if specified) indicates that this attribute can only
        be set to one of the values in the set <allowed_values>. If not specified,
        all values permitted by dtype are allowed.
        """
        return {
            # stride and shape of MaxPool kernel
            "kernel_shape": ("ints", True, []),
            "strides": ("ints", True, []),
            # amount of padding to be inserted before/after each non-dummy spatial dim
            # i.e. [H_begin, W_begin, H_end, W_end]
            "pads": ("ints", True, [0, 0, 0, 0]),  # default: no padding
        }

    def make_shape_compatible_op(self, model):
        """Returns a standard ONNX op which is compatible with this CustomOp
        for performing shape inference."""
        # Taken from file src/finn/custom_op/general/maxpoolnhwc.py
        # and function: MaxPoolNHWC.make_shape_compatible_op
        node = self.onnx_node
        iname = node.input[0]
        ishape = model.get_tensor_shape(iname)
        kernel_shape = self.get_nodeattr("kernel_shape")
        pads = self.get_nodeattr("pads")
        strides = self.get_nodeattr("strides")
        ndim = len(ishape)
        assert ndim == 3 or ndim == 4, "ChannelsLast MaxPool currently only supports 3D and 4D input tensors."
        ofm_dims = []
        for i in range(ndim - 2):
            if ndim == 3:
                # padding: [begin, end]
                assert pads[0] == pads[1], "Uneven padding not supported"
            elif ndim == 4:
                # padding: [H_begin, W_begin, H_end, W_end]
                assert pads[0] == pads[2], "Uneven padding not supported"
                assert pads[1] == pads[3], "Uneven padding not supported"
            else:
                raise ValueError(
                    f"Inputs of dimensionality ndim={ndim}, are currently not supported for "
                    f"the channels last MaxPool operation."
                )
            ofm_d = compute_pool_output_dim(ishape[i + 1], kernel_shape[i], strides[i], pads[i])
            ofm_dims.append(ofm_d)

        # Get the number of output channels form the input shape
        out_ch = ishape[-1]

        # implement tensor with correct shape
        output_shape = [
            1,
        ]
        for ofm_d in ofm_dims:
            output_shape.append(ofm_d)
        output_shape.append(out_ch)

        # implement tensor with correct shape
        values = np.random.randn(*output_shape).astype(np.float32)
        return helper.make_node(
            "Constant",
            inputs=[],
            outputs=[self.onnx_node.output[0]],
            value=helper.make_tensor(
                name="const_tensor",
                data_type=TensorProto.FLOAT,
                dims=values.shape,
                vals=values.flatten().astype(float),
            ),
            name=self.onnx_node.name,
        )

    def verify_node(self):
        node = self.onnx_node

        verification_successful = True
        info_messages = []

        wrapper_info = ChannelsLastWrappedOp.verify_node(self)
        info_messages.extend(wrapper_info)

        # verify number of attributes
        num_of_attr_min = 3
        num_of_attr_max = 7
        if (len(node.attribute) >= num_of_attr_min) and len(node.attribute) <= num_of_attr_max:
            info_messages.append("The number of attributes is correct")
        else:
            info_messages.append(
                """The number of attributes is incorrect,
            {} should have between {} and {} attributes""".format(
                    node.op_type, num_of_attr_min, num_of_attr_max
                )
            )
            verification_successful = False

        # verify that all necessary attributes exist
        try:
            self.get_nodeattr("kernel_shape")
            self.get_nodeattr("pads")
            self.get_nodeattr("strides")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append(
                """The necessary attributes do not exist.
                MaxPool needs the following attributes:
                kernel_shape, pads, strides"""
            )
            verification_successful = False

        # verify that attributes have the correct datatype.
        try:
            assert isinstance(self.get_nodeattr("kernel_shape"), list)
            assert isinstance(self.get_nodeattr("pads"), list)
            assert isinstance(self.get_nodeattr("strides"), list)
            info_messages.append("All attributes are of the correct type")
        except Exception:
            info_messages.append("One or more attributes are of the wrong datatype")
            verification_successful = False

        # verify the number of inputs
        if len(node.input) == 1:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append("{} needs 1 data input".format(node.op_type))
            verification_successful = False

        if not verification_successful:
            raise RuntimeError(
                f"Verification of node {node.name} failed, please check the " f"attached info messages: {info_messages}"
            )

        return info_messages
