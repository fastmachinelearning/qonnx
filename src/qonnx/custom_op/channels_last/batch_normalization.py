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


class BatchNormalization(ChannelsLastWrappedOp):
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
            # The epsilon value to use to avoid division by zero.
            "epsilon": ("f", True, 1e-05),
            # Factor used in computing the running mean and variance.
            # e.g., running_mean = running_mean * momentum + mean * (1 - momentum).
            "momentum": ("f", True, 0.9),
        }
        pass

    def make_shape_compatible_op(self, model):
        """Returns a standard ONNX op which is compatible with this CustomOp
        for performing shape inference."""
        # For BatchNorm the output shape should be the same as the input shape.
        # Get the output shape from the input
        out_shape = model.get_tensor_shape(self.onnx_node.input[0])

        # implement tensor with correct shape
        values = np.random.randn(*out_shape).astype(np.float32)
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
        num_of_attr = 2
        if len(node.attribute) >= num_of_attr:
            info_messages.append("The number of attributes is correct")
        else:
            info_messages.append(
                """The number of attributes is incorrect,
            {} should have {} attributes""".format(
                    node.op_type, num_of_attr
                )
            )
            verification_successful = False

        # verify that all necessary attributes exist
        try:
            self.get_nodeattr("epsilon")
            self.get_nodeattr("momentum")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append(
                """The necessary attributes do not exist.
                BatchNormalization needs the following attributes:
                epsilon, momentum"""
            )
            verification_successful = False

        # verify that attributes have the correct datatype.
        try:
            assert isinstance(self.get_nodeattr("epsilon"), float)
            assert isinstance(self.get_nodeattr("momentum"), float)
            info_messages.append("All attributes are of the correct type")
        except Exception:
            info_messages.append("One or more attributes are of the wrong datatype")
            verification_successful = False

        # verify the number of inputs
        if len(node.input) == 5:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append("{} needs 5 data inputs".format(node.op_type))
            verification_successful = False

        if not verification_successful:
            raise RuntimeError(
                f"Verification of node {node.name} failed, please check " f"the attached info messages: {info_messages}"
            )

        return info_messages
