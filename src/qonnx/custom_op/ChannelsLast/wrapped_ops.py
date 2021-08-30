import numpy as np
import onnxruntime as rt
from copy import deepcopy
from google.protobuf.pyext._message import RepeatedScalarContainer
from onnx import TensorProto, helper

from finn.custom_op.base import CustomOp
from finn.custom_op.general.im2col import compute_conv_output_dim
from finn.custom_op.general.maxpoolnhwc import compute_pool_output_dim


class ChannelsLastWrappedOp(CustomOp):
    # ToDo: _channelsLast_node_types should be loaded / inferred from this file or the registry.
    _channelsLast_node_types = ["Conv", "MaxPool", "BatchNormalization"]
    _to_chan_last_args = {
        3: (0, 2, 1),
        4: (0, 2, 3, 1),
    }
    _to_chan_first_args = {
        3: (0, 2, 1),
        4: (0, 3, 1, 2),
    }

    def infer_node_datatype(self, model):
        # data type stays the same for all supported nodes
        node = self.onnx_node
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def verify_node(self):
        # Check general compatibility
        node = self.onnx_node
        assert (
            node.op_type in self._channelsLast_node_types
        ), f"{node.op_type} is not supported by the ChannelsLast wrapper op."
        assert len(node.input) > 0, "The ChannelsLast wrapper op only supports nodes with inputs."
        assert len(node.output) == 1, "The ChannelsLast wrapper op only supports nodes with exactly one output."

        result = [
            "ONNX OP-type is supported by ChannelsLast wrapper for node execution.",
            "Number of inputs and outputs is valid for node execution.",
        ]

        return result

    def execute_node(self, context, graph):
        node = self.onnx_node

        # Create an intermediate node and remove the domain
        # This enables us to use onnxrutime to execute this node.
        intermediate_node = deepcopy(node)
        intermediate_node.domain = ""

        # Create an intermediate context
        # intermediate_context = {}
        input_dict = {}
        input_tensor_list = []
        output_tensor_list = []

        # Create transposed (channel first) arrays
        # and onnx tensors for the inputs and outputs.
        # And store them in the internal context.
        ndim = len(context[intermediate_node.input[0]].shape)
        for i, input in enumerate(intermediate_node.input):
            channelsFirst_array = context[input]
            # Generally we only transpose the first input
            transpose_input = i < 1
            # Conv is an exception, it also requires the second input to be transposed.
            transpose_input |= intermediate_node.op_type == "Conv" and i < 2
            if transpose_input:
                channelsFirst_array = channelsFirst_array.transpose(self._to_chan_first_args[ndim])
            assert channelsFirst_array.dtype == np.float32, "Requires float tensor, currently."
            tensor = helper.make_tensor_value_info(input, TensorProto.FLOAT, channelsFirst_array.shape)
            input_dict[input] = channelsFirst_array
            input_tensor_list.append(tensor)

        output = intermediate_node.output[0]
        channelsFirst_array = context[output]
        channelsFirst_array = channelsFirst_array.transpose(self._to_chan_first_args[ndim])
        assert channelsFirst_array.dtype == np.float32, "Requires float tensor, currently."
        tensor = helper.make_tensor_value_info(output, TensorProto.FLOAT, channelsFirst_array.shape)
        output_tensor_list.append(tensor)

        # Execute the intermediate node with onnxruntime,
        # using the transposed inputs / outputs
        intermediate_graph = helper.make_graph([intermediate_node], "test_model", input_tensor_list, output_tensor_list)
        intermediate_model = helper.make_model(intermediate_graph)
        sess = rt.InferenceSession(intermediate_model.SerializeToString())
        output_list = sess.run(None, input_dict)
        output_onnx = output_list[0]

        # Transpose the output back to channel last and save it in the external context.
        output_onnx = output_onnx.transpose(self._to_chan_last_args[ndim])
        context[node.output[0]] = output_onnx


class Conv(ChannelsLastWrappedOp):
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
            # stride and shape of convolution kernel
            "kernel_shape": ("ints", True, []),
            "strides": ("ints", True, []),
            # dilation factor applied to the conv kernel
            "dilations": ("ints", True, []),
            # amount of padding to be inserted before/after each non-dummy spatial dim
            # i.e. [H_begin, W_begin, H_end, W_end]
            "pads": ("ints", True, [0, 0, 0, 0]),  # default: no padding
            "group": ("i", True, 1),
        }

    def make_shape_compatible_op(self, model):
        """Returns a standard ONNX op which is compatible with this CustomOp
        for performing shape inference."""
        # Modified version of: Im2Col.make_shape_compatible_op
        # From file: src/finn/custom_op/general/im2col.py
        ishape = model.get_tensor_shape(self.onnx_node.input[0])
        ndim = len(ishape)
        assert ndim == 3 or ndim == 4, "ChannelsLast Conv currently only supports 3D and 4D input tensors."

        kernel_shape = self.get_nodeattr("kernel_shape")
        strides = self.get_nodeattr("strides")
        dilations = self.get_nodeattr("dilations")
        # Get the input shape from the previous tensor
        ishape = model.get_tensor_shape(self.onnx_node.input[0])
        pads = self.get_nodeattr("pads")

        ofm_dims = []
        for i in range(ndim - 2):
            if ndim == 3:
                # padding: [begin, end]
                padding = pads[0] + pads[1]
            elif ndim == 4:
                # padding: [H_begin, W_begin, H_end, W_end]
                padding = pads[i] + pads[i + 2]
            else:
                raise ValueError(
                    f"Inputs of dimensionality ndim={ndim}, are currently not supported for "
                    f"the channels last Conv operation."
                )
            ofm_d = compute_conv_output_dim(ishape[i + 1], kernel_shape[i], strides[i], padding, dilations[i])
            ofm_dims.append(ofm_d)

        # Get the number of output channels form the shape of the weight tensor.
        out_ch = model.get_tensor_shape(self.onnx_node.input[1])
        out_ch = out_ch[0]

        # implement tensor with correct shape
        output_shape = [
            1,
        ]
        for ofm_d in ofm_dims:
            output_shape.append(ofm_d)
        output_shape.append(out_ch)

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
        num_of_attr = 5
        if len(node.attribute) == num_of_attr:
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
            self.get_nodeattr("dilations")
            self.get_nodeattr("group")
            self.get_nodeattr("kernel_shape")
            self.get_nodeattr("pads")
            self.get_nodeattr("strides")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append(
                """The necessary attributes do not exist.
                Conv needs the following attributes:
                dilations, group, kernel_shape, pads, strides"""
            )
            verification_successful = False

        # verify that attributes have the correct datatype.
        try:
            assert isinstance(self.get_nodeattr("kernel_shape"), RepeatedScalarContainer)
            assert isinstance(self.get_nodeattr("pads"), RepeatedScalarContainer)
            assert isinstance(self.get_nodeattr("strides"), RepeatedScalarContainer)
            assert isinstance(self.get_nodeattr("dilations"), RepeatedScalarContainer)
            assert isinstance(self.get_nodeattr("group"), int)
            info_messages.append("All attributes are of the correct type")
        except Exception:
            info_messages.append("One or more attributes are of the wrong datatype")
            verification_successful = False

        # verify the number of inputs
        if len(node.input) == 2:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append("{} needs 2 data inputs".format(node.op_type))
            verification_successful = False

        if not verification_successful:
            raise RuntimeError(
                f"Verification of node {node.name} failed, please check the " f"attached info messages: {info_messages}"
            )

        return info_messages


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
            assert isinstance(self.get_nodeattr("kernel_shape"), RepeatedScalarContainer)
            assert isinstance(self.get_nodeattr("pads"), RepeatedScalarContainer)
            assert isinstance(self.get_nodeattr("strides"), RepeatedScalarContainer)
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
        if len(node.attribute) == num_of_attr:
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
