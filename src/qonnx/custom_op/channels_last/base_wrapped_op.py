import numpy as np
import onnxruntime as rt
from copy import deepcopy
from onnx import TensorProto, helper

from finn.custom_op.base import CustomOp


class ChannelsLastWrappedOp(CustomOp):
    # ToDo: _channelsLast_node_types should be loaded / inferred from this file or the registry.
    # Standard ONNX nodes which require a ChannelsLast data format to function properly
    _channelsLast_node_types = ["Conv", "MaxPool", "BatchNormalization"]
    # Required for ChannelsLast transformations and ops
    # Transpose parameters to convert to channels last for 3D and 4D tensors
    _to_chan_last_args = {
        3: (0, 2, 1),
        4: (0, 2, 3, 1),
    }
    # Similarly for converting back to channels first.
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
