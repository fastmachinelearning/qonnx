import numpy as np
from onnx import TensorProto, helper

from qonnx.custom_op.channels_last.base_wrapped_op import ChannelsLastWrappedOp

class Concat(ChannelsLastWrappedOp):
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
            # axis attribute of Concat layer, default 1
            "axis": ("i", True, 1)
        }

    def make_shape_compatible_op(self, model):
        """Returns a standard ONNX op which is compatible with this CustomOp
        for performing shape inference."""

        node = self.onnx_node
        iname0 = node.input[0]
        iname1 = node.input[1]
        ishape0 = model.get_tensor_shape(iname0)
        ishape1 = model.get_tensor_shape(iname1)
        # axis = self.get_nodeattr("axis")
        # not sure about what's the shape of inputs, don't know how to check it
        # check that ishape0[1] == ishape1[1] and ishape0[2] == ishape1[2]
        assert ishape0[1] == ishape1[1], "Input shape [1] has to be the same between the 2 input nodes of concat"
        assert ishape0[2] == ishape1[2], "Input shape [2] has to be the same between the 2 input nodes of concat"
        
        # implement tensor with correct shape
        output_shape = [1, ishape0[1], ishape0[2], ishape0[3] + ishape1[3]]
        
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
        num_of_attr_min = 1
        num_of_attr_max = 1
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
            self.get_nodeattr("axis")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append(
                """The necessary attributes do not exist.
                Concat needs the following attributes:
                axis"""
            )
            verification_successful = False

        # verify that attributes have the correct datatype.
        try:
            assert isinstance(self.get_nodeattr("axis"), int)
            info_messages.append("All attributes are of the correct type")
        except Exception:
            info_messages.append("One or more attributes are of the wrong datatype")
            verification_successful = False

        # verify the number of inputs
        if len(node.input) >= 2:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append("{} needs 2 data input".format(node.op_type))
            verification_successful = False

        if not verification_successful:
            raise RuntimeError(
                f"Verification of node {node.name} failed, please check the " f"attached info messages: {info_messages}"
            )

        return info_messages
