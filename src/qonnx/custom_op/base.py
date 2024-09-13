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

import onnx.helper as helper
import onnx.numpy_helper as np_helper
from abc import ABC, abstractmethod

from qonnx.util.basic import get_by_name, get_preferred_onnx_opset


class CustomOp(ABC):
    """CustomOp class all custom op nodes are based on. Contains different functions
    every custom node should have. Some as abstract methods, these have to be
    filled when writing a new custom op node."""

    def __init__(self, onnx_node, onnx_opset_version=get_preferred_onnx_opset()):
        super().__init__()
        self.onnx_node = onnx_node
        self.onnx_opset_version = onnx_opset_version

    def get_nodeattr_def(self, name):
        """Return 4-tuple (dtype, required, default_val, allowed_values) for attribute
        with name. allowed_values will be None if not specified."""
        allowed_values = None
        attrdef = self.get_nodeattr_types()[name]
        if len(attrdef) == 3:
            (dtype, req, def_val) = attrdef
        elif len(attrdef) == 4:
            (dtype, req, def_val, allowed_values) = attrdef
        else:
            raise Exception("Unexpected length %d n-tuple from get_nodeattr_types" % len(attrdef))
        return (dtype, req, def_val, allowed_values)

    def get_nodeattr_allowed_values(self, name):
        "Return set of allowed values for given attribute, None if not specified."
        return self.get_nodeattr_def(name)[3]

    def get_nodeattr(self, name):
        """Get a node attribute by name. Data is stored inside the ONNX node's
        AttributeProto container. Attribute must be part of get_nodeattr_types.
        Default value is returned if attribute is not set."""
        try:
            (dtype, req, def_val, allowed_values) = self.get_nodeattr_def(name)
            attr = get_by_name(self.onnx_node.attribute, name)
            if attr is not None:
                # dtype indicates which ONNX Attribute member to use
                # (such as i, f, s...)
                ret = attr.__getattribute__(dtype)
                if dtype == "s":
                    # decode string attributes
                    ret = ret.decode("utf-8")
                elif dtype == "strings":
                    ret = [x.decode("utf-8") for x in ret]
                elif dtype == "t":
                    # use numpy helper to convert TensorProto -> np array
                    ret = np_helper.to_array(ret)
                elif dtype == "ints":
                    # convert from RepeatedScalarContainer to list
                    # gives e.g. JSON serializability
                    ret = [x for x in ret]
                if allowed_values is not None:
                    assert ret in allowed_values, "%s = %s not in %s" % (
                        str(name),
                        str(ret),
                        str(allowed_values),
                    )
                return ret
            else:
                if req:
                    raise Exception(
                        """Required attribute %s unspecified in
                    a %s node"""
                        % (name, self.onnx_node.op_type)
                    )
                else:
                    # not set, return default value
                    return def_val
        except KeyError:
            raise AttributeError("Op has no such attribute: " + name)

    def set_nodeattr(self, name, value):
        """Set a node attribute by name. Data is stored inside the ONNX node's
        AttributeProto container. Attribute must be part of get_nodeattr_types."""
        try:
            (dtype, req, def_val, allowed_values) = self.get_nodeattr_def(name)
            if allowed_values is not None:
                assert value in allowed_values, "%s = %s not in %s" % (
                    str(name),
                    str(value),
                    str(allowed_values),
                )
            attr = get_by_name(self.onnx_node.attribute, name)
            if dtype == "t":
                # convert numpy array to TensorProto
                value = np_helper.from_array(value)
            if attr is not None:
                # dtype indicates which ONNX Attribute member to use
                # (such as i, f, s...)
                if dtype == "s":
                    # encode string attributes
                    value = value.encode("utf-8")
                    attr.__setattr__(dtype, value)
                elif dtype == "strings":
                    attr.strings[:] = [x.encode("utf-8") for x in value]
                elif dtype == "floats":  # list of floats
                    attr.floats[:] = value
                elif dtype == "ints":  # list of integers
                    attr.ints[:] = value
                elif dtype == "t":  # single tensor
                    attr.t.CopyFrom(value)
                elif dtype in ["tensors", "graphs", "sparse_tensors"]:
                    # untested / unsupported attribute types
                    # add testcases & appropriate getters before enabling
                    raise Exception("Attribute type %s not yet supported" % dtype)
                else:
                    # attempt to set attr.dtype = value directly
                    attr.__setattr__(dtype, value)
            else:
                # not set, create and insert AttributeProto
                attr_proto = helper.make_attribute(name, value)
                self.onnx_node.attribute.append(attr_proto)
        except KeyError:
            raise AttributeError("Op has no such attribute: " + name)

    def make_const_shape_op(self, shape):
        """Return an ONNX node that generates the desired output shape for
        shape inference."""
        return helper.make_node(
            "RandomNormal",
            inputs=[],
            outputs=[self.onnx_node.output[0]],
            mean=0.0,
            scale=1.0,
            dtype=1,
            shape=list(shape),
        )

    @abstractmethod
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
        pass

    @abstractmethod
    def make_shape_compatible_op(self, model):
        """Returns a standard ONNX op which is compatible with this CustomOp
        for performing shape inference."""
        pass

    @abstractmethod
    def infer_node_datatype(self, model):
        """Set the DataType annotations corresponding to the outputs of this
        node."""
        pass

    @abstractmethod
    def execute_node(self, context, graph):
        """Execute this CustomOp instance, given the execution context and
        ONNX graph."""
        pass

    @abstractmethod
    def verify_node(self):
        """Verifies that all attributes the node needs are there and
        that particular attributes are set correctly. Also checks if
        the number of inputs is equal to the expected number."""
        pass
