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
import onnx.parser as oprs
import pytest

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.base import CustomOp
from qonnx.custom_op.registry import add_op_to_domain, getCustomOp
from onnx import helper


class AttrTestOp(CustomOp):
    def get_nodeattr_types(self):
        my_attrs = {
            "tensor_attr": ("t", True, np.asarray([])),
            "strings_attr": ("strings", True, [""]),
        }
        return my_attrs

    def make_shape_compatible_op(self, model):
        param_tensor = self.get_nodeattr("tensor_attr")
        return super().make_const_shape_op(param_tensor.shape)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # data type stays the same
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def execute_node(self, context, graph):
        node = self.onnx_node
        param_tensor = self.get_nodeattr("tensor_attr")
        context[node.output[0]] = param_tensor

    def verify_node(self):
        pass


def test_attr():
    add_op_to_domain("qonnx.custom_op.general", AttrTestOp)
    ishp = (1, 10)
    wshp = (1, 3)
    oshp = wshp
    ishp_str = str(list(ishp))
    oshp_str = str(list(oshp))
    wshp_str = str(list(wshp))
    w = np.asarray([1, -2, 3], dtype=np.int8)
    strarr = np.array2string(w, separator=", ")
    w_str = strarr.replace("[", "{").replace("]", "}").replace(" ", "")
    tensor_attr_str = f"int8{wshp_str} {w_str}"
    strings_attr = ["a", "bc", "def"]

    input = f"""
    <
        ir_version: 7,
        opset_import: ["" : 9]
    >
    agraph (float{ishp_str} in0) => (int8{oshp_str} out0)
    {{
        out0 = qonnx.custom_op.general.AttrTestOp<
            tensor_attr={tensor_attr_str}
        >(in0)
    }}
    """
    model = oprs.parse_model(input)
    model = ModelWrapper(model)

    # Now getCustomOp should find it through the manual registry
    inst = getCustomOp(model.graph.node[0])

    w_prod = inst.get_nodeattr("tensor_attr")
    assert (w_prod == w).all()
    w = w - 1
    inst.set_nodeattr("tensor_attr", w)
    w_prod = inst.get_nodeattr("tensor_attr")
    assert (w_prod == w).all()

    inst.set_nodeattr("strings_attr", strings_attr)
    strings_attr_prod = inst.get_nodeattr("strings_attr")
    assert strings_attr_prod == strings_attr
    strings_attr_prod[0] = "test"
    inst.set_nodeattr("strings_attr", strings_attr_prod)
    assert inst.get_nodeattr("strings_attr") == ["test"] + strings_attr[1:]


class MyCustomOp(CustomOp):
    def __init__(self, onnx_node, onnx_opset_version=10):
        super().__init__(onnx_node, onnx_opset_version)

    def get_nodeattr_types(self):
        return {
            "my_int_attr": ("i", False, -1),
            "my_float_attr": ("f", False, 0.0),
            "my_string_attr": ("s", False, "default"),
            "my_ints_attr": ("ints", False, []),
            "my_floats_attr": ("floats", False, []),
            "my_strings_attr": ("strings", False, []),
            "my_allowed_attr": ("i", False, 1, {1, 2, 3}),
            "my_tensor_attr": ("t", False, np.array([])),
        }

    def execute_node(self, context, graph):
        pass

    def infer_node_datatype(self, model):
        pass

    def make_shape_compatible_op(self, model):
        pass

    def verify_node(self):
        pass


def test_set_get_nodeattr():
    node = helper.make_node("myOpType", [], [])
    myCustomOp = MyCustomOp(node, 13)

    # Test integer attribute
    assert myCustomOp.get_nodeattr("my_int_attr") == -1
    myCustomOp.set_nodeattr("my_int_attr", 2)
    assert myCustomOp.get_nodeattr("my_int_attr") == 2

    # Test that setting wrong type raises TypeError
    with pytest.raises(TypeError, match="expects int"):
        myCustomOp.set_nodeattr("my_int_attr", 2.5)
    with pytest.raises(TypeError, match="expects int"):
        myCustomOp.set_nodeattr("my_int_attr", "string")

    # Test float attribute
    assert myCustomOp.get_nodeattr("my_float_attr") == 0.0
    myCustomOp.set_nodeattr("my_float_attr", 3.14)
    assert abs(myCustomOp.get_nodeattr("my_float_attr") - 3.14) < 1e-6

    with pytest.raises(TypeError, match="expects float"):
        myCustomOp.set_nodeattr("my_float_attr", 42)
    with pytest.raises(TypeError, match="expects float"):
        myCustomOp.set_nodeattr("my_float_attr", "string")

    # Test string attribute
    assert myCustomOp.get_nodeattr("my_string_attr") == "default"
    myCustomOp.set_nodeattr("my_string_attr", "test_value")
    assert myCustomOp.get_nodeattr("my_string_attr") == "test_value"

    with pytest.raises(TypeError, match="expects str"):
        myCustomOp.set_nodeattr("my_string_attr", 123)
    with pytest.raises(TypeError, match="expects str"):
        myCustomOp.set_nodeattr("my_string_attr", 3.14)

    # Test ints attribute
    assert myCustomOp.get_nodeattr("my_ints_attr") == []
    myCustomOp.set_nodeattr("my_ints_attr", [1, 2, 3])
    assert myCustomOp.get_nodeattr("my_ints_attr") == [1, 2, 3]

    with pytest.raises(TypeError, match="expects list of ints"):
        myCustomOp.set_nodeattr("my_ints_attr", [1, 2.5, 3])
    with pytest.raises(TypeError, match="expects list of ints"):
        myCustomOp.set_nodeattr("my_ints_attr", [1, "two", 3])
    with pytest.raises(TypeError, match="expects list of ints"):
        myCustomOp.set_nodeattr("my_ints_attr", 123)

    # Test floats attribute
    assert myCustomOp.get_nodeattr("my_floats_attr") == []
    myCustomOp.set_nodeattr("my_floats_attr", [1.0, 2.5, 3.14])
    result = myCustomOp.get_nodeattr("my_floats_attr")
    assert len(result) == 3
    assert abs(result[0] - 1.0) < 1e-6
    assert abs(result[1] - 2.5) < 1e-6
    assert abs(result[2] - 3.14) < 1e-6
    # floats can accept ints
    myCustomOp.set_nodeattr("my_floats_attr", [1, 2, 3])
    assert myCustomOp.get_nodeattr("my_floats_attr") == [1, 2, 3]

    with pytest.raises(TypeError, match="expects list of floats"):
        myCustomOp.set_nodeattr("my_floats_attr", [1.0, "two", 3.0])
    with pytest.raises(TypeError, match="expects list of floats"):
        myCustomOp.set_nodeattr("my_floats_attr", 3.14)

    # Test strings attribute
    assert myCustomOp.get_nodeattr("my_strings_attr") == []
    myCustomOp.set_nodeattr("my_strings_attr", ["a", "b", "c"])
    assert myCustomOp.get_nodeattr("my_strings_attr") == ["a", "b", "c"]

    with pytest.raises(TypeError, match="expects list of strings"):
        myCustomOp.set_nodeattr("my_strings_attr", ["a", 2, "c"])
    with pytest.raises(TypeError, match="expects list of strings"):
        myCustomOp.set_nodeattr("my_strings_attr", "not a list")

    # Test allowed_values validation
    assert myCustomOp.get_nodeattr("my_allowed_attr") == 1
    myCustomOp.set_nodeattr("my_allowed_attr", 2)
    assert myCustomOp.get_nodeattr("my_allowed_attr") == 2
    myCustomOp.set_nodeattr("my_allowed_attr", 3)
    assert myCustomOp.get_nodeattr("my_allowed_attr") == 3

    with pytest.raises(ValueError, match="not in"):
        myCustomOp.set_nodeattr("my_allowed_attr", 5)

    # Test tensor attribute (numpy arrays)
    default_tensor = myCustomOp.get_nodeattr("my_tensor_attr")
    assert default_tensor.shape == (0,)

    # Set a 1D numpy array
    tensor_1d = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    myCustomOp.set_nodeattr("my_tensor_attr", tensor_1d)
    result_1d = myCustomOp.get_nodeattr("my_tensor_attr")
    assert np.array_equal(result_1d, tensor_1d)
    assert result_1d.dtype == tensor_1d.dtype

    # Set a 2D numpy array
    tensor_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    myCustomOp.set_nodeattr("my_tensor_attr", tensor_2d)
    result_2d = myCustomOp.get_nodeattr("my_tensor_attr")
    assert np.array_equal(result_2d, tensor_2d)
    assert result_2d.shape == tensor_2d.shape

    # Set a 3D numpy array with different dtype
    tensor_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int8)
    myCustomOp.set_nodeattr("my_tensor_attr", tensor_3d)
    result_3d = myCustomOp.get_nodeattr("my_tensor_attr")
    assert np.array_equal(result_3d, tensor_3d)
    assert result_3d.shape == (2, 2, 2)
    assert result_3d.dtype == np.int8

    # Test assigning numpy arrays to non-tensor attributes (should fail)
    numpy_arr = np.array([1, 2, 3])
    with pytest.raises(TypeError, match="expects int"):
        myCustomOp.set_nodeattr("my_int_attr", numpy_arr)
    with pytest.raises(TypeError, match="expects float"):
        myCustomOp.set_nodeattr("my_float_attr", numpy_arr)
    with pytest.raises(TypeError, match="expects str"):
        myCustomOp.set_nodeattr("my_string_attr", numpy_arr)
    with pytest.raises(TypeError, match="expects list of ints"):
        myCustomOp.set_nodeattr("my_ints_attr", numpy_arr)
    with pytest.raises(TypeError, match="expects list of floats"):
        myCustomOp.set_nodeattr("my_floats_attr", numpy_arr)
    with pytest.raises(TypeError, match="expects list of strings"):
        myCustomOp.set_nodeattr("my_strings_attr", numpy_arr)

    # Test assigning non-numpy values to tensor attribute (should fail or convert)
    # Scalars should fail
    with pytest.raises((TypeError, AttributeError)):
        myCustomOp.set_nodeattr("my_tensor_attr", 42)
    with pytest.raises((TypeError, AttributeError)):
        myCustomOp.set_nodeattr("my_tensor_attr", 3.14)
    with pytest.raises((TypeError, AttributeError)):
        myCustomOp.set_nodeattr("my_tensor_attr", "string")
    
    # Test assigning lists to tensor attribute (should fail with TypeError)
    # Plain lists are not accepted - must be numpy arrays
    with pytest.raises(TypeError, match="expects numpy array"):
        myCustomOp.set_nodeattr("my_tensor_attr", [10, 20, 30, 40])
    
    with pytest.raises(TypeError, match="expects numpy array"):
        myCustomOp.set_nodeattr("my_tensor_attr", [[1, 2, 3], [4, 5, 6]])
    
    with pytest.raises(TypeError, match="expects numpy array"):
        myCustomOp.set_nodeattr("my_tensor_attr", [1.5, 2.5, 3.5])
