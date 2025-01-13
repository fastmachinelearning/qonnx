# Copyright (c) 2025 Advanced Micro Devices, Inc.
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
# * Neither the name of AMD nor the names of its
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

import pytest

from onnx import TensorProto
from onnx import helper as oh

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.insert import InsertIdentity


@pytest.fixture
def simple_model():
    # Create a simple ONNX model for testing
    input_tensor = oh.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
    output_tensor = oh.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])
    node1 = oh.make_node("Relu", ["input"], ["intermediate"])
    node2 = oh.make_node("Relu", ["intermediate"], ["output"])
    graph = oh.make_graph([node1, node2], "test_graph", [input_tensor], [output_tensor])
    model = ModelWrapper(oh.make_model(graph))
    model = model.transform(InferShapes())
    return model


def save_transformed_model(model, test_name):
    output_path = f"{test_name}.onnx"
    model.save(output_path)


def test_insert_identity_before_input(simple_model):
    # Apply the transformation
    transformation = InsertIdentity("input", "producer")
    model = simple_model.transform(transformation)

    identity_node = model.find_producer("input")
    assert identity_node is not None
    assert identity_node.op_type == "Identity"

    # Save the transformed model
    save_transformed_model(model, "test_insert_identity_before_input")


def test_insert_identity_after_input(simple_model):
    # Apply the transformation
    transformation = InsertIdentity("input", "consumer")
    model = simple_model.transform(transformation)

    identity_node = model.find_consumer("input")
    assert identity_node is not None
    assert identity_node.op_type == "Identity"

    # Save the transformed model
    save_transformed_model(model, "test_insert_identity_after_input")


def test_insert_identity_before_intermediate(simple_model):
    # Apply the transformation
    transformation = InsertIdentity("intermediate", "producer")
    model = simple_model.transform(transformation)

    identity_node = model.find_producer("intermediate")
    assert identity_node is not None
    assert identity_node.op_type == "Identity"

    # Save the transformed model
    save_transformed_model(model, "test_insert_identity_before_intermediate")


def test_insert_identity_after_intermediate(simple_model):
    # Apply the transformation
    transformation = InsertIdentity("intermediate", "consumer")
    model = simple_model.transform(transformation)

    identity_node = model.find_consumer("intermediate")
    assert identity_node is not None
    assert identity_node.op_type == "Identity"

    # Save the transformed model
    save_transformed_model(model, "test_insert_identity_after_intermediate")


def test_insert_identity_before_output(simple_model):
    # Apply the transformation
    transformation = InsertIdentity("output", "producer")
    model = simple_model.transform(transformation)

    identity_node = model.find_producer("output")
    assert identity_node is not None
    assert identity_node.op_type == "Identity"

    # Save the transformed model
    save_transformed_model(model, "test_insert_identity_before_output")


def test_insert_identity_after_output(simple_model):
    # Apply the transformation
    transformation = InsertIdentity("output", "consumer")
    model = simple_model.transform(transformation)

    identity_node = model.find_consumer("output")
    assert identity_node is not None
    assert identity_node.op_type == "Identity"

    # Save the transformed model
    save_transformed_model(model, "test_insert_identity_after_output")


def test_tensor_not_found(simple_model):
    # Apply the transformation with a non-existent tensor
    transformation = InsertIdentity("non_existent_tensor", "producer")
    with pytest.raises(ValueError):
        simple_model.transform(transformation)
