# Set pytest parameters
import pytest

# Numpy for handling simulation of tensor operations
import numpy as np

# Helper for creating ONNX nodes
from onnx import TensorProto
from onnx import helper as oh

# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper

# Execute QONNX model graphs
from qonnx.core.onnx_exec import execute_onnx

# Graph transformation to be tested: Sorts the input list of commutative
# operations to have all dynamic inputs first followed by all initializer inputs
from qonnx.transformation.general import SortCommutativeInputsInitializerLast

# QONNX utility  for creating models from ONNX graphs
from qonnx.util.basic import qonnx_make_model


# Specify how many inputs the test should cover
@pytest.mark.parametrize("num_inputs", [4, 5, 6])
# Specify which inputs should be turned into initializers
@pytest.mark.parametrize(
    # fmt: off
    "initializers", [[], [0], [1], [0, 1], [0, 3], [0, 1, 2, 3]]
    # fmt: on
)
# Tests the SortCommutativeInputsInitializerLast transformation
def test_sort_commutative_inputs_initializer_last(num_inputs, initializers):
    # Generate the input tensor names
    inputs = [f"in{i}" for i in range(num_inputs)]
    # We will use the Sum ONNX operation to test this behavior, as it allows for
    # arbitrary many inputs
    node = oh.make_node(
        # fmt: off
        op_type="Sum", inputs=inputs, outputs=["out"], name="Sum"
        # fmt: on
    )
    # Create value infos for all input and the output tensor
    inputs = [
        # fmt: off
        oh.make_tensor_value_info(i, TensorProto.FLOAT, (16,)) for i in inputs
        # fmt: on
    ]
    out = oh.make_tensor_value_info("out", TensorProto.FLOAT, (16,))
    # Make a graph comprising the Sum node and value infos for all inputs and
    # the output
    graph = oh.make_graph([node], inputs=inputs, outputs=[out], name="Sum")
    # Wrap the graph in an QONNX model wrapper
    model = ModelWrapper(qonnx_make_model(graph, producer_name="qonnx-tests"))
    # Prepare the execution context
    context = {f"in{i}": np.random.rand(16) for i in range(num_inputs)}
    # Make sure all inputs are of type float32
    context = {key: value.astype(np.float32) for key, value in context.items()}
    # Turn selected inputs into initializers
    for i in initializers:
        model.set_initializer(f"in{i}", context[f"in{i}"])

    # Execute the ONNX model before transforming
    out_expected = execute_onnx(model, context)["out"]
    # Apply the transformation to be tested
    # Note: No cleanup, as the tested transformation is part of the cleanup, and
    # we want to test this in isolation
    model = model.transform(
        # fmt: off
        SortCommutativeInputsInitializerLast(), cleanup=False
        # fmt: on
    )
    # Execute the ONNX model after transforming
    out_produced = execute_onnx(model, context)["out"]

    # Start with no initializer input seen so far
    seen_initializer = False
    # Verify that no "dynamic" input follows an initializer input
    for i in model.graph.node[0].input:
        # Keep track of when an initializer has been seen
        if model.get_initializer(i) is not None:
            seen_initializer = True
        # If there has already been an initializer, this input must be an
        # initializer as well
        assert (
            not seen_initializer or model.get_initializer(i) is not None
        ), "Non-initializer input following initializer after sorting"

    # Outputs before and after must match
    assert np.allclose(out_produced, out_expected)
