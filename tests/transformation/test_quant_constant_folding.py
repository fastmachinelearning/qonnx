# Set pytest parameters
import pytest

# Numpy for handling simulation of tensor operations
import numpy as np

# Helper for creating ONNX nodes
from onnx import NodeProto, TensorProto  # noqa
from onnx import helper as oh  # noqa

# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper  # noqa

# Execute QONNX model graphs
from qonnx.core.onnx_exec import execute_onnx  # noqa

# QONNX quantizer function modeling the behavior of the Quant operator
from qonnx.custom_op.general.quant import quant as quant_fn  # noqa

# QONNX graph transformations for inferring datatypes and shapes required by
# test setup
from qonnx.transformation.infer_datatypes import InferDataTypes  # noqa
from qonnx.transformation.infer_shapes import InferShapes  # noqa

# Graph transformation to be tested: Transposes the initializers to Quantizer if
# ALL inputs are initializers
from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit  # noqa

# QONNX utility  for creating models from ONNX graphs
from qonnx.util.basic import qonnx_make_model  # noqa


@pytest.mark.parametrize("quant_init", [True, False])
@pytest.mark.parametrize("signed", [0, 1])
@pytest.mark.parametrize("narrow", [0, 1])
@pytest.mark.parametrize("rounding_mode", ["ROUND"])
@pytest.mark.parametrize("shape", [(16, 8, 12)])
@pytest.mark.parametrize(
    "perm",
    [
        # All axis permutations
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ],
)
@pytest.mark.parametrize("scale", [0.01])
@pytest.mark.parametrize("zeropoint", [0])
@pytest.mark.parametrize("bitwidth", [8])
# Tests the FoldTransposeIntoQuantInit transformation
def test_fold_transpose_into_quant_init(quant_init, signed, narrow, rounding_mode, shape, perm, scale, zeropoint, bitwidth):
    # Prepare the quantizer node attributes and input/output lists
    node_attrs = {
        # Type of the operation
        "op_type": "Quant",
        # This operator type is defined within QONNX
        "domain": "qonnx.custom_op.general",
        # List the inputs to the operator in order
        # Note: The proper input followed by initializers configuring the
        # quantizer
        "inputs": ["input", "scale", "zeropoint", "bitwidth"],
        # List the outputs of the operator in order
        # Note: Intermediate feeds to the next operator input
        "outputs": ["intermediate"],
        # Whether the quantization interval should be signed or not
        # (e.g. at 8b unsigned=[0, 255] vs signed=[-128, 127])
        "signed": signed,
        # When signed=1, whether to use narrow range or not
        # (e.g. at 8b regular=[-128, 127] vs narrow=[-127, 127])
        "narrow": narrow,
        # The rounding mode, which is used for the quant function
        "rounding_mode": rounding_mode,
    }
    # Create a dummy quantizer node
    quant = oh.make_node(**node_attrs, name="Quant")
    # Attach a Transpose operation to the quantizer
    transpose = oh.make_node("Transpose", ["intermediate"], ["output"], name="Transpose", perm=perm)
    # Derive the transposed shape
    transposed_shape = np.transpose(np.zeros(shape), perm).shape
    # Create tensor information for the input, intermediate and output tensor
    x = oh.make_tensor_value_info("input", TensorProto.FLOAT, shape)  # noqa
    y = oh.make_tensor_value_info("output", TensorProto.FLOAT, transposed_shape)
    # Create the initializer tensors for quantizer parameters
    s = oh.make_tensor_value_info("scale", TensorProto.FLOAT, (1,))
    z = oh.make_tensor_value_info("zeropoint", TensorProto.FLOAT, (1,))
    b = oh.make_tensor_value_info("bitwidth", TensorProto.FLOAT, (1,))
    # Create the graph connecting the nodes and tensors
    graph = oh.make_graph(
        [quant, transpose],
        "quant-transpose",
        [x, s, z, b],
        [y],
    )
    # Wrap the graph in an QONNX model wrapper
    model = ModelWrapper(qonnx_make_model(graph, producer_name="qonnx-tests"))
    # Add the actual initializers to the initializer tensors
    model.set_initializer("scale", np.array(scale))
    model.set_initializer("zeropoint", np.array(zeropoint))
    model.set_initializer("bitwidth", np.array(bitwidth))
    # Prepare the model graph by inferring all missing shape and datatype
    # information
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # Get a random dummy input for testing
    x = np.random.rand(*shape)  # noqa
    # Fill the execution context with dummy input data
    context = {"input": x}

    # Some test cases even turn the input into an initializer
    if quant_init:
        # Turn the model input into an initializer
        model.set_initializer("input", x)
        # Clear the execution context removing the input as it is now baked into
        # the model graph
        context = {}

    # Run the transformation to be tested
    model = model.transform(FoldTransposeIntoQuantInit())
    # Verify that shape and datatype inference still works
    # Note: This has been an issue, please see
    #   https://github.com/fastmachinelearning/qonnx/issues/77
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # For the case of quant-initializers there must not be a Transpose left
    # after transforming and contrariwise, the Transpose must remain in place if
    # there is non-initializer input.
    assert quant_init != ("Transpose" in [n.op_type for n in model.graph.node])

    # Execute the ONNX model
    o_produced = execute_onnx(model, context)["output"]
    # Use numpy and QONNX quantizer to generate expectation
    o_expected = np.transpose(
        quant_fn(x, np.array(scale), np.array(zeropoint), np.array(bitwidth), signed, narrow, rounding_mode), perm
    )

    # The output must match the "manual" execution using numpy
    assert np.allclose(o_produced, o_expected)
