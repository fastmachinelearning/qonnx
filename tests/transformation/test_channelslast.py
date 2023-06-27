import pytest

import numpy as np

import qonnx.core.onnx_exec as oxe
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op import channels_last
from qonnx.custom_op.channels_last.base_wrapped_op import to_channels_last_args
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.channels_last import (
    AbsorbChanFirstIntoMatMul,
    InsertChannelsLastDomainsAndTrafos,
    MoveChanFirstDownstream,
    MoveChanLastUpstream,
    RemoveConsecutiveChanFirstAndChanLastTrafos,
)
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.make_input_chanlast import MakeInputChannelsLast
from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit
from qonnx.util.basic import is_finn_op
from qonnx.util.test import download_model, get_golden_in_and_output, test_model_details
from qonnx.util.to_channels_last import to_channels_last

model_details_chanlast = {
    "FINN-CNV_W2A2": {
        "layout_sensitive": True,
    },
    "FINN-TFC_W2A2": {
        "layout_sensitive": False,
    },
    "RadioML_VGG10": {
        "layout_sensitive": True,
    },
    "Conv_bias_example": {
        "layout_sensitive": True,
    },
}
# inherit basics for matching testcases from test util
model_details = {k: v for (k, v) in test_model_details.items() if k in model_details_chanlast.keys()}
model_details = {**model_details, **model_details_chanlast}


def analysis_testing_for_chanlast_domain(model):
    # Define for each ChannelsLast operation the number of minimum dimensions, it needs to have
    ChanLast_node_types_and_min_dim_input = {
        "Conv": 3,
        "MaxPool": 3,
        "BatchNormalization": 3,
    }
    # Check that all wrapped_ops in the registry have a definition here
    chanlast_op_types = list(channels_last.custom_op.keys())
    testable_op_types = list(ChanLast_node_types_and_min_dim_input.keys())
    for op_name in chanlast_op_types:
        assert (
            op_name in testable_op_types
        ), f"The channelsLast op {op_name} is missing a definition for the domain string test."

    for n_type, min_dim in ChanLast_node_types_and_min_dim_input.items():
        nodes = model.get_nodes_by_op_type(n_type)
        for n in nodes:
            input_shape = model.get_tensor_shape(n.input[0])
            if len(input_shape) >= min_dim:
                assert (
                    n.domain == "qonnx.custom_op.channels_last"
                ), f"Node domain is not set correctly for node with name: {n.name}"
    return dict()


def analysis_test_for_left_transposes(model, test_model, make_input_channels_last):
    t_nodes = model.get_nodes_by_op_type("Transpose")
    if test_model == "FINN-TFC_W2A2":
        # For the TFC the assertion should be the other way around.
        make_input_channels_last = not make_input_channels_last
    if make_input_channels_last:
        assert len(t_nodes) == 0, "There should be no transposes left in the network."
    else:
        assert len(t_nodes) == 1, "There should be only one transposes node left in the network."
    return dict()


def verify_all_nodes(model):
    result = dict()
    for n in model.graph.node:
        if is_finn_op(n.domain):
            n_instance = getCustomOp(n)
            verify_result = n_instance.verify_node()
            result[n.name] = verify_result
    return result


def analysis_first_node_is_transpose(model):
    result = dict()
    input_tensor = model.graph.input[0].name
    first_node = model.find_consumer(input_tensor)
    assert first_node.op_type == "Transpose", "First node in the network should be a transpose node."
    return result


@pytest.mark.parametrize("make_input_channels_last", [True, False])
@pytest.mark.parametrize("test_model", model_details.keys())
def test_channelslast_conversion_end2end(test_model, make_input_channels_last):
    # Download an clean model
    onnx_file = download_model(test_model, do_cleanup=True)
    input_tensor, golden_result = get_golden_in_and_output(test_model)

    # Execute transformation
    qonnx_all_trafos = onnx_file.split(".onnx")[0] + "_all_nhwc_trafos_test.onnx"
    to_channels_last(onnx_file, make_input_channels_last=make_input_channels_last, out_file=qonnx_all_trafos)

    # Check output
    model = ModelWrapper(qonnx_all_trafos)
    if make_input_channels_last:
        input_dims = len(model.get_tensor_shape(model.graph.input[0].name))
        input_tensor = input_tensor.transpose(to_channels_last_args(input_dims))
    input_dict = {model.graph.input[0].name: input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict)
    current_result = output_dict[model.graph.output[0].name]
    assert np.isclose(
        golden_result, current_result, atol=1e-7
    ).all(), "Output of cleaned QONNX model and channels last model should match."
    assert model.check_all_tensor_shapes_specified(), "All tensor shapes should be specified."

    # Check that the ops, which should be ChannelsLast actually are
    _ = model.analysis(analysis_testing_for_chanlast_domain)

    # This would throw an error if anything is misconfigured
    _ = model.analysis(verify_all_nodes)

    # Check that the transposes are gone
    _ = analysis_test_for_left_transposes(model, test_model, make_input_channels_last)

    # Check that the first node is a transpose node for layout-sensitive models
    if (not make_input_channels_last) and (model_details[test_model]["layout_sensitive"]):
        _ = model.analysis(analysis_first_node_is_transpose)


@pytest.mark.parametrize("test_model", model_details.keys())
def test_channelslast_conversion_step_by_step(test_model):
    # Download an clean model
    onnx_file = download_model(test_model, do_cleanup=True)
    input_tensor, golden_result = get_golden_in_and_output(test_model)

    # Execute transformation
    model = ModelWrapper(onnx_file)
    qonnx_all_trafos = onnx_file.split(".onnx")[0] + "_all_nhwc_trafos.onnx"

    # Run trafo
    model = model.transform(InsertChannelsLastDomainsAndTrafos())
    model = model.transform(GiveUniqueNodeNames())
    # Check output
    input_dict = {model.graph.input[0].name: input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict)
    current_result = output_dict[model.graph.output[0].name]
    assert np.isclose(
        golden_result, current_result, atol=1e-7
    ).all(), "Output of cleaned QONNX model and model after applying InsertChannelsLastDomainsAndTrafos should match."
    assert model.check_all_tensor_shapes_specified(), "All tensor shapes should be specified."

    # Run trafo
    model = model.transform(RemoveConsecutiveChanFirstAndChanLastTrafos())
    # Check output
    input_dict = {model.graph.input[0].name: input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict)
    current_result = output_dict[model.graph.output[0].name]
    assert np.isclose(
        golden_result, current_result, atol=1e-7
    ).all(), (
        "Output of cleaned QONNX model and model after applying RemoveConsecutiveChanFirstAndChanLastTrafos should match."
    )
    assert model.check_all_tensor_shapes_specified(), "All tensor shapes should be specified."

    # Run trafo
    model = model.transform(MoveChanLastUpstream())
    # Check output
    input_dict = {model.graph.input[0].name: input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict)
    current_result = output_dict[model.graph.output[0].name]
    assert np.isclose(
        golden_result, current_result, atol=1e-7
    ).all(), "Output of cleaned QONNX model and model after applying MoveChanLastUpstream should match."
    assert model.check_all_tensor_shapes_specified(), "All tensor shapes should be specified."

    # Run trafo
    model = model.transform(FoldTransposeIntoQuantInit())
    # Check output
    input_dict = {model.graph.input[0].name: input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict)
    current_result = output_dict[model.graph.output[0].name]
    assert np.isclose(
        golden_result, current_result, atol=1e-7
    ).all(), "Output of cleaned QONNX model and model after applying MoveChanLastUpstream should match."
    assert model.check_all_tensor_shapes_specified(), "All tensor shapes should be specified."

    # Run trafo
    model = model.transform(RemoveConsecutiveChanFirstAndChanLastTrafos())
    # Check output
    input_dict = {model.graph.input[0].name: input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict)
    current_result = output_dict[model.graph.output[0].name]
    assert np.isclose(
        golden_result, current_result, atol=1e-7
    ).all(), (
        "Output of cleaned QONNX model and model after applying RemoveConsecutiveChanFirstAndChanLastTrafos should match."
    )
    assert model.check_all_tensor_shapes_specified(), "All tensor shapes should be specified."

    # Run trafo
    model = model.transform(MoveChanFirstDownstream())
    # Check output
    input_dict = {model.graph.input[0].name: input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict)
    current_result = output_dict[model.graph.output[0].name]
    assert np.isclose(
        golden_result, current_result, atol=1e-7
    ).all(), "Output of cleaned QONNX model and model after applying MoveChanFirstDownstream should match."
    assert model.check_all_tensor_shapes_specified(), "All tensor shapes should be specified."

    # Run trafo
    model = model.transform(AbsorbChanFirstIntoMatMul())
    model = model.transform(InferShapes())
    # Check output
    input_dict = {model.graph.input[0].name: input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict)
    current_result = output_dict[model.graph.output[0].name]
    assert np.isclose(
        golden_result, current_result, atol=1e-7
    ).all(), "Output of cleaned QONNX model and model after applying AbsorbChanFirstIntoMatMul should match."
    assert model.check_all_tensor_shapes_specified(), "All tensor shapes should be specified."

    # Run trafo
    model = model.transform(MakeInputChannelsLast())
    model = model.transform(RemoveConsecutiveChanFirstAndChanLastTrafos())
    # Check output
    input_dims = len(model.get_tensor_shape(model.graph.input[0].name))
    input_tensor = input_tensor.transpose(to_channels_last_args(input_dims))
    input_dict = {model.graph.input[0].name: input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict)
    current_result = output_dict[model.graph.output[0].name]
    assert np.isclose(
        golden_result, current_result, atol=1e-7
    ).all(), "Output of cleaned QONNX model and model after applying AbsorbChanFirstIntoMatMul should match."
    assert model.check_all_tensor_shapes_specified(), "All tensor shapes should be specified."

    model.save(qonnx_all_trafos)
