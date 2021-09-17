import pytest

import numpy as np
import urllib.request

import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp
from finn.transformation.general import GiveUniqueNodeNames
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.make_input_chanlast import MakeInputChannelsLast
from finn.util.basic import is_finn_op
from qonnx.custom_op import channels_last
from qonnx.custom_op.channels_last.base_wrapped_op import to_channels_last_args
from qonnx.transformation.channels_last import (
    AbsorbChanFirstIntoMatMul,
    ConvertToChannelsLastAndClean,
    InsertChannelsLastDomainsAndTrafos,
    MoveChanFirstDownstream,
    MoveChanLastUpstream,
    RemoveConsecutiveChanFirstAndChanLastTrafos,
)
from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit
from qonnx.util.cleanup import cleanup


def download_model(test_model):
    if test_model == "FINN-CNV_W2A2":
        qonnx_url = (
            "https://raw.githubusercontent.com/fastmachinelearning/"
            "QONNX_model_zoo/main/models/CIFAR10/Brevitas_FINN_CNV/CNV_2W2A.onnx"
        )

    elif "RadioML_VGG10":
        qonnx_url = (
            "https://github.com/Xilinx/brevitas-radioml-challenge-21/raw/"
            "9eef6a2417d6a0c078bfcc3a4dc95033739c5550/sandbox/notebooks/models/pretrained_VGG10_w8a8_20_export.onnx"
        )
    else:
        raise ValueError(f"Model called {test_model} is not supported")

    # download test data
    dl_dir = "/tmp"
    dl_file = dl_dir + f"/{test_model}.onnx"
    urllib.request.urlretrieve(qonnx_url, dl_file)
    # run cleanup with default settings
    out_file = dl_dir + f"/{test_model}_clean.onnx"
    cleanup(dl_file, out_file=out_file)
    return out_file


def get_golden_in_and_output(onnx_file, test_model):
    rng = np.random.RandomState(42)
    if test_model == "FINN-CNV_W2A2":
        input_shape = (1, 3, 32, 32)
        size = np.prod(np.asarray(input_shape))
        input_tensor = rng.uniform(low=-1.0, high=1.0 - 2.0 ** (-7), size=size)

    elif "RadioML_VGG10":
        input_shape = (1, 2, 1024)
        size = np.prod(np.asarray(input_shape))
        input_tensor = rng.uniform(low=-2, high=2, size=size)
    else:
        raise ValueError(f"Model called {test_model} is not supported")
    input_tensor = input_tensor.astype(np.float32)
    input_tensor = input_tensor.reshape(input_shape)

    model = ModelWrapper(onnx_file)
    input_dict = {model.graph.input[0].name: input_tensor}
    golden_output_dict = oxe.execute_onnx(model, input_dict)
    golden_result = golden_output_dict[model.graph.output[0].name]

    return input_tensor, golden_result


def analysis_testing_for_chanLast_domain(model):
    # Define for each ChannelsLast operation the number of minimum dimensions, it needs to have
    ChanLast_node_types_and_min_dim_input = {
        "Conv": 3,
        "MaxPool": 3,
        "BatchNormalization": 3,
    }
    # Check that all wrapped_ops in the registry have a definition here
    chanLast_op_types = list(channels_last.custom_op.keys())
    testable_op_types = list(ChanLast_node_types_and_min_dim_input.keys())
    for op_name in chanLast_op_types:
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
@pytest.mark.parametrize("test_model", ["FINN-CNV_W2A2", "RadioML_VGG10"])
def test_ChannelsLast_conversion_end2end(test_model, make_input_channels_last):
    # Download an clean model
    onnx_file = download_model(test_model)
    input_tensor, golden_result = get_golden_in_and_output(onnx_file, test_model)

    # Execute transformation
    model = ModelWrapper(onnx_file)
    qonnx_all_trafos = onnx_file.split(".onnx")[0] + "_all_nhwc_trafos_test.onnx"
    model = model.transform(ConvertToChannelsLastAndClean(make_input_channels_last=make_input_channels_last))
    model.save(qonnx_all_trafos)

    # Check output
    if make_input_channels_last:
        input_dims = len(model.get_tensor_shape(model.graph.input[0].name))
        input_tensor = input_tensor.transpose(to_channels_last_args(input_dims))
    input_dict = {model.graph.input[0].name: input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict)
    current_result = output_dict[model.graph.output[0].name]
    assert (golden_result == current_result).all(), "Output of cleaned QONNX model and channels last model should match."
    assert model.check_all_tensor_shapes_specified(), "All tensor shapes should be specified."

    # Check that the ops, which should be ChannelsLast actually are
    _ = model.analysis(analysis_testing_for_chanLast_domain)

    # This would throw an error if anything is misconfigured
    _ = model.analysis(verify_all_nodes)

    # Check that the first node is a transpose node
    if not make_input_channels_last:
        _ = model.analysis(analysis_first_node_is_transpose)


@pytest.mark.parametrize("test_model", ["FINN-CNV_W2A2", "RadioML_VGG10"])
def test_ChannelsLast_conversion_step_by_step(test_model):
    # Download an clean model
    onnx_file = download_model(test_model)
    input_tensor, golden_result = get_golden_in_and_output(onnx_file, test_model)

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
    assert (
        golden_result == current_result
    ).all(), "Output of cleaned QONNX model and model after applying InsertChannelsLastDomainsAndTrafos should match."
    assert model.check_all_tensor_shapes_specified(), "All tensor shapes should be specified."

    # Run trafo
    model = model.transform(RemoveConsecutiveChanFirstAndChanLastTrafos())
    # Check output
    input_dict = {model.graph.input[0].name: input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict)
    current_result = output_dict[model.graph.output[0].name]
    assert (
        golden_result == current_result
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
    assert (
        golden_result == current_result
    ).all(), "Output of cleaned QONNX model and model after applying MoveChanLastUpstream should match."
    assert model.check_all_tensor_shapes_specified(), "All tensor shapes should be specified."

    # Run trafo
    model = model.transform(FoldTransposeIntoQuantInit())
    # Check output
    input_dict = {model.graph.input[0].name: input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict)
    current_result = output_dict[model.graph.output[0].name]
    assert (
        golden_result == current_result
    ).all(), "Output of cleaned QONNX model and model after applying MoveChanLastUpstream should match."
    assert model.check_all_tensor_shapes_specified(), "All tensor shapes should be specified."

    # Run trafo
    model = model.transform(RemoveConsecutiveChanFirstAndChanLastTrafos())
    # Check output
    input_dict = {model.graph.input[0].name: input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict)
    current_result = output_dict[model.graph.output[0].name]
    assert (
        golden_result == current_result
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
    assert (
        golden_result == current_result
    ).all(), "Output of cleaned QONNX model and model after applying MoveChanFirstDownstream should match."
    assert model.check_all_tensor_shapes_specified(), "All tensor shapes should be specified."

    # Run trafo
    model = model.transform(AbsorbChanFirstIntoMatMul())
    model = model.transform(InferShapes())
    # Check output
    input_dict = {model.graph.input[0].name: input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict)
    current_result = output_dict[model.graph.output[0].name]
    assert (
        golden_result == current_result
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
    assert (
        golden_result == current_result
    ).all(), "Output of cleaned QONNX model and model after applying AbsorbChanFirstIntoMatMul should match."
    assert model.check_all_tensor_shapes_specified(), "All tensor shapes should be specified."

    model.save(qonnx_all_trafos)
