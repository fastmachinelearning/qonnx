import pytest

import numpy as np
import urllib.request

import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from qonnx.transformation.channelsLast import ConvertToChannelsLastAndClean
from qonnx.util.cleanup import cleanup


def download_model(test_model):
    if test_model == "FINN-CNV_W2A2":
        qonnx_url = "https://drive.google.com/uc?id=1BAyYM3N28fDHSWMgK8dZ3kVUsxGZLO3U&export=download"

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
    if test_model == "FINN-CNV_W2A2":
        input_shape = (1, 3, 32, 32)
    elif "RadioML_VGG10":
        input_shape = (1, 2, 1024)
    else:
        raise ValueError(f"Model called {test_model} is not supported")
    input_tensor = np.ones(np.prod(np.asarray(input_shape)), dtype=np.float32)
    input_tensor = input_tensor.reshape(input_shape)

    model = ModelWrapper(onnx_file)
    input_dict = {model.graph.input[0].name: input_tensor}
    golden_output_dict = oxe.execute_onnx(model, input_dict, True)
    golden_result = golden_output_dict[model.graph.output[0].name]

    return input_tensor, golden_result


@pytest.mark.parametrize("test_model", ["FINN-CNV_W2A2", "RadioML_VGG10"])
def test_ChannelsLast_conversion_end2end(test_model):
    # Download an clean model
    onnx_file = download_model(test_model)
    input_tensor, golden_result = get_golden_in_and_output(onnx_file, test_model)

    # Execute transformation
    model = ModelWrapper(onnx_file)
    qonnx_all_trafos = onnx_file.split(".onnx")[0] + "_all_nhwc_trafos.onnx"
    model = model.transform(ConvertToChannelsLastAndClean())
    model.save(qonnx_all_trafos)
    print(f"hi, saved to: {qonnx_all_trafos}")

    # Check output
    input_dict = {model.graph.input[0].name: input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict, True)
    current_result = output_dict[model.graph.output[0].name]
    assert (golden_result == current_result).all(), "Output of cleaned QONNX model and channels last model should match."


# @pytest.mark.parametrize("test_model", ["FINN-CNV_W2A2", "RadioML_VGG10"])
# def test_ChannelsLast_conversion_step_by_step(test_model):
#     #onnx_file = download_model(test_model)
#     #input_tensor, golden_result = get_golden_in_and_output(onnx_file, test_model)
#
#     raise NotImplementedError
