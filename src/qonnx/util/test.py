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
import clize
import numpy as np
import os
import urllib.request

import qonnx.core.onnx_exec as oxe
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup

# utility functions to fetch models and data for
# testing various qonnx transformations

test_model_details = {
    "FINN-CNV_W2A2": {
        "description": "2-bit VGG-10-like CNN on CIFAR-10",
        "url": (
            "https://raw.githubusercontent.com/fastmachinelearning/"
            "QONNX_model_zoo/main/models/CIFAR10/Brevitas_FINN_CNV/CNV_2W2A.onnx"
        ),
        "input_shape": (1, 3, 32, 32),
        "input_range": (0, +1),
    },
    "FINN-CNV_W1A2": {
        "description": "1/2-bit VGG-10-like CNN on CIFAR-10",
        "url": (
            "https://raw.githubusercontent.com/fastmachinelearning/"
            "QONNX_model_zoo/main/models/CIFAR10/Brevitas_FINN_CNV/CNV_1W2A.onnx"
        ),
        "input_shape": (1, 3, 32, 32),
        "input_range": (0, +1),
    },
    "FINN-CNV_W1A1": {
        "description": "1-bit VGG-10-like CNN on CIFAR-10",
        "url": (
            "https://raw.githubusercontent.com/fastmachinelearning/"
            "QONNX_model_zoo/main/models/CIFAR10/Brevitas_FINN_CNV/CNV_1W1A.onnx"
        ),
        "input_shape": (1, 3, 32, 32),
        "input_range": (0, +1),
    },
    "FINN-TFC_W1A1": {
        "description": "1-bit tiny MLP on MNIST",
        "url": (
            "https://github.com/fastmachinelearning/QONNX_model_zoo/"
            "raw/main/models/MNIST/Brevitas_FINN_TFC/TFC/TFC_1W1A.onnx"
        ),
        "input_shape": (1, 1, 28, 28),
        "input_range": (0, +1),
    },
    "FINN-TFC_W1A2": {
        "description": "1/2-bit tiny MLP on MNIST",
        "url": (
            "https://github.com/fastmachinelearning/QONNX_model_zoo/"
            "raw/main/models/MNIST/Brevitas_FINN_TFC/TFC/TFC_1W2A.onnx"
        ),
        "input_shape": (1, 1, 28, 28),
        "input_range": (0, +1),
    },
    "FINN-TFC_W2A2": {
        "description": "2-bit tiny MLP on MNIST",
        "url": (
            "https://github.com/fastmachinelearning/QONNX_model_zoo/"
            "raw/main/models/MNIST/Brevitas_FINN_TFC/TFC/TFC_2W2A.onnx"
        ),
        "input_shape": (1, 1, 28, 28),
        "input_range": (0, +1),
    },
    "RadioML_VGG10": {
        "description": "8-bit VGG-10-like CNN on RadioML 2018",
        "url": (
            "https://github.com/Xilinx/brevitas-radioml-challenge-21/raw/"
            "9eef6a2417d6a0c078bfcc3a4dc95033739c5550/sandbox/notebooks/models/pretrained_VGG10_w8a8_20_export.onnx"
        ),
        "input_shape": (1, 2, 1024),
        "input_range": (-1, +1),
    },
    "Conv_bias_example": {
        "description": "",
        "url": "https://zenodo.org/record/7626922/files/super_resolution.onnx",
        "input_shape": (1, 1, 28, 28),
        "input_range": (-1, +1),
    },
    "MobileNetv1-w4a4": {
        "description": "4-bit MobileNet-v1 on ImageNet",
        "url": (
            "https://raw.githubusercontent.com/fastmachinelearning/"
            "qonnx_model_zoo/main/models/ImageNet/Brevitas_FINN_mobilenet/mobilenet_4W4A.onnx"
        ),
        "input_shape": (1, 3, 224, 224),
        "input_range": (0, 1),
    },
}


test_model_keys = clize.parameters.mapped(
    [(x, [x], test_model_details[x]["description"]) for x in test_model_details.keys()]
)


def download_model(test_model: test_model_keys, *, dl_dir="/tmp", do_cleanup=False, return_modelwrapper=False):
    qonnx_url = test_model_details[test_model]["url"]
    # download test data
    dl_file = dl_dir + f"/{test_model}.onnx"
    ret = dl_file
    if not os.path.isfile(dl_file):
        urllib.request.urlretrieve(qonnx_url, dl_file)
    if do_cleanup:
        # run cleanup with default settings
        out_file = dl_dir + f"/{test_model}_clean.onnx"
        cleanup(dl_file, out_file=out_file)
        ret = out_file
    if return_modelwrapper:
        ret = ModelWrapper(ret)
    return ret


def qonnx_download_model():
    clize.run(download_model)


def get_golden_in_and_output(test_model):
    model = download_model(test_model, do_cleanup=True, return_modelwrapper=True)
    rng = np.random.RandomState(42)
    input_shape = test_model_details[test_model]["input_shape"]
    (low, high) = test_model_details[test_model]["input_range"]
    size = np.prod(np.asarray(input_shape))
    input_tensor = rng.uniform(low=low, high=high, size=size)
    input_tensor = input_tensor.astype(np.float32)
    input_tensor = input_tensor.reshape(input_shape)
    input_dict = {model.graph.input[0].name: input_tensor}
    golden_output_dict = oxe.execute_onnx(model, input_dict)
    golden_result = golden_output_dict[model.graph.output[0].name]
    return input_tensor, golden_result
