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

import pytest

import numpy as np
import os
import urllib.request

import qonnx.core.onnx_exec as oxe
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.qcdq_to_qonnx import QCDQToQuant
from qonnx.util.cleanup import cleanup_model

model_details = {
    "MobileNetv2-w8a8": {
        "url": ("https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-12-qdq.onnx"),
        "input_shape": (1, 3, 224, 224),
        "input_range": (-1, +1),
        "exp_q_nodes": 171,
    },
}


def download_model(test_model):
    qonnx_url = model_details[test_model]["url"]
    # download test data
    dl_dir = "/tmp"
    dl_file = dl_dir + f"/{test_model}.onnx"
    urllib.request.urlretrieve(qonnx_url, dl_file)
    return dl_file


def get_golden_in_and_output(model, test_model):
    rng = np.random.RandomState(42)
    input_shape = model_details[test_model]["input_shape"]
    (low, high) = model_details[test_model]["input_range"]
    size = np.prod(np.asarray(input_shape))
    input_tensor = rng.uniform(low=low, high=high, size=size)
    input_tensor = input_tensor.astype(np.float32)
    input_tensor = input_tensor.reshape(input_shape)
    # use batch dim of 1 where needed (appears as 0-valued dim)
    for tensor_name in model.get_all_tensor_names():
        ts = model.get_tensor_shape(tensor_name)
        if len(ts) > 0 and ts[0] == 0:
            ts = list(ts)
            ts[0] = 1
            model.set_tensor_shape(tensor_name, ts)
    model.set_tensor_shape(model.graph.input[0].name, input_shape)
    input_dict = {model.graph.input[0].name: input_tensor}
    golden_output_dict = oxe.execute_onnx(model, input_dict)
    golden_result = golden_output_dict[model.graph.output[0].name]
    return input_tensor, golden_result


@pytest.mark.parametrize("test_model", model_details.keys())
def test_qcdq_to_qonnx(test_model):
    test_details = model_details[test_model]
    dl_file = download_model(test_model=test_model)
    assert os.path.isfile(dl_file)
    model = ModelWrapper(dl_file)
    model = cleanup_model(model)
    input_tensor, golden_result = get_golden_in_and_output(model, test_model)
    model = model.transform(QCDQToQuant())
    assert len(model.get_nodes_by_op_type("Quant")) == test_details["exp_q_nodes"]
    assert len(model.get_nodes_by_op_type("QuantizeLinear")) == 0
    assert len(model.get_nodes_by_op_type("DequantizeLinear")) == 0
    if test_model == "MobileNetv2-w8a8":
        pytest.xfail("MNv2 known to have off-by-one difference in inputs to Conv_30")
    model = cleanup_model(model)
    input_dict = {model.graph.input[0].name: input_tensor}
    produced_output_dict = oxe.execute_onnx(model, input_dict)
    produced_result = produced_output_dict[model.graph.output[0].name]
    assert np.isclose(golden_result, produced_result).all()
