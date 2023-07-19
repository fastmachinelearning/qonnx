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

import qonnx.core.onnx_exec as oxe
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.qcdq_to_qonnx import QCDQToQuant
from qonnx.transformation.qonnx_to_qcdq import QuantToQCDQ
from qonnx.util.cleanup import cleanup_model
from qonnx.util.test import download_model, get_golden_in_and_output, test_model_details

qonnxtoqcdq_details = {
    "FINN-CNV_W2A2": {
        "nonconvertible_quant": 0,
        "exp_qdq_nodes": 18,
        # input quantizer doesn't need Clip so 1 less
        "exp_clip_nodes": 17,
    },
    "FINN-TFC_W2A2": {
        # all Quant nodes convertible to QCDQ
        "nonconvertible_quant": 0,
        "exp_qdq_nodes": 8,
        "exp_clip_nodes": 8,
    },
    "RadioML_VGG10": {
        # 23 bit bias quant not convertible to QCDQ
        "nonconvertible_quant": 1,
        "exp_qdq_nodes": 20,
        # half the Quants don't need Clip (not signed narrow)
        "exp_clip_nodes": 10,
    },
    "MobileNetv1-w4a4": {
        # 18 bit bias quant not convertible to QCDQ
        "nonconvertible_quant": 1,
        "exp_qdq_nodes": 55,
        "exp_clip_nodes": 55,
    },
}

# inherit basics for matching testcases from test util
model_details = {k: v for (k, v) in test_model_details.items() if k in qonnxtoqcdq_details.keys()}
model_details = {**model_details, **qonnxtoqcdq_details}


@pytest.mark.parametrize("test_model", model_details.keys())
def test_qonnx_to_qcdq_to_qonnx(test_model):
    test_details = model_details[test_model]
    dl_file = download_model(test_model=test_model)
    assert os.path.isfile(dl_file)
    model = ModelWrapper(dl_file)
    model = cleanup_model(model)
    input_tensor, golden_result = get_golden_in_and_output(test_model)
    # test Quant -> QCDQ conversion
    model = model.transform(QuantToQCDQ())
    assert len(model.get_nodes_by_op_type("Quant")) == test_details["nonconvertible_quant"]
    assert len(model.get_nodes_by_op_type("QuantizeLinear")) > 0
    assert len(model.get_nodes_by_op_type("DequantizeLinear")) > 0
    assert len(model.get_nodes_by_op_type("Clip")) == test_details["exp_clip_nodes"]
    model = cleanup_model(model)
    input_dict = {model.graph.input[0].name: input_tensor}
    produced_output_dict = oxe.execute_onnx(model, input_dict)
    produced_result = produced_output_dict[model.graph.output[0].name]
    assert np.isclose(golden_result, produced_result).all()
    # now test QCDQ -> Quant conversion
    model = model.transform(QCDQToQuant())
    assert len(model.get_nodes_by_op_type("QuantizeLinear")) == 0
    assert len(model.get_nodes_by_op_type("DequantizeLinear")) == 0
    assert len(model.get_nodes_by_op_type("Clip")) == 0
    model = cleanup_model(model)
    new_output_dict = oxe.execute_onnx(model, input_dict)
    roundtrip_result = new_output_dict[model.graph.output[0].name]
    assert np.isclose(golden_result, roundtrip_result).all()
    os.unlink(dl_file)
