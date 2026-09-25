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

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.qcdq_to_qop import QCDQToQOp
from qonnx.util.cleanup import cleanup_model
import onnxruntime as ort

model_details = {
    "model1.onnx": {
        "path": "src/qonnx/data/onnx/qop-sub-graph/model1.onnx",
    },
    "model2.onnx": {
        "path": "src/qonnx/data/onnx/qop-sub-graph/model2.onnx",
    },
}

def load_graph_session(model_file, execution_provider):
    EP_List=[]
    if 'CPU' in execution_provider:
        EP_List.append('CPUExecutionProvider')
    elif 'Zendnn' in execution_provider:
        EP_List.append('ZendnnExecutionProvider')
    elif 'Dnnl' in execution_provider:
        EP_List.append('DnnlExecutionProvider')
    sess = ort.InferenceSession(model_file , providers=EP_List)
    return sess

def get_output(model_file, engine):
    infer_sess = load_graph_session(model_file, engine)
    inputNames = []
    inputTensors = []
    batch_size = 32
    for inp in infer_sess.get_inputs():
        inputNames.append(inp.name)
        x_i = batch_size
        x_shape = [x_i if isinstance(s, str) or s==None else s for s in inp.shape]
        if not x_shape:
            x_shape.append(1)
        inputX = np.random.uniform(low=-10, high=100, size=(np.product(x_shape))).astype(np.int64)
        inputX = np.reshape(inputX, x_shape)
        inputTensors.append(inputX)
    feed_dict=dict(zip(inputNames, inputTensors))
    preds = infer_sess.run([], feed_dict)
    return preds

@pytest.mark.parametrize("test_model", model_details.keys())
def test_qcdq_to_qop(test_model):
    dl_file = model_details[test_model]["path"]
    dl_file_reference = dl_file.replace(".onnx", "_qop_ref.onnx")
    assert os.path.isfile(dl_file)
    assert os.path.isfile(dl_file_reference)
    output_reference = get_output(dl_file_reference, "Zendnn")
    model = ModelWrapper(dl_file)
    model = model.transform(QCDQToQOp())
    output_file = dl_file.replace(".onnx", "_qop.onnx")
    model.save(output_file)
    assert os.path.isfile(output_file)
    output_converted_model = get_output(output_file, "Zendnn")
    print("QCDQ to QOp Validation [", test_model, "] = ", np.isclose(output_reference, output_converted_model).all())

test_qcdq_to_qop("model1.onnx")
