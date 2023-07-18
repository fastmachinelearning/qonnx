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

import qonnx.core.onnx_exec as oxe
from qonnx.transformation.change_batchsize import ChangeBatchSize
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.onnx import valueinfo_to_tensor
from qonnx.util.test import download_model, test_model_details

model_details = test_model_details


@pytest.mark.parametrize("test_model", model_details.keys())
def test_change_batchsize(test_model):
    test_details = model_details[test_model]
    batch_size = 10
    old_ishape = test_details["input_shape"]
    imin, imax = test_details["input_range"]
    model = download_model(test_model=test_model, do_cleanup=True, return_modelwrapper=True)
    iname = model.graph.input[0].name
    oname = model.graph.output[0].name
    example_inp = valueinfo_to_tensor(model.get_tensor_valueinfo(iname))
    assert tuple(model.get_tensor_shape(iname)) == old_ishape
    model = model.transform(ChangeBatchSize(batch_size))
    model = model.transform(InferShapes())
    exp_ishape = (batch_size, *old_ishape[1:])
    assert tuple(model.get_tensor_shape(iname)) == exp_ishape
    new_inp = np.random.uniform(imin, imax, exp_ishape).astype(example_inp.dtype)
    ret = oxe.execute_onnx(model, {iname: new_inp})
    assert ret[oname].shape[0] == batch_size
