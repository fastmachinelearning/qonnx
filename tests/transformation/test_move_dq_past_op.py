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
from qonnx.transformation.move_dq_past_op import MoveDequantizePastOp
from qonnx.transformation.qonnx_to_qcdq import QuantToQCDQ
from qonnx.util.test import download_model, get_golden_in_and_output, test_model_details

pushdq_details = {
    "FINN-TFC_W2A2": {},
}

# inherit basics for matching testcases from test util
model_details = {k: v for (k, v) in test_model_details.items() if k in pushdq_details.keys()}
model_details = {**model_details, **pushdq_details}


@pytest.mark.parametrize("test_model", model_details.keys())
def test_move_dq_past_matmul(test_model):
    # test_details = model_details[test_model]
    model = download_model(test_model=test_model, return_modelwrapper=True, do_cleanup=True)
    input_tensor, golden_result = get_golden_in_and_output(test_model)
    model = model.transform(QuantToQCDQ())
    model = model.transform(MoveDequantizePastOp())
    model.save("dbg.onnx")
    iname = model.graph.input[0].name
    oname = model.graph.output[0].name
    idict = {iname: input_tensor}
    ret = oxe.execute_onnx(model, idict)
    assert np.isclose(ret[oname], golden_result).all()
