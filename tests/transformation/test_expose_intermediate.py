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

from qonnx.transformation.expose_intermediate import ExposeIntermediateTensorsPatternList
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.util.test import download_model, test_model_details

model_details_expint = {
    "FINN-TFC_W2A2": {"n_quant_outputs": 4},
    "FINN-CNV_W2A2": {"n_quant_outputs": 9},
    "MobileNetv1-w4a4": {"n_quant_outputs": 27},
}

# inherit basics for matching testcases from test util
model_details = {k: v for (k, v) in test_model_details.items() if k in model_details_expint.keys()}
model_details = {**model_details, **model_details_expint}


@pytest.mark.parametrize("model_name", model_details.keys())
def test_expose_intermediate(model_name):
    model = download_model(model_name, do_cleanup=True, return_modelwrapper=True)
    # do folding for weights
    model = model.transform(FoldConstants(exclude_op_types=[]))
    # break out all dynamic (non-weight) quantizer outputs
    pattern_list = ["Quant"]
    model = model.transform(ExposeIntermediateTensorsPatternList(pattern_list, dynamic_only=True))
    model.save(model_name + "_dbg.onnx")
    assert len(model.graph.output) == model_details_expint[model_name]["n_quant_outputs"] + 1
