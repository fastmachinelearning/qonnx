# Copyright (c) 2024 Advanced Micro Devices, Inc.
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

import os
import urllib.request

from qonnx.analysis.l0_resource_estimates import l0_resource_estimates
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup
from qonnx.util.inference_cost import inference_cost
from qonnx.util.l0_performance_estimate import l0_performance_estimate

download_url = "https://github.com/onnx/models/raw/main/validated/vision/"
download_url += "classification/resnet/model/resnet18-v1-7.onnx?download="

model_details = {
    "resnet18-v1-7": {
        "description": "Resnet18 Opset version 7.",
        "url": download_url,
        "r_bdgt1": {"LUT": 375000, "BRAM_18K": 1152, "URAM": 300, "DSP48": 2800},
        "r_bdgt2": {"LUT": 375000, "BRAM_18K": 11520, "URAM": 1200, "DSP48": 2800},  # created for the test.
        "res_limit": {
            "LUT": 0.7,
            "BRAM": 0.80,
            "BRAM36": 0.80,
            "BRAM_36K": 0.80,
            "BRAM_18K": 0.80,
            "URAM": 0.80,
            "DSP48": 0.80,
            "DSP58": 0.80,
        },
    },
}


def download_model(test_model, do_cleanup=False, return_modelwrapper=False):
    qonnx_url = model_details[test_model]["url"]
    # download test data
    dl_dir = "/tmp"
    dl_file = dl_dir + f"/{test_model}.onnx"
    ret = dl_file
    if not os.path.isfile(dl_file):
        urllib.request.urlretrieve(qonnx_url, dl_file)
    if do_cleanup:
        out_file = dl_dir + f"/{test_model}_clean.onnx"
        cleanup(dl_file, out_file=out_file, override_inpsize=1)
        ret = out_file
    if return_modelwrapper:
        ret = ModelWrapper(ret)
    return ret


def performance_check(test_model, r_bdgt, core_res_req, clock_freq):
    expected_inference = {}
    res_limit = model_details[test_model]["res_limit"]
    for i in core_res_req:
        inf_sec = ((res_limit[i] * r_bdgt[i]) / core_res_req[i]) * clock_freq
        expected_inference[i] = inf_sec
    min_infc_res = min(expected_inference, key=expected_inference.get)
    min_infc_sec = expected_inference[min_infc_res]
    ret = (min_infc_res, min_infc_sec)
    return ret


@pytest.mark.parametrize("test_model", model_details.keys())
def test_l0_performance_estimate(test_model):
    test_details = model_details[test_model]
    model = download_model(test_model, do_cleanup=True, return_modelwrapper=True)
    inf_cost = inference_cost(model, discount_sparsity=False)
    res_est_req = l0_resource_estimates(
        inf_cost,
        dsp_type="dsp48",
        bram_type="BRAM_18K",
        d_fator=0.7,
    )
    core_res_req = res_est_req["CORE"]  # processing resources.
    r_bdgt1 = test_details["r_bdgt1"]
    r_bdgt2 = test_details["r_bdgt2"]
    perf_check = performance_check(test_model, r_bdgt2, core_res_req, clock_freq=3000000)  # r_bdgt fails mem_check.
    per_est1 = l0_performance_estimate(
        r_bdgt1,
        inf_cost,
        dsp_type="dsp48",
        bram_type="BRAM_18K",
        d_fator=0.7,
    )
    per_est2 = l0_performance_estimate(
        r_bdgt2,
        inf_cost,
        dsp_type="dsp48",
        bram_type="BRAM_18K",
        d_fator=0.7,
    )
    assert per_est1 == "Memory out of budget", "Discrepancy"  # memory check fails.
    assert per_est2 == perf_check, "Discrepancy"  # memory check will pass.
