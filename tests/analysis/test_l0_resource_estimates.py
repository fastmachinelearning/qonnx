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

download_url = "https://github.com/onnx/models/raw/main/validated/vision/"
download_url += "classification/resnet/model/resnet18-v1-7.onnx?download="

model_details = {
    "resnet18-v1-7": {
        "description": "Resnet18 Opset version 7.",
        "url": download_url,
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


@pytest.mark.parametrize("test_model", model_details.keys())
def test_l0_resource_estimates(test_model):
    model = download_model(test_model, do_cleanup=True, return_modelwrapper=True)
    inf_cost = inference_cost(model, discount_sparsity=False)
    infc_dict = inf_cost["total_cost"]
    estimates1 = l0_resource_estimates(
        infc_dict,
        dsp_type="DSP48",
        uram_type="URAM",
        bram_type="BRAM_18K",
        d_factor=0.7,
    )
    estimates2 = l0_resource_estimates(infc_dict)  # default values for dsp_type, bram, d_fator are None.
    estimates3 = l0_resource_estimates(
        infc_dict,
        dsp_type="DSP58",
        uram_type="URAM",
        bram_type="BRAM36",
        d_factor=0.5,
    )
    ocm_res1, core_res1 = estimates1["OCM"], estimates1["CORE"]
    ocm_res2, core_res2 = estimates2["OCM"], estimates2["CORE"]
    ocm_res3, core_res3 = estimates3["OCM"], estimates3["CORE"]
    assert infc_dict["total_mem_w_bits"] == (ocm_res1["BRAM_18K"] * 18 + ocm_res1["URAM"] * 288) * 1024, "discrepancy"
    assert infc_dict["total_mem_w_bits"] == (ocm_res2["LUT"] * 64), "discrepancy"
    assert infc_dict["total_mem_w_bits"] == (ocm_res3["BRAM36"] * 36 + ocm_res3["URAM"] * 288) * 1024, "discrepancy"
    # 1 mac fp32*fp32 requires 2 DSP48 and 700 LUTs.
    assert core_res1["LUT"] == 700 * infc_dict["op_mac_FLOAT32_FLOAT32"], "discrepancy"
    assert core_res1["DSP48"] == 2 * infc_dict["op_mac_FLOAT32_FLOAT32"], "discrepancy"
    # when dsp_type is None all processing is performed in LUTs.
    assert core_res2["LUT"] == 1.1 * 32 * 32 * infc_dict["op_mac_FLOAT32_FLOAT32"], "discrepancy"
    # 1 mac fp32*fp32 requires 1 DSP58 and no LUTs.
    assert core_res3["DSP58"] == infc_dict["op_mac_FLOAT32_FLOAT32"], "discrepancy"
    assert core_res3["LUT"] == 0.0, "discrepancy"
