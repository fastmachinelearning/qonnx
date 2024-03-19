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

from qonnx.analysis.inference_cost import aggregate_dict_keys
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup
from qonnx.util.inference_cost import inference_cost as infca

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
def test_inference_cost_breakdown(test_model):
    model = download_model(test_model, do_cleanup=True, return_modelwrapper=True)
    inf_cost = infca(model, discount_sparsity=False, cost_breakdown=True)
    t_cost = inf_cost["total_cost"]  # total cost
    op_cost = aggregate_dict_keys(inf_cost["optype_cost"])  # cost per optype
    n_cost = aggregate_dict_keys(inf_cost["node_cost"])  # cost per node.
    assert (
        t_cost["op_mac_FLOAT32_FLOAT32"] == op_cost["op_mac_FLOAT32_FLOAT32"] == n_cost["op_mac_FLOAT32_FLOAT32"]
    ), "inf discrepancy"
    assert t_cost["total_mem_w_bits"] == op_cost["total_mem_w_bits"] == n_cost["total_mem_w_bits"], "inf discrepancy"
    assert t_cost["total_mem_w_elems"] == op_cost["total_mem_w_elems"] == n_cost["total_mem_w_elems"], "inf discrepancy"
    assert t_cost["total_mem_o_bits"] == op_cost["total_mem_o_bits"] == n_cost["total_mem_o_bits"], "inf discrepancy"
    assert t_cost["total_mem_o_elems"] == op_cost["total_mem_o_elems"] == n_cost["total_mem_o_elems"], "inf discrepancy"
