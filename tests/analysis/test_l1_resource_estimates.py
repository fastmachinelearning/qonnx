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

from pkgutil import get_data

from qonnx.analysis.l1_resource_estimates import l1_resource_estimates
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.inference_cost import inference_cost

details = {
    "details": {
        "description": "Cleaned mnv1 model.",
        "resource_budget": {"LUT": 520704, "BRAM": 600, "URAM": 264, "DSP58": 1312},  # ve2802
        "res_match": {"LUT": 0, "BRAM": 0, "URAM": 0, "DSP58": 0},
        "res_limit": {"LUT": 0.70, "BRAM": 0.80, "URAM": 0.80, "DSP58": 0.80},
    }
}


@pytest.mark.parametrize("test_details", details.keys())
def test_l1_resource_estimates(test_details):
    model_path = get_data("qonnx", "data/onnx/l1_resource_estimates/network1.onnx")  # cleaned mnv1.
    wrapped_model = ModelWrapper(model_path)
    inf_cost = inference_cost(wrapped_model, discount_sparsity=False, cost_breakdown=True)
    total_cost, per_node_cost = inf_cost["total_cost"], inf_cost["node_cost"]
    r_bdgt = details[test_details]["resource_budget"]
    res_match = details[test_details]["res_match"]
    res_limit = details[test_details]["res_limit"]
    all_node_res_bdgt = l1_resource_estimates(r_bdgt, total_cost, per_node_cost)
    assert len(all_node_res_bdgt) == 28, "Discrepancy"  # Total number of layers/nodes in the network.
    for name, res in all_node_res_bdgt.items():
        res_match["LUT"] += res["LUT"]
        res_match["BRAM"] += res["BRAM"]
        res_match["URAM"] += res["URAM"]
        res_match["DSP58"] += res["DSP58"]

    assert (
        res_limit["LUT"] * r_bdgt["LUT"] >= res_match["LUT"] and res_match["LUT"] == 3100.0000
    ), "Over Budget or Discrepancy"
    assert (
        res_limit["BRAM"] * r_bdgt["BRAM"] >= res_match["BRAM"] and res_match["BRAM"] == 288.0000
    ), "Over Budget and Discrepancy"
    assert (
        res_limit["URAM"] * r_bdgt["URAM"] >= res_match["URAM"] and res_match["URAM"] == 0.0000
    ), "Over Budget or Discrepancy"
    assert (
        res_limit["DSP58"] * r_bdgt["DSP58"] >= res_match["DSP58"] and res_match["DSP58"] == 1049.5988
    ), "Over Budget or Discrepancy"
