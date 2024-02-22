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


from qonnx.analysis.l0_resource_estimates import l0_resource_estimates

""" Calculate the estimate amount of resources required for a model (from inference cost dict).
    The estimates will be divided into two parts:
        1) CORE: For processing
        2) OCM: On-Chip Memory
    First, a memory check is performed to verify enough memory is availble to accomodate the model on the FPGA.
    Then, for the resources required for processing (CORE), inference per second is calculated.
    Args:
        resource_budget (dict): Representing the resources available in a respective FPGA.
        inf_cost (dict): Inference cost dict.
        resource_estimates(): dsp_type (str), bram_type (str), bwidth_lower_limit (int),
        h_upper_limit (int), d_factor (float)
        clock_freq: Default 3MHZ.
    Returns:
        A dictionary containing CORE and OCM resource estimates.
    Examples:
            1) est_res_req: {'CORE': {'LUT': 1198735769600.0, 'DSP48': 3450357760.0},
                                                'OCM': {'BRAM_18K': 8798, 'URAM': 672}}

            2) resource_budget: {'LUT': 4397752190000, 'BRAM_18K': 1182, 'URAM': 0, 'DSP48': 500000}
"""
resource_map = {
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
    "enc_lut": {"BRAM": 576, "BRAM36": 576, "BRAM_36K": 576, "BRAM_18K": 288, "URAM": 4608},
}


def l0_performance_estimate(
    resource_budget,
    inf_cost,
    dsp_type=None,
    bram_type="BRAM",
    bwidth_lower_limit=8,
    bwidth_upper_limit=32,
    d_fator=1,
    clock_freq=3000000,
):
    expected_inference = {}
    res_limit, enc_lut = resource_map["res_limit"], resource_map["enc_lut"]
    est_res_req = l0_resource_estimates(inf_cost, dsp_type, bram_type, bwidth_lower_limit, bwidth_upper_limit, d_fator)
    ocm_res_req, core_res_req = est_res_req["OCM"], est_res_req["CORE"]
    luts_for_mem = (1 - res_limit["LUT"]) * resource_budget["LUT"]  # some amount of LUTs for memory requirement.

    for type, res in ocm_res_req.items():
        if type in resource_budget.keys():
            resource_tally = res_limit[type] * resource_budget[type] - res
            if resource_tally >= 0:  # do param fit on ocm.
                memory_check = True
            else:
                luts_req = enc_lut[type] * abs(resource_tally)
                resource_tally = res_limit["LUT"] * luts_for_mem - luts_req
                if resource_tally >= 0:
                    print(f"{type} out of budget, using luts")
                    memory_check = True
                    luts_for_mem = luts_for_mem - luts_req
                else:
                    luts_for_mem = 0
                    memory_check = False
                    break
        else:
            luts_req = enc_lut[type] * res
            resource_tally = res_limit["LUT"] * (luts_for_mem - luts_req)
            if resource_tally >= 0:
                print(f"{type} not available in the budget, using luts")
                luts_for_mem = luts_for_mem - luts_req
                memory_check = True
            else:
                luts_for_mem = 0
                memory_check = False
                break
    if memory_check is True:
        for i in core_res_req.keys():
            inf_sec = ((res_limit[i] * resource_budget[i]) / core_res_req[i]) * clock_freq
            expected_inference[i] = inf_sec
        min_infc_res = min(expected_inference, key=expected_inference.get)
        min_infc_sec = expected_inference[min_infc_res]
        ret = (min_infc_res, min_infc_sec)
    else:
        ret = "Memory out of budget"
    return ret
