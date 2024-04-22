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


import math

from qonnx.analysis.l0_resource_estimates import core_resources
from qonnx.util.l0_performance_estimate import l0_performance_estimate

resource_map = {
    "res_limit": {"LUT": 0.7, "BRAM": 0.80, "URAM": 0.80, "DSP48": 0.80, "DSP58": 0.80},
    "bits_per_res": {"BRAM": 36864, "URAM": 294912, "LUT": 64},
}


def rounding_down(value, decimal_place):
    fact = 10**decimal_place
    return math.floor(value * fact) / fact


def memory_distribution(resource_budget, total_cost, node_cost, unused_resources):
    """This function is used for memory resource allocation to each node. In this function, we
    are trying to project either one of the following resources URAM, BRAM, LUTRAM to a particular node.
    However, for some nodes a combination of resources could be projected.
    Parameters:-
                resource_budget (dict):- Resource budget of an FPGA in a dictionary format.
                                         Example (ve2802) :-{"LUT": 520704, "BRAM": 600, "URAM": 264, "DSP58": 1312}
                total_cost (dict):- Inference cost dictionary of the full model.
                per_node_cost (dict):- Inference cost associated with each layer/node in the neural network.
                unused_resources (dict):- Dictionary in the format of resource budget to keep record of resources used,
                provided internally.
    """

    res_limit, bits_per_res = resource_map["res_limit"], resource_map["bits_per_res"]

    mem_port = (
        node_cost["total_mem_w_bits"] / total_cost["total_mem_w_bits"]
    )  # mem portion of this layer in the model (between 0 and 1).

    available_bram = (
        mem_port * (res_limit["BRAM"] * resource_budget["BRAM"]) + unused_resources["BRAM"]
    )  # Approximate Bram available for this node.
    available_uram = (
        mem_port * (res_limit["URAM"] * resource_budget["URAM"]) + unused_resources["URAM"]
    )  # Approximate Uram available for this node.
    available_lut = (
        mem_port * (res_limit["LUT"] * resource_budget["LUT"]) + unused_resources["LUT"]
    )  # Approximate LUT available for this node.

    req_bram = node_cost["total_mem_w_bits"] / bits_per_res["BRAM"]  # BRAM required for this node.
    req_uram = node_cost["total_mem_w_bits"] / bits_per_res["URAM"]  # URAM required for this node.
    req_lut = node_cost["total_mem_w_bits"] / bits_per_res["LUT"]  # LUT required for this node.

    if req_bram < 1 and available_lut > req_lut:
        per_node_res_bdgt = {"BRAM": 0, "URAM": 0, "LUT": math.ceil(req_lut)}
    else:
        if available_uram > math.ceil(req_uram) and available_bram > math.ceil(req_bram):
            uram_bit_waste = (math.ceil(req_uram) - req_uram) * bits_per_res["URAM"]
            bram_bit_waste = (math.ceil(req_bram) - req_bram) * bits_per_res["BRAM"]
            min_bit_waste = min(uram_bit_waste, bram_bit_waste)
            if min_bit_waste == bram_bit_waste:
                per_node_res_bdgt = {"BRAM": math.ceil(req_bram), "URAM": 0, "LUT": 0}
            else:
                per_node_res_bdgt = {"BRAM": 0, "URAM": math.ceil(req_uram), "LUT": 0}
        elif available_uram > math.ceil(req_uram) and available_bram < math.ceil(req_bram):
            per_node_res_bdgt = {"BRAM": 0, "URAM": math.ceil(req_uram), "LUT": 0}
        elif available_uram < math.ceil(req_uram) and available_bram > math.ceil(req_bram):
            per_node_res_bdgt = {"BRAM": math.ceil(req_bram), "URAM": 0, "LUT": 0}
        else:
            bits_in_bram, bits_in_uram = available_bram * bits_per_res["BRAM"], available_uram * bits_per_res["URAM"]
            total_available_bits = bits_in_bram + bits_in_uram
            if total_available_bits >= node_cost["total_mem_w_bits"]:
                uram_dist_factor = bits_in_uram / total_available_bits
                bram_dist_factor = 1 - uram_dist_factor
                bram_dist = bram_dist_factor * node_cost["total_mem_w_bits"] / bits_per_res["BRAM"]
                uram_dist = uram_dist_factor * node_cost["total_mem_w_bits"] / bits_per_res["URAM"]
                per_node_res_bdgt = {"BRAM": math.ceil(bram_dist), "URAM": math.ceil(uram_dist), "LUT": 0}
            else:
                per_node_res_bdgt = {"BRAM": math.ceil(available_bram), "URAM": math.ceil(available_uram), "LUT": 0}

    unused_resources = {
        "BRAM": (available_bram - per_node_res_bdgt["BRAM"]),  # -ve value indicates over budget in unused resources.
        "URAM": (available_uram - per_node_res_bdgt["URAM"]),
        "LUT": (available_lut - per_node_res_bdgt["LUT"]),
    }

    return per_node_res_bdgt, unused_resources


def l1_resource_estimates(
    resource_budget,
    total_cost,
    per_node_cost,
    bwidth_lower_limit=8,
    bwidth_upper_limit=32,
    clock_freq=300000000,
    decimal_place=4,
):
    """
    This function provides an estimate about the resources used by each layer of the neural network
    when deployed on a particular FPGA.

    Parameters:-
        resource_budget (dict):- Resource budget of an FPGA in a dictionary format.
                                 Example (ve2802) :-{"LUT": 520704, "BRAM": 600, "URAM": 264, "DSP58": 1312}
        total_cost (dict):- Inference cost dictionary of the full model.
        per_node_cost (dict):- Inference cost associated with each layer/node in the neural network.
        bwidth_lower_limit (int):- Lower cut off value to use DSPs instead of LUTs.
        bwidth_upper_limit (int):- Upper cut off value to use LUTs instead of DSPs.
        clock_freq (int):- Default at 300 million (single pump)
        decimal_place (int):- used to round upto "decimal_place" values after decimal.
    """

    # 300 MHZ
    all_node_res_bdgt = {}
    # rounding upto

    dsp_type = next((key for key in resource_budget if "DSP" in key), None)

    unused_resources = {"LUT": 0, "BRAM": 0, "URAM": 0}  # passing unused resources to the following layers.

    resource_performance = l0_performance_estimate(
        resource_budget,  # Inference per sec (slowdown factor * clock cycle)
        total_cost,
        bwidth_lower_limit,
        bwidth_upper_limit,
        clock_freq,
    )

    slowdown_factor = resource_performance[1] / clock_freq

    for node_name, node_cost in per_node_cost.items():
        per_node_res_bdgt, unused_resources = memory_distribution(resource_budget, total_cost, node_cost, unused_resources)

        core_res = core_resources(node_cost, dsp_type, bwidth_lower_limit, bwidth_upper_limit)

        for i, j in core_res.items():
            core_amount = slowdown_factor * j
            if i in per_node_res_bdgt.keys():
                per_node_res_bdgt[i] += rounding_down(core_amount, decimal_place)
            else:
                per_node_res_bdgt[i] = rounding_down(core_amount, decimal_place)
        all_node_res_bdgt[node_name] = per_node_res_bdgt
    return all_node_res_bdgt
