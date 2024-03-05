# Copyright (c) 2024 Advanced Micro Devices, Inc.
# # All rights reserved.
# #
# # Redistribution and use in source and binary forms, with or without
# # modification, are permitted provided that the following conditions are met:
# #
# # * Redistributions of source code must retain the above copyright notice, this
# #   list of conditions and the following disclaimer.
# #
# # * Redistributions in binary form must reproduce the above copyright notice,
# #   this list of conditions and the following disclaimer in the documentation
# #   and/or other materials provided with the distribution.
# #
# # * Neither the name of qonnx nor the names of its
# #   contributors may be used to endorse or promote products derived from
# #   this software without specific prior written permission.
# #
# # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from qonnx.core.datatype import DataType

"""DSP Type: a) None:
                    For Fixed Points and floating point
                        1) When dsp_type is None. All operations will be processed using LUTs.
                        2) LUTs are calculated using: 1.1*b_width1*b_width2
                        2) Example:
                            a) op_mac_Int4_Int2: 1.1*4*2 = 8.8 LUTs.
                            b) op_mac_Int8_INT8: 1.1*8*8 = 70.4 LUTs.
                            c) op_mac_Int8_FLOAT16: 1.1*8*16 = 140.8 LUTs
                            d) op_mac_FLOAT16_FLOAT16: 1.1*16*16 = 281.6 LUTs.

             b) DSP48:
                    For Fixed Points
                        1) Everything less than 4 will be promoted to 4. For ex: INT2 will use the same resources as INT4.
                        2) INT4: One dsp48 + 200 LUTs can accomodate 4 (4*4) bit mac.
                                So, no of dsp's from mac's can be calculated as (0.25).mac_count + (200*0.5)*mac_count LUTs.
                        3) Everything between 5 and 8 will be promoted to 8, Ex: INT6 will use the same resources as INT8.
                        4) INT88: One dsp48 + 200 LUTs can accomodate 2 (8*8) bit mac. So,
                                 no of dsp's from mac's can be calculated as (0.5).mac_count + (200*0.25)*mac_count LUTs.
                    For Floating Points
                        1) FLOAT32: 2 dsp + 700 LUT can accomodate 1 mac count.
                        2) FLOAT16: 1 dsp + 400 LUT can accomodate 1 mac count.
            c) DSP58:
                    For Fixed Points
                        1) INT8: One dsp58 can accomodate 3 (8*8) bit mac.
                           So, no of dsp's from mac's can be calculated as (0.33)*mac_count.
                        2) INT4: One dsp58 can accomodate 4 (4*4) bit mac.
                           So, no of dsp's from mac's can be calculated as (0.25)*mac_count.
                        3) INT16: 1 mac count requires 1 dsp.
                    For Floating Points
                        1) FLOAT32:  1 mac count requires 1 dsp.
                        2) FLOAT16: 1 mac count requires 1 dsp.
    Mapping strategy for On-Chip Memory (bits_per_res):
                        a) 1 "BRAM", 1 "BRAM36" and 1 "BRAM_36K" can accomodate 36*1024 = 36864 bits.
                        b) 1 "BRAM_18K" can accomodate 18*1024 = 18432 bits.
                        c) 1 "URAM" can accomodate 288*1024 = 294912 bits.
                        d) 1 LUT can accomodate 64 bits.
"""
resource_table = {
    "FLOAT32": {"NONE": (0, 1100), "DSP48": (2, 700), "DSP58": (1, 0)},
    "FLOAT16": {"NONE": (0, 1100), "DSP48": (1, 400), "DSP58": (1, 0)},
    "INT32": {"NONE": (0, 1100), "DSP48": (1, 0), "DSP58": (1, 0)},
    "INT16": {"NONE": (0, 282), "DSP48": (1, 0), "DSP58": (1, 0)},
    "INT8": {"NONE": (0, 71), "DSP48": (0.5, 100), "DSP58": (0.33, 0)},
    "INT4": {"NONE": (0, 18), "DSP48": (0.25, 50), "DSP58": (0.25, 0)},
}

bits_per_res = {"BRAM": 36864, "BRAM36": 36864, "BRAM_36K": 36864, "BRAM_18K": 18432, "URAM": 294912, "LUT": 64}


def ocm_resources(num_mem_bits, uram_type, bram_type, d_factor):
    """Provides an estimate about the number of urams and brams required for the
       on-chip memory depending upon the distribution factor.
    Args:
        num_mem_bits (int): Number of memory bits.
        d_factor (float): Distribution factor between 0 and 1.
                         To distribute memory between BRAM and URAM.
        bram_type (str): can be BRAM, BRAM36, BRAM_36K,BRAM_18K.
    Returns:
        A dictionary for ocm resources containing memory requirements for luts, brams and urams
    """
    if d_factor is None:
        luts_req = num_mem_bits / bits_per_res["LUT"]  # neither bram nor uram.
        ocm_res = {"LUT": luts_req}
    elif d_factor == 1:  # everything in uram.
        uram_req = num_mem_bits / bits_per_res[uram_type]  # URAM: 288kbit/URAM
        ocm_res = {uram_type: uram_req}
    elif d_factor == 0:  # everything in bram (BRAM_18K/BRAM/BRAM36/BRAM_36K)
        bram_req = num_mem_bits / bits_per_res[bram_type]
        ocm_res = {bram_type: bram_req}
    else:  # both bram and uram.
        uram_por, bram_por = d_factor, 1 - d_factor
        bram_req = (bram_por * num_mem_bits) / bits_per_res[bram_type]
        uram_req = (uram_por * num_mem_bits) / bits_per_res[uram_type]
        ocm_res = {bram_type: bram_req, uram_type: uram_req}
    return ocm_res


def promoting_datatype(dtype, b_width):
    """Datatype promoting criterion. Only used when DSPs are used for processing.
    Args:
        dtype (str): conatining "INT" or "FLOAT".
        b_width (int): precision of the respective datatype.
    Returns:
        Returns promoted datatype and precision value."""

    if "INT" in dtype:
        promoted_dtype = "INT"
        if b_width <= 4:
            promoted_bwidth = 4
        elif 4 < b_width <= 8:
            promoted_bwidth = 8
        elif 8 < b_width <= 16:
            promoted_bwidth = 16
        else:
            promoted_bwidth = 32
    elif "FLOAT" in dtype:
        promoted_dtype = "FLOAT"
        if b_width <= 16:
            promoted_bwidth = 16
        else:
            promoted_bwidth = 32
    else:
        raise Exception("Unsupported data type")

    return promoted_dtype, promoted_bwidth


def dtype_casting(dtype1, dtype2, b_width1, b_width2):
    """Implementing datatype promotion."""

    promoted_dtype1, promoted_bwidth1 = promoting_datatype(dtype1, b_width1)  # either INT or FLOAT
    promoted_dtype2, promoted_bwidth2 = promoting_datatype(dtype2, b_width2)

    if promoted_dtype1 == promoted_dtype2:  # same datatype
        if promoted_bwidth1 == promoted_bwidth2:  # same precision.
            dtype = promoted_dtype1 + str(promoted_bwidth1)  # can also use dtype_2 + new_bwidth2
        else:  # different precision.
            if promoted_bwidth1 >= promoted_bwidth2:
                dtype = promoted_dtype1 + str(promoted_bwidth1)
            else:
                dtype = promoted_dtype2 + str(promoted_bwidth2)
    else:  # dtype_1 != dtype_2 (Different datatype and same/different precision)
        if promoted_dtype1 == "FLOAT":  # with different datatypes, using float and it's respective precision.
            dtype = promoted_dtype1 + str(promoted_bwidth1)
        else:
            dtype = promoted_dtype2 + str(promoted_bwidth2)

    return dtype


def core_resources(inf_cost, dsp_type, bwidth_lower_limit, bwidth_upper_limit):
    """Provide estimate resources required for the processing ("CORE"), assuming maximum unfolding.
    Args:
        inf_cost (dict): Inference cost dict.
        dsp_type (str): None OR "DSP48" OR "DSP58". Default to None.
        bwidth_lower_limit (int): Default to 8. It indicates bit values less than 8 will be processed using LUTs.
        bwidth_upper_limit (int): Default to 32. It indicates bit values less than 32 will be processed using LUTs.
    Returns:
        A dictionary containing CORE resource estimates."""

    dsp_res_mac = 0
    lut_res_mac = 0
    for i in inf_cost.keys():
        if "op_mac" in i:
            mac_count = inf_cost[i]
            detail_list = i.split("_")
            dtype1, dtype2 = detail_list[-1], detail_list[-2]
            b_width1, b_width2 = DataType[dtype1].bitwidth(), DataType[dtype2].bitwidth()
            if dsp_type is None:  # Computing everything in LUTs.
                lut_res_mac += 1.1 * b_width1 * b_width2 * mac_count
                dsp_comp = "DSP"  # default name for DSP and dsp_res_mac = 0
            else:  # dsp_type == "DSP48" or dsp_type == "DSP58"
                if (b_width1 < bwidth_lower_limit or b_width2 < bwidth_lower_limit) or (
                    b_width1 > bwidth_upper_limit or b_width2 > bwidth_upper_limit
                ):  # Computing everything in LUTs.
                    lut_res_mac += 1.1 * b_width1 * b_width2 * mac_count  # dsp_res_mac = 0
                else:
                    casted_dtype = dtype_casting(dtype1, dtype2, b_width1, b_width2)
                    casted_bwidth = DataType[casted_dtype].bitwidth()
                    if casted_bwidth > bwidth_upper_limit:  # Computing everything in LUTs.
                        lut_res_mac += (
                            1.1 * b_width1 * b_width2 * mac_count
                        )  # original bwidth values are used, since dsp_res_mac = 0.
                    else:
                        dsp_res_mac += (
                            resource_table[casted_dtype][dsp_type][0] * mac_count
                        )  # at index zero, we expect to have dsp factor.
                        lut_res_mac += (
                            resource_table[casted_dtype][dsp_type][1] * mac_count
                        )  # at index one, we expect to have lut factor.
                dsp_comp = dsp_type  # assigning name as per dsp type.
        else:
            continue

    core_res = {"LUT": lut_res_mac, dsp_comp: dsp_res_mac}

    return core_res


def l0_resource_estimates(
    inf_cost, dsp_type=None, uram_type=None, bram_type=None, bwidth_lower_limit=8, bwidth_upper_limit=32, d_factor=None
):
    """Provide estimate resources required for the processing ("CORE") and memory ("OCM"), assuming maximum unfolding.
    Args:
        inf_cost (dict): Inference cost dict.
        dsp_type (str): None OR "DSP48" OR "DSP58". Default to None.
        bram_type (str): Default to "BRAM". It can be BRAM, BRAM36, BRAM_36K, BRAM_18K.
        bwidth_lower_limit (int): Default to 8. It indicates bit values less than 8 will be processed using LUTs.
        bwidth_upper_limit (int): Default to 32. It indicates bit values less than 32 will be processed using LUTs.
        d_factor (float): Default to 1. It can have values between 0 and 1.
    Returns:
        A dictionary containing CORE and OCM resource estimates."""

    core_res = core_resources(inf_cost, dsp_type, bwidth_lower_limit, bwidth_upper_limit)

    num_mem_bits = inf_cost["total_mem_w_bits"]
    ocm_res = ocm_resources(num_mem_bits, uram_type, bram_type, d_factor)

    est_res_req = {"CORE": core_res, "OCM": ocm_res}

    return est_res_req
