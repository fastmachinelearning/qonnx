# Copyright (c) 2025 Advanced Micro Devices, Inc.
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
import os

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fixedpt_quantize import FixedPointQuantizeParams, FixedPointQuantizeParamsFromDict
from qonnx.core.datatype import DataType
from qonnx.util.cleanup import cleanup_model
from qonnx.util.test import download_model


fixedpt_dict_details = {
    "Conv_bias_example_round": {
        "test_model": "Conv_bias_example",
        "quant_dict": {
            "Conv_0_param0": "FIXED<6,1>",
            "Conv_0_param1": "FIXED<6,1>",
            "Conv_1_param0": "FIXED<8,1>",
            "Conv_1_param1": "FIXED<8,1>",
            "Gemm_0_param0": "FIXED<12,1>",
            "Gemm_0_param1": "FIXED<12,1>"
        },
        "rounding_mode": "ROUND"
    },
    "Conv_bias_example_floor": {
        "test_model": "Conv_bias_example",
        "quant_dict": {
            "Conv_0_param0": "FIXED<6,1>",
            "Conv_0_param1": "FIXED<6,1>",
            "Conv_1_param0": "FIXED<8,1>",
            "Conv_1_param1": "FIXED<8,1>",
            "Gemm_0_param0": "FIXED<12,1>",
            "Gemm_0_param1": "FIXED<12,1>"
        },
        "rounding_mode": "FLOOR"
    },
    "FINN-CNV_W2A2_round": {
        "test_model": "FINN-CNV_W2A2",
        "quant_dict": {
            "BatchNormalization_0_param0": "FIXED<10,4>",
            "BatchNormalization_0_param1": "FIXED<10,4>",
            "BatchNormalization_0_param2": "FIXED<9,3>",
            "BatchNormalization_0_param3": "FIXED<12,6>",
            "BatchNormalization_1_param0": "FIXED<9,3>",
            "BatchNormalization_1_param1": "FIXED<10,4>",
            "BatchNormalization_1_param2": "FIXED<12,7>",
            "BatchNormalization_1_param3": "FIXED<14,13>",
            "BatchNormalization_2_param0": "FIXED<9,3>",
            "BatchNormalization_2_param1": "FIXED<10,4>",
            "BatchNormalization_2_param2": "FIXED<12,8>",
            "BatchNormalization_2_param3": "FIXED<12,11>",
            "BatchNormalization_3_param0": "FIXED<9,3>",
            "BatchNormalization_3_param1": "FIXED<10,4>",
            "BatchNormalization_3_param2": "FIXED<12,9>",
            "BatchNormalization_3_param3": "FIXED<13,12>",
            "BatchNormalization_4_param0": "FIXED<9,3>",
            "BatchNormalization_4_param1": "FIXED<10,4>",
            "BatchNormalization_4_param2": "FIXED<12,9>",
            "BatchNormalization_4_param3": "FIXED<12,11>",
            "BatchNormalization_5_param0": "FIXED<9,3>",
            "BatchNormalization_5_param1": "FIXED<10,3>",
            "BatchNormalization_5_param2": "FIXED<12,10>",
            "BatchNormalization_5_param3": "FIXED<14,13>",
            "BatchNormalization_6_param0": "FIXED<9,4>",
            "BatchNormalization_6_param1": "FIXED<10,4>",
            "BatchNormalization_6_param2": "FIXED<12,6>",
            "BatchNormalization_6_param3": "FIXED<14,11>",
            "BatchNormalization_7_param0": "FIXED<9,4>",
            "BatchNormalization_7_param1": "FIXED<10,3>",
            "BatchNormalization_7_param2": "FIXED<12,8>",
            "BatchNormalization_7_param3": "FIXED<14,13>"
        },
        "rounding_mode": "ROUND"
    },
    "FINN-CNV_W2A2_floor": {
        "test_model": "FINN-CNV_W2A2",
        "quant_dict": {
            "BatchNormalization_0_param0": "FIXED<10,4>",
            "BatchNormalization_0_param1": "FIXED<10,4>",
            "BatchNormalization_0_param2": "FIXED<9,3>",
            "BatchNormalization_0_param3": "FIXED<12,6>",
            "BatchNormalization_1_param0": "FIXED<9,3>",
            "BatchNormalization_1_param1": "FIXED<10,4>",
            "BatchNormalization_1_param2": "FIXED<12,7>",
            "BatchNormalization_1_param3": "FIXED<14,13>",
            "BatchNormalization_2_param0": "FIXED<9,3>",
            "BatchNormalization_2_param1": "FIXED<10,4>",
            "BatchNormalization_2_param2": "FIXED<12,8>",
            "BatchNormalization_2_param3": "FIXED<12,11>",
            "BatchNormalization_3_param0": "FIXED<9,3>",
            "BatchNormalization_3_param1": "FIXED<10,4>",
            "BatchNormalization_3_param2": "FIXED<12,9>",
            "BatchNormalization_3_param3": "FIXED<13,12>",
            "BatchNormalization_4_param0": "FIXED<9,3>",
            "BatchNormalization_4_param1": "FIXED<10,4>",
            "BatchNormalization_4_param2": "FIXED<12,9>",
            "BatchNormalization_4_param3": "FIXED<12,11>",
            "BatchNormalization_5_param0": "FIXED<9,3>",
            "BatchNormalization_5_param1": "FIXED<10,3>",
            "BatchNormalization_5_param2": "FIXED<12,10>",
            "BatchNormalization_5_param3": "FIXED<14,13>",
            "BatchNormalization_6_param0": "FIXED<9,4>",
            "BatchNormalization_6_param1": "FIXED<10,4>",
            "BatchNormalization_6_param2": "FIXED<12,6>",
            "BatchNormalization_6_param3": "FIXED<14,11>",
            "BatchNormalization_7_param0": "FIXED<9,4>",
            "BatchNormalization_7_param1": "FIXED<10,3>",
            "BatchNormalization_7_param2": "FIXED<12,8>",
            "BatchNormalization_7_param3": "FIXED<14,13>"
        },
        "rounding_mode": "FLOOR"
    },
    "MobileNetv1-w4a4_round": {
        "test_model": "MobileNetv1-w4a4",
        "quant_dict": {
            "BatchNormalization_0_param0": "FIXED<7,2>",
            "BatchNormalization_0_param1": "FIXED<6,2>",
            "BatchNormalization_0_param2": "FIXED<3,1>",
            "BatchNormalization_0_param3": "FIXED<12,5>",
            "BatchNormalization_1_param0": "FIXED<3,2>",
            "BatchNormalization_1_param1": "FIXED<11,2>",
            "BatchNormalization_1_param2": "FIXED<4,2>",
            "BatchNormalization_1_param3": "FIXED<8,1>",
            "BatchNormalization_2_param0": "FIXED<8,1>",
            "BatchNormalization_2_param1": "FIXED<7,2>",
            "BatchNormalization_2_param2": "FIXED<8,3>",
            "BatchNormalization_2_param3": "FIXED<11,1>",
            "BatchNormalization_3_param0": "FIXED<9,2>",
            "BatchNormalization_3_param1": "FIXED<11,2>",
            "BatchNormalization_3_param2": "FIXED<7,2>",
            "BatchNormalization_3_param3": "FIXED<8,1>",
            "BatchNormalization_4_param0": "FIXED<3,1>",
            "BatchNormalization_4_param1": "FIXED<7,2>",
            "BatchNormalization_4_param2": "FIXED<10,2>",
            "BatchNormalization_4_param3": "FIXED<8,2>",
            "BatchNormalization_5_param0": "FIXED<3,1>",
            "BatchNormalization_5_param1": "FIXED<8,2>",
            "BatchNormalization_5_param2": "FIXED<6,2>",
            "BatchNormalization_5_param3": "FIXED<7,1>",
            "BatchNormalization_6_param0": "FIXED<3,1>",
            "BatchNormalization_6_param1": "FIXED<5,1>",
            "BatchNormalization_6_param2": "FIXED<10,3>",
            "BatchNormalization_6_param3": "FIXED<5,1>",
            "BatchNormalization_7_param0": "FIXED<4,2>",
            "BatchNormalization_7_param1": "FIXED<4,2>",
            "BatchNormalization_7_param2": "FIXED<5,2>",
            "BatchNormalization_7_param3": "FIXED<11,1>",
            "BatchNormalization_8_param0": "FIXED<3,1>",
            "BatchNormalization_8_param1": "FIXED<4,2>",
            "BatchNormalization_8_param2": "FIXED<6,3>",
            "BatchNormalization_8_param3": "FIXED<11,1>",
            "BatchNormalization_9_param0": "FIXED<7,1>",
            "BatchNormalization_9_param1": "FIXED<10,2>",
            "BatchNormalization_9_param2": "FIXED<4,2>",
            "BatchNormalization_9_param3": "FIXED<10,1>",
            "BatchNormalization_10_param0": "FIXED<8,1>",
            "BatchNormalization_10_param1": "FIXED<6,1>",
            "BatchNormalization_10_param2": "FIXED<13,3>",
            "BatchNormalization_10_param3": "FIXED<11,1>",
            "BatchNormalization_11_param0": "FIXED<8,2>",
            "BatchNormalization_11_param1": "FIXED<6,2>",
            "BatchNormalization_11_param2": "FIXED<5,1>",
            "BatchNormalization_11_param3": "FIXED<11,1>",
            "BatchNormalization_12_param0": "FIXED<3,1>",
            "BatchNormalization_12_param1": "FIXED<10,1>",
            "BatchNormalization_12_param2": "FIXED<10,3>",
            "BatchNormalization_12_param3": "FIXED<10,1>",
            "BatchNormalization_13_param0": "FIXED<9,2>",
            "BatchNormalization_13_param1": "FIXED<7,2>",
            "BatchNormalization_13_param2": "FIXED<5,1>",
            "BatchNormalization_13_param3": "FIXED<6,1>",
            "BatchNormalization_14_param0": "FIXED<3,1>",
            "BatchNormalization_14_param1": "FIXED<8,1>",
            "BatchNormalization_14_param2": "FIXED<8,2>",
            "BatchNormalization_14_param3": "FIXED<9,1>",
            "BatchNormalization_15_param0": "FIXED<4,1>",
            "BatchNormalization_15_param1": "FIXED<5,1>",
            "BatchNormalization_15_param2": "FIXED<6,1>",
            "BatchNormalization_15_param3": "FIXED<8,1>",
            "BatchNormalization_16_param0": "FIXED<5,1>",
            "BatchNormalization_16_param1": "FIXED<3,1>",
            "BatchNormalization_16_param2": "FIXED<9,2>",
            "BatchNormalization_16_param3": "FIXED<6,1>",
            "BatchNormalization_17_param0": "FIXED<6,1>",
            "BatchNormalization_17_param1": "FIXED<11,2>",
            "BatchNormalization_17_param2": "FIXED<9,1>",
            "BatchNormalization_17_param3": "FIXED<8,1>",
            "BatchNormalization_18_param0": "FIXED<5,1>",
            "BatchNormalization_18_param1": "FIXED<11,1>",
            "BatchNormalization_18_param2": "FIXED<4,3>",
            "BatchNormalization_18_param3": "FIXED<5,1>",
            "BatchNormalization_19_param0": "FIXED<4,1>",
            "BatchNormalization_19_param1": "FIXED<6,2>",
            "BatchNormalization_19_param2": "FIXED<10,1>",
            "BatchNormalization_19_param3": "FIXED<3,1>",
            "BatchNormalization_20_param0": "FIXED<7,1>",
            "BatchNormalization_20_param1": "FIXED<8,1>",
            "BatchNormalization_20_param2": "FIXED<6,2>",
            "BatchNormalization_20_param3": "FIXED<6,1>",
            "BatchNormalization_21_param0": "FIXED<4,1>",
            "BatchNormalization_21_param1": "FIXED<8,2>",
            "BatchNormalization_21_param2": "FIXED<12,2>",
            "BatchNormalization_21_param3": "FIXED<6,1>",
            "BatchNormalization_22_param0": "FIXED<6,1>",
            "BatchNormalization_22_param1": "FIXED<11,1>",
            "BatchNormalization_22_param2": "FIXED<13,3>",
            "BatchNormalization_22_param3": "FIXED<11,1>",
            "BatchNormalization_23_param0": "FIXED<6,1>",
            "BatchNormalization_23_param1": "FIXED<10,2>",
            "BatchNormalization_23_param2": "FIXED<10,1>",
            "BatchNormalization_23_param3": "FIXED<10,1>",
            "BatchNormalization_24_param0": "FIXED<8,1>",
            "BatchNormalization_24_param1": "FIXED<8,1>",
            "BatchNormalization_24_param2": "FIXED<11,2>",
            "BatchNormalization_24_param3": "FIXED<7,1>",
            "BatchNormalization_25_param0": "FIXED<3,2>",
            "BatchNormalization_25_param1": "FIXED<7,2>",
            "BatchNormalization_25_param2": "FIXED<7,1>",
            "BatchNormalization_25_param3": "FIXED<10,1>",
            "BatchNormalization_26_param0": "FIXED<10,3>",
            "BatchNormalization_26_param1": "FIXED<5,2>",
            "BatchNormalization_26_param2": "FIXED<4,2>",
            "BatchNormalization_26_param3": "FIXED<11,1>"
        },
        "rounding_mode": "ROUND"
    },
    "MobileNetv1-w4a4_floor": {
        "test_model": "MobileNetv1-w4a4",
        "quant_dict": {
            "BatchNormalization_0_param0": "FIXED<7,2>",
            "BatchNormalization_0_param1": "FIXED<6,2>",
            "BatchNormalization_0_param2": "FIXED<3,1>",
            "BatchNormalization_0_param3": "FIXED<12,5>",
            "BatchNormalization_1_param0": "FIXED<3,2>",
            "BatchNormalization_1_param1": "FIXED<11,2>",
            "BatchNormalization_1_param2": "FIXED<4,2>",
            "BatchNormalization_1_param3": "FIXED<8,1>",
            "BatchNormalization_2_param0": "FIXED<8,1>",
            "BatchNormalization_2_param1": "FIXED<7,2>",
            "BatchNormalization_2_param2": "FIXED<8,3>",
            "BatchNormalization_2_param3": "FIXED<11,1>",
            "BatchNormalization_3_param0": "FIXED<9,2>",
            "BatchNormalization_3_param1": "FIXED<11,2>",
            "BatchNormalization_3_param2": "FIXED<7,2>",
            "BatchNormalization_3_param3": "FIXED<8,1>",
            "BatchNormalization_4_param0": "FIXED<3,1>",
            "BatchNormalization_4_param1": "FIXED<7,2>",
            "BatchNormalization_4_param2": "FIXED<10,2>",
            "BatchNormalization_4_param3": "FIXED<8,2>",
            "BatchNormalization_5_param0": "FIXED<3,1>",
            "BatchNormalization_5_param1": "FIXED<8,2>",
            "BatchNormalization_5_param2": "FIXED<6,2>",
            "BatchNormalization_5_param3": "FIXED<7,1>",
            "BatchNormalization_6_param0": "FIXED<3,1>",
            "BatchNormalization_6_param1": "FIXED<5,1>",
            "BatchNormalization_6_param2": "FIXED<10,3>",
            "BatchNormalization_6_param3": "FIXED<5,1>",
            "BatchNormalization_7_param0": "FIXED<4,2>",
            "BatchNormalization_7_param1": "FIXED<4,2>",
            "BatchNormalization_7_param2": "FIXED<5,2>",
            "BatchNormalization_7_param3": "FIXED<11,1>",
            "BatchNormalization_8_param0": "FIXED<3,1>",
            "BatchNormalization_8_param1": "FIXED<4,2>",
            "BatchNormalization_8_param2": "FIXED<6,3>",
            "BatchNormalization_8_param3": "FIXED<11,1>",
            "BatchNormalization_9_param0": "FIXED<7,1>",
            "BatchNormalization_9_param1": "FIXED<10,2>",
            "BatchNormalization_9_param2": "FIXED<4,2>",
            "BatchNormalization_9_param3": "FIXED<10,1>",
            "BatchNormalization_10_param0": "FIXED<8,1>",
            "BatchNormalization_10_param1": "FIXED<6,1>",
            "BatchNormalization_10_param2": "FIXED<13,3>",
            "BatchNormalization_10_param3": "FIXED<11,1>",
            "BatchNormalization_11_param0": "FIXED<8,2>",
            "BatchNormalization_11_param1": "FIXED<6,2>",
            "BatchNormalization_11_param2": "FIXED<5,1>",
            "BatchNormalization_11_param3": "FIXED<11,1>",
            "BatchNormalization_12_param0": "FIXED<3,1>",
            "BatchNormalization_12_param1": "FIXED<10,1>",
            "BatchNormalization_12_param2": "FIXED<10,3>",
            "BatchNormalization_12_param3": "FIXED<10,1>",
            "BatchNormalization_13_param0": "FIXED<9,2>",
            "BatchNormalization_13_param1": "FIXED<7,2>",
            "BatchNormalization_13_param2": "FIXED<5,1>",
            "BatchNormalization_13_param3": "FIXED<6,1>",
            "BatchNormalization_14_param0": "FIXED<3,1>",
            "BatchNormalization_14_param1": "FIXED<8,1>",
            "BatchNormalization_14_param2": "FIXED<8,2>",
            "BatchNormalization_14_param3": "FIXED<9,1>",
            "BatchNormalization_15_param0": "FIXED<4,1>",
            "BatchNormalization_15_param1": "FIXED<5,1>",
            "BatchNormalization_15_param2": "FIXED<6,1>",
            "BatchNormalization_15_param3": "FIXED<8,1>",
            "BatchNormalization_16_param0": "FIXED<5,1>",
            "BatchNormalization_16_param1": "FIXED<3,1>",
            "BatchNormalization_16_param2": "FIXED<9,2>",
            "BatchNormalization_16_param3": "FIXED<6,1>",
            "BatchNormalization_17_param0": "FIXED<6,1>",
            "BatchNormalization_17_param1": "FIXED<11,2>",
            "BatchNormalization_17_param2": "FIXED<9,1>",
            "BatchNormalization_17_param3": "FIXED<8,1>",
            "BatchNormalization_18_param0": "FIXED<5,1>",
            "BatchNormalization_18_param1": "FIXED<11,1>",
            "BatchNormalization_18_param2": "FIXED<4,3>",
            "BatchNormalization_18_param3": "FIXED<5,1>",
            "BatchNormalization_19_param0": "FIXED<4,1>",
            "BatchNormalization_19_param1": "FIXED<6,2>",
            "BatchNormalization_19_param2": "FIXED<10,1>",
            "BatchNormalization_19_param3": "FIXED<3,1>",
            "BatchNormalization_20_param0": "FIXED<7,1>",
            "BatchNormalization_20_param1": "FIXED<8,1>",
            "BatchNormalization_20_param2": "FIXED<6,2>",
            "BatchNormalization_20_param3": "FIXED<6,1>",
            "BatchNormalization_21_param0": "FIXED<4,1>",
            "BatchNormalization_21_param1": "FIXED<8,2>",
            "BatchNormalization_21_param2": "FIXED<12,2>",
            "BatchNormalization_21_param3": "FIXED<6,1>",
            "BatchNormalization_22_param0": "FIXED<6,1>",
            "BatchNormalization_22_param1": "FIXED<11,1>",
            "BatchNormalization_22_param2": "FIXED<13,3>",
            "BatchNormalization_22_param3": "FIXED<11,1>",
            "BatchNormalization_23_param0": "FIXED<6,1>",
            "BatchNormalization_23_param1": "FIXED<10,2>",
            "BatchNormalization_23_param2": "FIXED<10,1>",
            "BatchNormalization_23_param3": "FIXED<10,1>",
            "BatchNormalization_24_param0": "FIXED<8,1>",
            "BatchNormalization_24_param1": "FIXED<8,1>",
            "BatchNormalization_24_param2": "FIXED<11,2>",
            "BatchNormalization_24_param3": "FIXED<7,1>",
            "BatchNormalization_25_param0": "FIXED<3,2>",
            "BatchNormalization_25_param1": "FIXED<7,2>",
            "BatchNormalization_25_param2": "FIXED<7,1>",
            "BatchNormalization_25_param3": "FIXED<10,1>",
            "BatchNormalization_26_param0": "FIXED<10,3>",
            "BatchNormalization_26_param1": "FIXED<5,2>",
            "BatchNormalization_26_param2": "FIXED<4,2>",
            "BatchNormalization_26_param3": "FIXED<11,1>"
        },
        "rounding_mode": "FLOOR"
    }
}


@pytest.mark.parametrize("test_case", fixedpt_dict_details.keys())
def test_fixedpt_quantize_from_dict(test_case):
    test_details = fixedpt_dict_details[test_case]
    dl_file = download_model(test_model=test_details["test_model"])
    assert os.path.isfile(dl_file)
    model = ModelWrapper(dl_file)
    model = cleanup_model(model)
    # test Fixedpt conversion
    fxp_transform = FixedPointQuantizeParamsFromDict(test_details["quant_dict"], rounding_mode=test_details["rounding_mode"])
    model = model.transform(fxp_transform)
    model = cleanup_model(model)

    for tname in test_details["quant_dict"].keys():
        tdtype = DataType[test_details["quant_dict"][tname]]
        tdata = model.get_initializer(tname)

        # Check if the tensor exists in the graph
        assert tdata is not None

        # Check if all values of the tensor are allowed with the target datatype
        assert tdtype.allowed(tdata).all()

        # Check if the maximum error is within the LSB bound of the datatype
        allowed_max_error = tdtype.scale_factor()
        # The bound reduces by a factor of 2 if the mode is "ROUND"
        if test_details["rounding_mode"] == "ROUND":
            allowed_max_error /= 2
        assert fxp_transform.max_err[tname] <= allowed_max_error

    os.unlink(dl_file)

fixedpt_details = {
    "FINN-CNV_W2A2_round_0": {
        "test_model": "FINN-CNV_W2A2",
        "dtype": "FIXED<8,3>",
        "rounding_mode": "ROUND",
        "quant_tensors": [
            "Mul_0_param0",
            "Mul_1_param0",
            "Add_0_param0"
        ]
    },
    "FINN-CNV_W2A2_floor_0": {
        "test_model": "FINN-CNV_W2A2",
        "dtype": "FIXED<8,3>",
        "rounding_mode": "FLOOR",
        "quant_tensors": [
            "Mul_0_param0",
            "Mul_1_param0",
            "Add_0_param0"
        ]
    },
    "FINN-CNV_W2A2_round_1": {
        "test_model": "FINN-CNV_W2A2",
        "dtype": "FIXED<4,3>",
        "rounding_mode": "ROUND",
        "quant_tensors": [
            "Mul_0_param0",
            "Mul_1_param0",
            "Add_0_param0"
        ]
    },
    "FINN-CNV_W2A2_floor_1": {
        "test_model": "FINN-CNV_W2A2",
        "dtype": "FIXED<4,3>",
        "rounding_mode": "FLOOR",
        "quant_tensors": [
            "Mul_0_param0",
            "Mul_1_param0",
            "Add_0_param0"
        ]
    },
    "FINN-CNV_W2A2_round_2": {
        "test_model": "FINN-CNV_W2A2",
        "dtype": "FIXED<12,3>",
        "rounding_mode": "ROUND",
        "quant_tensors": [
            "Mul_0_param0",
            "Mul_1_param0",
            "Add_0_param0"
        ]
    },
    "FINN-CNV_W2A2_floor_2": {
        "test_model": "FINN-CNV_W2A2",
        "dtype": "FIXED<12,3>",
        "rounding_mode": "FLOOR",
        "quant_tensors": [
            "Mul_0_param0",
            "Mul_1_param0",
            "Add_0_param0"
        ]
    }
}


@pytest.mark.parametrize("test_case", fixedpt_details.keys())
def test_fixedpt_quantize(test_case):
    test_details = fixedpt_details[test_case]
    dl_file = download_model(test_model=test_details["test_model"])
    assert os.path.isfile(dl_file)
    model = ModelWrapper(dl_file)
    model = cleanup_model(model)

    tdtype = test_details["dtype"]
    fxp_transform = FixedPointQuantizeParams(tdtype, rounding_mode=test_details["rounding_mode"])
    tdtype = DataType[tdtype]
    model = model.transform(fxp_transform)
    model = cleanup_model(model)

    # Check if all the valid tensors are traversed by the transform
    assert set(test_details["quant_tensors"]) == set(fxp_transform.max_err.keys())

    allowed_max_error = tdtype.scale_factor()
    if test_details["rounding_mode"] == "ROUND":
        allowed_max_error /= 2

    for tname in test_details["quant_tensors"]:
        tdata = model.get_initializer(tname)
        assert tdata is not None

        # Check if all values of the tensor are allowed with the target datatype
        assert tdtype.allowed(tdata).all()

        # Check if the maximum error is within the LSB bound of the datatype
        assert fxp_transform.max_err[tname] <= allowed_max_error

    os.unlink(dl_file)
