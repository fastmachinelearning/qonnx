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

from qonnx.util.range_analysis import range_analysis
from qonnx.util.test import download_model, test_model_details

model_details_stuckchans = {
    "MobileNetv1-w4a4": {
        "stuck_chans": {
            "Conv_0_out0": [
                (0, 0.0),
                (4, 0.0),
                (6, 0.0),
                (10, 0.0),
                (13, 0.0),
                (15, 0.0),
                (16, 0.0),
                (19, 0.0),
                (26, 0.0),
                (28, 0.0),
            ],
            "BatchNormalization_0_out0": [
                (0, 0.4344967),
                (4, -0.06948418),
                (6, -0.26580426),
                (10, -0.369785),
                (13, -0.29186976),
                (15, -0.20715548),
                (16, -0.41614214),
                (19, -0.32901502),
                (26, 0.0046599484),
                (28, -0.25468823),
            ],
            "Relu_0_out0": [
                (0, 0.4344967),
                (4, 0.0),
                (6, 0.0),
                (10, 0.0),
                (13, 0.0),
                (15, 0.0),
                (16, 0.0),
                (19, 0.0),
                (26, 0.0046599484),
                (28, 0.0),
            ],
            "Quant_0_out0": [
                (0, 0.4813263),
                (4, 0.0),
                (6, 0.0),
                (10, 0.0),
                (13, 0.0),
                (15, 0.0),
                (16, 0.0),
                (19, 0.0),
                (26, 0.0),
                (28, 0.0),
            ],
            "Conv_1_out0": [
                (0, 0.0),
                (4, 0.0),
                (6, 0.0),
                (10, 0.0),
                (13, 0.0),
                (15, 0.0),
                (16, 0.0),
                (19, 0.0),
                (26, 0.0),
                (28, 0.0),
            ],
            "BatchNormalization_1_out0": [
                (0, -0.35108954),
                (4, -0.0851285),
                (6, -0.056262717),
                (10, -0.18598184),
                (13, 0.109134674),
                (15, -0.243637),
                (16, 0.39661717),
                (19, -0.2065738),
                (26, -0.10047761),
                (28, -0.4328182),
            ],
            "Relu_1_out0": [
                (0, 0.0),
                (4, 0.0),
                (6, 0.0),
                (10, 0.0),
                (13, 0.109134674),
                (15, 0.0),
                (16, 0.39661717),
                (19, 0.0),
                (26, 0.0),
                (28, 0.0),
            ],
            "Quant_1_out0": [
                (0, 0.0),
                (4, 0.0),
                (6, 0.0),
                (10, 0.0),
                (13, 0.15743902),
                (15, 0.0),
                (16, 0.47231707),
                (19, 0.0),
                (26, 0.0),
                (28, 0.0),
            ],
            "Conv_2_out0": [(42, 0.0)],
            "BatchNormalization_2_out0": [(42, -0.6043172)],
            "Relu_2_out0": [(42, 0.0)],
            "Quant_2_out0": [(42, 0.0)],
            "Conv_3_out0": [(42, 0.0)],
            "BatchNormalization_3_out0": [(42, 0.03516292)],
            "Relu_3_out0": [(42, 0.03516292)],
            "Quant_3_out0": [(42, 0.0)],
            "Conv_6_out0": [(102, 0.0)],
            "BatchNormalization_6_out0": [(102, -0.28346187)],
            "Relu_6_out0": [(102, 0.0)],
            "Quant_6_out0": [(102, 0.0)],
            "Conv_7_out0": [(102, 0.0)],
            "BatchNormalization_7_out0": [(102, -0.09163331)],
            "Relu_7_out0": [(102, 0.0)],
            "Quant_7_out0": [(102, 0.0)],
        }
    },
    "FINN-CNV_W2A2": {
        "stuck_chans": {
            "Quant_1_out0": [(5, -1.0), (10, 1.0), (26, 1.0), (30, -1.0), (34, -1.0), (54, -1.0)],
            "Quant_2_out0": [(30, 1.0), (35, 1.0), (37, -1.0), (42, 1.0), (45, -1.0), (57, -1.0)],
            "MaxPool_0_out0": [(30, 1.0), (35, 1.0), (37, -1.0), (42, 1.0), (45, -1.0), (57, -1.0)],
            "Quant_4_out0": [(40, -1.0)],
            "MaxPool_1_out0": [(40, -1.0)],
            "Quant_5_out0": [(4, 1.0), (175, 1.0), (209, -1.0)],
            "Quant_7_out0": [
                (5, -1.0),
                (50, 1.0),
                (77, -1.0),
                (95, -1.0),
                (153, 1.0),
                (186, 1.0),
                (199, 1.0),
                (209, -1.0),
                (241, 1.0),
                (329, 1.0),
                (340, 1.0),
                (465, -1.0),
                (478, -1.0),
                (510, -1.0),
            ],
            "Quant_8_out0": [(101, 0.0), (230, 0.0), (443, 0.0)],
        }
    },
}

# inherit basics for matching testcases from test util
model_details = {k: v for (k, v) in test_model_details.items() if k in model_details_stuckchans.keys()}
model_details = {**model_details, **model_details_stuckchans}


@pytest.mark.parametrize("model_name", model_details.keys())
def test_range_analysis(model_name):
    model = download_model(model_name, return_modelwrapper=True)
    irange = test_model_details[model_name]["input_range"]
    ret = range_analysis(model, do_cleanup=True, irange=irange, report_mode="stuck_channel")
    golden_stuck_channels = model_details[model_name]["stuck_chans"]
    for tname, ret_chans in ret.items():
        tg_chans = golden_stuck_channels[tname]
        for i in range(len(tg_chans)):
            tg_ind, tg_val = tg_chans[i]
            ret_ind, ret_val = ret_chans[i]
            assert tg_ind == ret_ind
            assert np.isclose(tg_val, ret_val)
