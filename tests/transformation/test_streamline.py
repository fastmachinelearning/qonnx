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


import numpy as np
import os

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.streamline import ExtractAggregateScaleBias, Streamline
from qonnx.util.range_analysis import RangeInfo, range_analysis
from qonnx.util.test import download_model, test_model_details

model_details_scaledint = {
    "FINN-TFC_W2A2": {
        "scaledint_input_range": RangeInfo(
            shape=(1, 1, 28, 28),
            range=(np.asarray(0.0, dtype=np.float32), np.asarray(1.0, dtype=np.float32)),
            int_range=(np.asarray(0.0, dtype=np.float32), np.asarray(255.0, dtype=np.float32)),
            scale=np.asarray(1.0 / 255.0, dtype=np.float32),
            bias=np.asarray(0.0, dtype=np.float32),
            is_initializer=False,
        )
    },
    "FINN-CNV_W2A2": {
        "scaledint_input_range": RangeInfo(
            shape=(1, 3, 32, 32),
            range=(np.asarray(0.0, dtype=np.float32), np.asarray(1.0, dtype=np.float32)),
            int_range=(np.asarray(0.0, dtype=np.float32), np.asarray(255.0, dtype=np.float32)),
            scale=np.asarray(1.0 / 255.0, dtype=np.float32),
            bias=np.asarray(0.0, dtype=np.float32),
            is_initializer=False,
        )
    },
}


def test_extractaggregatescalebias():
    model_name = "FINN-TFC_W2A2"
    current_details = {**model_details_scaledint[model_name], **test_model_details[model_name]}
    model = download_model(model_name, return_modelwrapper=True, do_cleanup=True)
    golden_dict = range_analysis(
        model,
        irange=current_details["scaledint_input_range"],
        report_mode="range",
        lower_ops=True,
        do_cleanup=True,
        scaled_int=True,
        save_modified_model="scaledint-ra.onnx",
    )
    model = ModelWrapper("scaledint-ra.onnx")
    target_tensor_name = "global_out"
    model = model.transform(ExtractAggregateScaleBias(golden_dict, target_tensor_name))
    ret_dict = range_analysis(
        model,
        irange=current_details["scaledint_input_range"],
        report_mode="range",
        lower_ops=True,
        do_cleanup=True,
        scaled_int=True,
    )
    ri_golden = golden_dict[target_tensor_name]
    ri_ret = ret_dict[target_tensor_name]
    assert len(ri_ret.history_scale) == 1
    assert len(ri_ret.history_bias) == 1
    assert np.isclose(ri_golden.bias, ri_ret.bias).all()
    assert np.isclose(ri_golden.scale, ri_ret.scale).all()
    os.unlink("scaledint-ra.onnx")


def test_streamline():
    model_name = "FINN-CNV_W2A2"
    current_details = {**model_details_scaledint[model_name], **test_model_details[model_name]}
    model = download_model(model_name, return_modelwrapper=True, do_cleanup=True)
    tmpfile = "streamline-scaledint-ra.onnx"
    golden_dict = range_analysis(
        model,
        irange=current_details["scaledint_input_range"],
        report_mode="range",
        lower_ops=True,
        do_cleanup=True,
        scaled_int=True,
        save_modified_model=tmpfile,
    )
    model = ModelWrapper(tmpfile)
    model = model.transform(Streamline(golden_dict))
    model.save("dbg.onnx")
    os.unlink(tmpfile)
