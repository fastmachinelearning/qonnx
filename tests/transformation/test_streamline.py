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

from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.streamline import ExtractAggregateScaleBias, Streamline
from qonnx.util.range_analysis import RangeInfo, range_analysis
from qonnx.util.test import download_model, get_golden_in_and_output, test_model_details

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
        scaled_int=True,
    )
    target_tensor_name = "global_out"
    model = model.transform(ExtractAggregateScaleBias(golden_dict, target_tensor_name))
    ret_dict = range_analysis(
        model,
        irange=current_details["scaledint_input_range"],
        report_mode="range",
        scaled_int=True,
    )
    ri_golden = golden_dict[target_tensor_name]
    ri_ret = ret_dict[target_tensor_name]
    assert len(ri_ret.history_scale) == 1
    assert len(ri_ret.history_bias) == 1
    assert np.isclose(ri_golden.bias, ri_ret.bias).all()
    assert np.isclose(ri_golden.scale, ri_ret.scale).all()


def test_streamline_full_network():
    model_name = "FINN-CNV_W2A2"
    current_details = {**model_details_scaledint[model_name], **test_model_details[model_name]}
    inp_t, golden_t = get_golden_in_and_output(model_name)
    model = download_model(model_name, return_modelwrapper=True, do_cleanup=True)
    current_irange = current_details["scaledint_input_range"]
    model = model.transform(Streamline(irange=current_irange))
    # check if streamlining succeeded structurally:
    # all compute-intensive ops (MatMul and Conv) must have integer inputs
    model = model.transform(InferDataTypes())
    all_compint_ops = [x for x in model.graph.node if x.op_type in ["Conv", "MatMul"]]
    for op in all_compint_ops:
        idt0 = model.get_tensor_datatype(op.input[0])
        idt1 = model.get_tensor_datatype(op.input[1])
        assert idt0.is_integer()
        assert idt1.is_integer()
    # check that streamlined model produces the ~same results
    # streamlining should remove the effect of input scaling too, so we multiply
    # our original input by (1/original scale)
    inp_dict = {model.graph.input[0].name: inp_t * (1 / current_irange.scale)}
    ret_dict = execute_onnx(model, inp_dict)
    ret_t = ret_dict[model.graph.output[0].name]
    assert np.isclose(golden_t, ret_t, atol=1e-04).all()
