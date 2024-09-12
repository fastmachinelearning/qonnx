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

import numpy as np

from qonnx.core.datatype import DataType
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.extract_quant_scale_zeropt import ExtractQuantScaleZeroPt
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.streamline import ExtractAggregateScaleBias, Streamline
from qonnx.util.range_analysis import RangeInfo, range_analysis
from qonnx.util.test import download_model, get_random_input, test_model_details

uint8_to_unitfloat = {
    "range": (np.asarray(0.0, dtype=np.float32), np.asarray(1.0, dtype=np.float32)),
    "int_range": (np.asarray(0.0, dtype=np.float32), np.asarray(255.0, dtype=np.float32)),
    "scale": np.asarray(1.0 / 255.0, dtype=np.float32),
    "bias": np.asarray(0.0, dtype=np.float32),
    "is_initializer": False,
}

model_details_scaledint = {
    "FINN-TFC_W2A2": {"scaledint_input_range": RangeInfo(shape=(1, 1, 28, 28), **uint8_to_unitfloat)},
    "FINN-CNV_W2A2": {"scaledint_input_range": RangeInfo(shape=(1, 3, 32, 32), **uint8_to_unitfloat)},
    "MobileNetv1-w4a4": {"scaledint_input_range": RangeInfo(shape=(1, 3, 224, 224), **uint8_to_unitfloat)},
    "rn18_w4a4_a2q_16b": {
        "scaledint_input_range": RangeInfo(
            shape=(1, 3, 32, 32),
            # TODO: the A2Q networks have actually different scale and bias
            # due to input preprocessing, should be taken into account
            **uint8_to_unitfloat
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


@pytest.mark.parametrize("model_name", model_details_scaledint.keys())
def test_streamline_full_network(model_name):
    current_details = {**model_details_scaledint[model_name], **test_model_details[model_name]}
    orig_model = download_model(model_name, return_modelwrapper=True, do_cleanup=True)
    current_irange = current_details["scaledint_input_range"]
    model = orig_model.transform(Streamline(irange=current_irange))
    # check if streamlining succeeded structurally:
    # all compute-intensive ops (MatMul and Conv) must have integer inputs
    iname = model.graph.input[0].name
    oname = model.graph.output[0].name
    # check that streamlined model produces the ~same results
    inp_t = get_random_input(model_name)
    inp_dict = {iname: inp_t}
    # no-scales Quant version as reference, for easier comparison/debug
    noscales_model = orig_model.transform(ExtractQuantScaleZeroPt())
    noscales_dict = execute_onnx(noscales_model, inp_dict, return_full_exec_context=True)
    noscales_t = noscales_dict[oname]
    # streamlining should remove the effect of input scaling too, so we multiply
    # our original input by (1/original scale)
    inp_dict = {iname: inp_t * (1 / current_irange.scale)}
    ret_dict = execute_onnx(model, inp_dict, return_full_exec_context=True)
    ret_t = ret_dict[oname]
    assert np.isclose(noscales_t, ret_t, atol=1e-04).all()
    # note that we expect the input data type to change for the streamlined graph
    model.set_tensor_datatype(iname, DataType["UINT8"])
    model = model.transform(InferDataTypes())
    all_compint_ops = [x for x in model.graph.node if x.op_type in ["Conv", "MatMul"]]
    for op in all_compint_ops:
        idt0 = model.get_tensor_datatype(op.input[0])
        idt1 = model.get_tensor_datatype(op.input[1])
        assert idt0.is_integer()
        assert idt1.is_integer()
