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
# * Neither the name of Xilinx nor the names of its
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

from qonnx.util.inference_cost import inference_cost
from qonnx.util.test import download_model, test_model_details

model_details_infcost = {
    "FINN-CNV_W2A2": {
        "expected_sparse": {
            "op_mac_SCALEDINT<8>_INT2": 1345500.0,
            "mem_w_INT2": 908033.0,
            "mem_o_SCALEDINT<32>": 57600.0,
            "op_mac_INT2_INT2": 35615771.0,
            "mem_o_INT32": 85002.0,
            "unsupported": "set()",
            "discount_sparsity": True,
            "total_bops": 163991084.0,
            "total_mem_w_bits": 1816066.0,
            "total_mem_o_bits": 4563264.0,
        },
        "expected_dense": {
            "op_mac_SCALEDINT<8>_INT2": 1555200.0,
            "mem_w_INT2": 1542848.0,
            "mem_o_SCALEDINT<32>": 57600.0,
            "op_mac_INT2_INT2": 57906176.0,
            "mem_o_INT32": 85002.0,
            "unsupported": "set()",
            "discount_sparsity": False,
            "total_bops": 256507904.0,
            "total_mem_w_bits": 3085696.0,
            "total_mem_o_bits": 4563264.0,
        },
    },
    "FINN-TFC_W2A2": {
        "expected_sparse": {
            "discount_sparsity": True,
            "mem_o_INT32": 202.0,
            "mem_w_INT2": 22355.0,
            "op_mac_INT2_INT2": 22355.0,
            "total_bops": 89420.0,
            "total_mem_o_bits": 6464.0,
            "total_mem_w_bits": 44710.0,
            "unsupported": "set()",
        },
        "expected_dense": {
            "discount_sparsity": False,
            "mem_o_INT32": 202.0,
            "mem_w_INT2": 59008.0,
            "op_mac_INT2_INT2": 59008.0,
            "total_bops": 236032.0,
            "total_mem_o_bits": 6464.0,
            "total_mem_w_bits": 118016.0,
            "unsupported": "set()",
        },
    },
    "RadioML_VGG10": {
        "expected_sparse": {
            "op_mac_SCALEDINT<8>_SCALEDINT<8>": 12620311.0,
            "mem_w_SCALEDINT<8>": 155617.0,
            "mem_o_SCALEDINT<32>": 130328.0,
            "unsupported": "set()",
            "discount_sparsity": True,
            "total_bops": 807699904.0,
            "total_mem_w_bits": 1244936.0,
            "total_mem_o_bits": 4170496.0,
        },
        "expected_dense": {
            "op_mac_SCALEDINT<8>_SCALEDINT<8>": 12864512.0,
            "mem_w_SCALEDINT<8>": 159104.0,
            "mem_o_SCALEDINT<32>": 130328.0,
            "unsupported": "set()",
            "discount_sparsity": False,
            "total_bops": 823328768.0,
            "total_mem_w_bits": 1272832.0,
            "total_mem_o_bits": 4170496.0,
        },
    },
}

# inherit basics for matching testcases from test util
model_details = {k: v for (k, v) in test_model_details.items() if k in model_details_infcost.keys()}
model_details = {**model_details, **model_details_infcost}


@pytest.mark.parametrize("test_model", model_details.keys())
def test_inference_cost(test_model):
    onnx_file = download_model(test_model, do_cleanup=True)
    exp_dense = model_details[test_model]["expected_dense"]
    exp_sparse = model_details[test_model]["expected_sparse"]
    ret_dense = inference_cost(onnx_file, discount_sparsity=False)
    assert ret_dense == exp_dense
    ret_sparse = inference_cost(onnx_file, discount_sparsity=True)
    assert ret_sparse == exp_sparse
