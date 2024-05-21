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


from pkgutil import get_data

import qonnx.util.inference_cost as infc
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model


def test_matmul_mac_cost():
    raw_model = get_data("qonnx", "data/onnx/matmul_update/sdp.onnx")
    model = ModelWrapper(raw_model)
    cleaned_model = cleanup_model(model)
    # Two Matmul layers with shape (i_shape, w_shape, o_shape),
    # L1: ([4, 64, 32], [4, 32, 64], [4, 64, 64]) and L2: ([4, 64, 64], [4, 64, 32], [4, 64, 32])
    inf_cost_dict = infc.inference_cost(cleaned_model, discount_sparsity=False)["total_cost"]
    mac_cost = inf_cost_dict["op_mac_FLOAT32_FLOAT32"]  # Expected mac cost 4*32*64*64 + 4*64*64*32 = 1048576
    assert mac_cost == 1048576.0, "Error: discrepancy in mac cost."
