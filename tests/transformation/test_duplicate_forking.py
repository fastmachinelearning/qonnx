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

import qonnx.util.test as test_util
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.extract_quant_scale_zeropt import ExtractQuantScaleZeroPt
from qonnx.transformation.general import DuplicateForkingMulAdd


@pytest.fixture
def rn18_w4a4_a2q_16b_model():
    # Load the rn18_w4a4_a2q_16b model for testing
    ret = test_util.download_model("rn18_w4a4_a2q_16b", do_cleanup=True, return_modelwrapper=True)
    return ret


def test_duplicate_forking_mul_add(rn18_w4a4_a2q_16b_model):
    x, golden_y = test_util.get_golden_in_and_output("rn18_w4a4_a2q_16b")
    transformation = DuplicateForkingMulAdd()
    model = rn18_w4a4_a2q_16b_model
    # extract scales from Quant nodes to get forking Mul nodes
    model = model.transform(ExtractQuantScaleZeroPt())
    forking_mul_nodes = [x for x in model.graph.node if model.is_fork_node(x) and x.op_type == "Mul"]
    assert len(forking_mul_nodes) > 0
    model = model.transform(transformation)
    # check that no forking add or mul nodes are left
    forking_mul_nodes = [x for x in model.graph.node if model.is_fork_node(x) and x.op_type == "Mul"]
    assert len(forking_mul_nodes) == 0
    # Run the transformed model and ensure identical output
    inp_dict = {"global_in": x}
    y = execute_onnx(model, inp_dict)["global_out"]
    assert (y == golden_y).all()
