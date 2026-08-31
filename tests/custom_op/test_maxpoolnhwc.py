# Copyright (c) 2024 Xilinx, Inc.
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

import numpy as np
import onnxruntime as rt
from onnx import TensorProto, helper

from qonnx.custom_op.general.maxpoolnhwc import compute_pool_output_dim
from qonnx.util.basic import qonnx_make_model

# fmt: off
# (ifm_dim, kernel, stride, pad, ceil_mode) -> expected output dim.
# The (9, 2, 2, 1, 1) case previously overcounted by one: the naive ceil-based
# formula produces a final sliding window that starts entirely inside the
# right/bottom padding region, which both PyTorch and the ONNX spec (see
# MaxPool's shape inference, https://github.com/onnx/onnx/pull/5741, and the
# clarified operator doc as of opset 22) exclude from the output.
test_cases = [
    (8, 2, 2, 0, 0, 4),
    (8, 2, 2, 0, 1, 4),
    (8, 2, 2, 1, 0, 5),
    (8, 2, 2, 1, 1, 5),
    (9, 2, 2, 0, 0, 4),
    (9, 2, 2, 0, 1, 5),
    (9, 2, 2, 1, 0, 5),
    (9, 2, 2, 1, 1, 5),  # previously computed 6, onnxruntime actually produces 5
]
# fmt: on


@pytest.mark.parametrize("ifm_dim,kernel,stride,pad,ceil_mode,expected", test_cases)
def test_compute_pool_output_dim(ifm_dim, kernel, stride, pad, ceil_mode, expected):
    assert compute_pool_output_dim(ifm_dim, kernel, stride, pad, ceil_mode) == expected


@pytest.mark.parametrize("ifm_dim,kernel,stride,pad,ceil_mode,expected", test_cases)
def test_compute_pool_output_dim_matches_onnxruntime(ifm_dim, kernel, stride, pad, ceil_mode, expected):
    # Cross-check compute_pool_output_dim() against onnxruntime's actual MaxPool
    # execution for the same parameters, to guard against the two silently
    # drifting apart again in the future.
    ifm_ch = 2
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ifm_ch, ifm_dim, ifm_dim])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, None)
    node = helper.make_node(
        "MaxPool",
        inputs=["inp"],
        outputs=["outp"],
        ceil_mode=ceil_mode,
        kernel_shape=[kernel, kernel],
        pads=[pad, pad, pad, pad],
        strides=[stride, stride],
    )
    graph = helper.make_graph([node], "maxpool_graph", [inp], [outp])
    model = qonnx_make_model(graph, producer_name="test_maxpoolnhwc")
    input_tensor = np.random.randn(1, ifm_ch, ifm_dim, ifm_dim).astype(np.float32)
    sess = rt.InferenceSession(model.SerializeToString())
    result = sess.run(None, {"inp": input_tensor})
    assert result[0].shape == (1, ifm_ch, expected, expected)
    assert compute_pool_output_dim(ifm_dim, kernel, stride, pad, ceil_mode) == expected
