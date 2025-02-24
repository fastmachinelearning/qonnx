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

from qonnx.custom_op.general.floatquant import compute_max_val, float_quantize


def test_compute_max_val():
    # reference max normal values from OCP MX 1.0 standard
    assert compute_max_val(2, 3) == 7.5  # FP6 E2M3
    assert compute_max_val(3, 2) == 28.0  # FP6 E3M2
    assert compute_max_val(2, 1) == 6.0  # FP4 E2M1


def test_float_quantize():
    zero_tensor = np.zeros((2, 2))
    unit_scale = np.asarray([1.0], dtype=np.float32)
    assert np.all(float_quantize(zero_tensor, unit_scale, 2, 3) == zero_tensor)
    testcase_a = np.asarray([1.5], dtype=np.float32)
    testcase_b = np.asarray([3.25], dtype=np.float32)
    testcase_c = np.asarray([8.0], dtype=np.float32)
    testcase_d = np.asarray([28.2], dtype=np.float32)
    testcase_e = np.asarray([6.1], dtype=np.float32)
    testcase_f = np.asarray([0.124], dtype=np.float32)
    assert np.all(float_quantize(testcase_a, unit_scale, 2, 3) == testcase_a)
    assert np.all(float_quantize(testcase_b, unit_scale, 2, 3) == testcase_b)
    assert np.all(float_quantize(testcase_c, unit_scale, 2, 3) == compute_max_val(2, 3))
    assert np.all(float_quantize(testcase_d, unit_scale, 3, 2) == compute_max_val(3, 2))
    assert np.all(float_quantize(testcase_e, unit_scale, 2, 1) == compute_max_val(2, 1))
    assert np.all(float_quantize(testcase_f, unit_scale, 2, 3, lt_subnorm_to_zero=True) == 0.0)
