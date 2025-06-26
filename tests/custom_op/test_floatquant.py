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


import io
import mock
import numpy as np
from brevitas.core.function_wrapper.clamp import FloatClamp, TensorClamp
from brevitas.core.function_wrapper.misc import Identity
from brevitas.core.quant.float import FloatQuant as BrevitasFloatQuant
from hypothesis import HealthCheck, Verbosity, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from pkgutil import get_data

import qonnx.core.onnx_exec as oxe
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.floatquant import compute_default_exponent_bias, compute_max_val
from qonnx.custom_op.general.floatquant import float_quant as qonnx_float_quant
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes


def test_float_exported_graph_exec():
    # Load the exported QONNX model and reference values
    qonnx_model = ModelWrapper(get_data("qonnx.data", "onnx/floatquant_exec/qonnx_act_weight_fp8.onnx"))
    input_values = np.load(io.BytesIO(get_data("qonnx.data", "onnx/floatquant_exec/test_data/input.npy")), allow_pickle=True)
    brevitas_output = np.load(
        io.BytesIO(get_data("qonnx.data", "onnx/floatquant_exec/test_data/output.npy")), allow_pickle=True
    )
    activation = np.load(
        io.BytesIO(get_data("qonnx.data", "onnx/floatquant_exec/test_data/activation.npz")), allow_pickle=True
    )
    qonnx_model = qonnx_model.transform(InferShapes())
    qonnx_model = qonnx_model.transform(GiveUniqueNodeNames())
    qonnx_model = qonnx_model.transform(GiveReadableTensorNames())

    input_name = qonnx_model.graph.input[0].name
    input_dict = {input_name: input_values}
    qonnx_output_dict = oxe.execute_onnx(qonnx_model, input_dict, return_full_exec_context=True)
    qonnx_output = qonnx_output_dict[qonnx_model.graph.output[0].name]

    # Compare the outputs
    assert np.isclose(brevitas_output, qonnx_output, atol=1e-4).all()

    brevitas_qi = activation["input_quant"]
    qonnx_qi = qonnx_output_dict["FloatQuant_0_out0"]
    assert np.isclose(brevitas_qi, qonnx_qi, atol=1e-4).all()

    brevitas_qw = activation["weight_quant"]
    qonnx_qw = qonnx_output_dict["FloatQuant_1_out0"]
    assert np.isclose(brevitas_qw, qonnx_qw, atol=1e-4).all()


def test_compute_max_val():
    # reference max normal values from OCP MX 1.0 standard
    assert compute_max_val(2, 3) == 7.5  # FP6 E2M3
    assert compute_max_val(3, 2) == 28.0  # FP6 E3M2
    assert compute_max_val(2, 1) == 6.0  # FP4 E2M1


def test_float_quantize():
    zero_tensor = np.zeros((2, 2))
    unit_scale = np.asarray([1.0], dtype=np.float32)
    assert np.all(qonnx_float_quant(zero_tensor, unit_scale, 2, 3) == zero_tensor)
    testcase_a = np.asarray([1.5], dtype=np.float32)
    testcase_b = np.asarray([3.25], dtype=np.float32)
    testcase_c = np.asarray([8.0], dtype=np.float32)
    testcase_d = np.asarray([28.2], dtype=np.float32)
    testcase_e = np.asarray([6.1], dtype=np.float32)
    assert np.all(qonnx_float_quant(testcase_a, unit_scale, 2, 3) == testcase_a)
    assert np.all(qonnx_float_quant(testcase_b, unit_scale, 2, 3) == testcase_b)
    assert np.all(qonnx_float_quant(testcase_c, unit_scale, 2, 3) == compute_max_val(2, 3))
    assert np.all(qonnx_float_quant(testcase_d, unit_scale, 3, 2) == compute_max_val(3, 2))
    assert np.all(qonnx_float_quant(testcase_e, unit_scale, 2, 1) == compute_max_val(2, 1))


def brevitas_float_quant(x, bit_width, exponent_bit_width, mantissa_bit_width, exponent_bias, signed, max_val):
    float_clamp = FloatClamp(
        tensor_clamp_impl=TensorClamp(),
        signed=signed,
        inf_values=None,
        nan_values=None,
        max_available_float=max_val,
        saturating=True,
    )
    float_scaling_impl = mock.Mock(side_effect=lambda x, y, z: 1.0)
    float_quant = BrevitasFloatQuant(
        bit_width=bit_width,
        float_scaling_impl=float_scaling_impl,
        exponent_bit_width=exponent_bit_width,
        mantissa_bit_width=mantissa_bit_width,
        exponent_bias=exponent_bias,
        input_view_impl=Identity(),
        signed=signed,
        float_clamp_impl=float_clamp,
    )
    expected_out, *_ = float_quant(x)
    return expected_out


@given(
    x=arrays(
        dtype=np.float64,
        shape=100,
        elements=st.floats(
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=True,
            width=64,  # Use 64-bit floats
        ),
        unique=True,
    ),
    exponent_bit_width=st.integers(1, 8),
    mantissa_bit_width=st.integers(1, 8),
    sign=st.booleans(),
)
@settings(
    max_examples=1000, verbosity=Verbosity.verbose, suppress_health_check=list(HealthCheck)
)  # Adjust the number of examples as needed
def test_brevitas_vs_qonnx(x, exponent_bit_width, mantissa_bit_width, sign):
    bit_width = exponent_bit_width + mantissa_bit_width + int(sign)

    assume(bit_width <= 8 and bit_width >= 4)
    scale = 1.0
    exponent_bias = compute_default_exponent_bias(exponent_bit_width)
    max_val = compute_max_val(exponent_bit_width, mantissa_bit_width, exponent_bias)
    xq_t = brevitas_float_quant(x, bit_width, exponent_bit_width, mantissa_bit_width, exponent_bias, sign, max_val).numpy()
    xq = qonnx_float_quant(x, scale, exponent_bit_width, mantissa_bit_width, exponent_bias, sign, max_val)
    np.testing.assert_array_equal(xq, xq_t)
