# Copyright (c) 2020 Xilinx, Inc.
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

from qonnx.core.datatype import DataType, resolve_datatype


def test_datatypes():
    assert DataType["BIPOLAR"].allowed(-1)
    assert bool(DataType["BIPOLAR"].allowed(0)) is False
    assert bool(DataType["BINARY"].allowed(-1)) is False
    assert DataType["BINARY"].allowed(1)
    assert bool(DataType["TERNARY"].allowed(2)) is False
    assert DataType["TERNARY"].allowed(-1)
    assert DataType["UINT2"].allowed(2)
    assert bool(DataType["UINT2"].allowed(10)) is False
    assert DataType["UINT3"].allowed(5)
    assert bool(DataType["UINT3"].allowed(-7)) is False
    assert DataType["UINT4"].allowed(15)
    assert bool(DataType["UINT4"].allowed(150)) is False
    assert DataType["UINT8"].allowed(150)
    assert bool(DataType["UINT8"].allowed(777)) is False
    assert DataType["UINT8"].to_numpy_dt() == np.uint8
    assert DataType["UINT16"].allowed(14500)
    assert DataType["UINT16"].to_numpy_dt() == np.uint16
    assert bool(DataType["UINT16"].allowed(-1)) is False
    assert DataType["UINT32"].allowed(2**10)
    assert bool(DataType["UINT32"].allowed(-1)) is False
    assert DataType["UINT32"].to_numpy_dt() == np.uint32
    assert DataType["INT2"].allowed(-1)
    assert bool(DataType["INT2"].allowed(-10)) is False
    assert bool(DataType["INT3"].allowed(5)) is False
    assert DataType["INT3"].allowed(-2)
    assert bool(DataType["INT4"].allowed(15)) is False
    assert DataType["INT4"].allowed(-5)
    assert bool(DataType["INT8"].allowed(150)) is False
    assert DataType["INT8"].allowed(-127)
    assert DataType["INT8"].to_numpy_dt() == np.int8
    assert bool(DataType["INT16"].allowed(-1.04)) is False
    assert DataType["INT16"].allowed(-7777)
    assert DataType["INT16"].to_numpy_dt() == np.int16
    assert bool(DataType["INT32"].allowed(7.77)) is False
    assert DataType["INT32"].allowed(-5)
    assert DataType["INT32"].allowed(5)
    assert DataType["INT32"].to_numpy_dt() == np.int32
    assert DataType["BINARY"].signed() is False
    assert DataType["FLOAT32"].signed()
    assert DataType["BIPOLAR"].signed()
    assert DataType["TERNARY"].signed()
    assert str(DataType["TERNARY"]) == "TERNARY"


def test_datatypes_fixedpoint():
    assert DataType["FIXED<4,2>"].allowed(0.5)
    assert DataType["FIXED<4,2>"].to_numpy_dt() == np.float32
    assert DataType["FIXED<4,2>"].signed()
    assert DataType["FIXED<4,2>"].scale_factor() == 0.25
    assert DataType["FIXED<4,2>"].min() == -2.0
    assert DataType["FIXED<4,2>"].max() == 1.75
    assert DataType["FIXED<4,2>"].allowed(1.5)
    assert DataType["FIXED<4,2>"].allowed(-1.5)
    assert bool(DataType["FIXED<4,2>"].allowed(1.8)) is False
    assert DataType["FIXED<4,2>"].is_integer() is False
    assert DataType["FIXED<4,2>"].is_fixed_point() is True
    assert str(DataType["FIXED<4,2"]) == "FIXED<4,2>"


def test_datatypes_arbprecfloat():
    assert DataType["FLOAT<4,3>"].allowed(0.0)
    assert DataType["FLOAT<4,0>"].allowed(0.0)
    assert DataType["FLOAT<4,3>"].allowed(0.5)
    assert DataType["FLOAT<4,3>"].allowed(1.875)
    assert DataType["FLOAT<4,3>"].allowed(-1.5)
    assert bool(DataType["FLOAT<4,3>"].allowed(1.8)) is False
    assert bool(DataType["FLOAT<4,3>"].allowed(-(2.0 * 2**8))) is False
    assert DataType["FLOAT<4,3>"].min() == -1.875 * 2**8
    assert DataType["FLOAT<4,3>"].max() == 1.875 * 2**8
    assert DataType["FLOAT<4,3>"].to_numpy_dt() == np.float32
    assert DataType["FLOAT<4,3>"].signed()
    assert DataType["FLOAT<4,3>"].is_integer() is False
    assert DataType["FLOAT<4,3>"].is_fixed_point() is False
    assert str(DataType["FLOAT<4,3>"]) == "FLOAT<4,3,7>"
    # test denormals
    assert bool(DataType["FLOAT<4,3>"].allowed(0.013671875)) is True  # b1.110 * 2**-7
    assert bool(DataType["FLOAT<4,3>"].allowed(0.0087890625)) is False  # b1.001 * 2**-7
    assert bool(DataType["FLOAT<4,3>"].allowed(0.001953125)) is True  # b1.000 * 2**-9
    assert bool(DataType["FLOAT<4,3>"].allowed(0.0009765625)) is False  # b1.000 * 2**-10
    assert bool(DataType["FLOAT<4,0>"].allowed(0.5)) is True  # b1.000 * 2**-1
    assert bool(DataType["FLOAT<4,0>"].allowed(0.75)) is False  # b1.100 * 2**-1
    assert bool(DataType["FLOAT<4,0>"].allowed(0.015625)) is True  # b1.000 * 2**-6
    assert bool(DataType["FLOAT<4,0>"].allowed(0.0078125)) is False  # b1.000 * 2**-7
    # test custom exponent bias
    assert DataType["FLOAT<4,3,5>"].allowed(0.0)
    assert DataType["FLOAT<4,0,5>"].allowed(0.0)
    assert DataType["FLOAT<4,3,5>"].allowed(0.5)
    assert DataType["FLOAT<4,3,5>"].allowed(1.875)
    assert DataType["FLOAT<4,3,5>"].allowed(-1.5)
    assert bool(DataType["FLOAT<4,3,5>"].allowed(1.8)) is False
    assert bool(DataType["FLOAT<4,3,5>"].allowed(-(2.0 * 2**8))) is True
    assert DataType["FLOAT<4,3,5>"].min() == -1.875 * 2**10
    assert DataType["FLOAT<4,3,5>"].max() == 1.875 * 2**10
    assert str(DataType["FLOAT<4,3,5>"]) == "FLOAT<4,3,5>"
    assert bool(DataType["FLOAT<4,0,5>"].allowed(0.0625)) is True  # b1.000 * 2**-4
    assert bool(DataType["FLOAT<4,0,5>"].allowed(0.03125)) is False  # b1.000 * 2**-5


def test_smallest_possible():
    assert DataType.get_smallest_possible(1) == DataType["BINARY"]
    assert DataType.get_smallest_possible(1.1) == DataType["FLOAT32"]
    assert DataType.get_smallest_possible(-1) == DataType["BIPOLAR"]
    assert DataType.get_smallest_possible(-3) == DataType["INT3"]
    assert DataType.get_smallest_possible(-3.2) == DataType["FLOAT32"]


def test_resolve_datatype():
    assert resolve_datatype("BIPOLAR")
    assert resolve_datatype("BINARY")
    assert resolve_datatype("TERNARY")
    assert resolve_datatype("UINT2")
    assert resolve_datatype("UINT3")
    assert resolve_datatype("UINT4")
    assert resolve_datatype("UINT8")
    assert resolve_datatype("UINT16")
    assert resolve_datatype("UINT32")
    assert resolve_datatype("INT2")
    assert resolve_datatype("INT3")
    assert resolve_datatype("INT4")
    assert resolve_datatype("INT8")
    assert resolve_datatype("INT16")
    assert resolve_datatype("INT32")
    assert resolve_datatype("FLOAT32")


def test_input_type_error():
    def test_resolve_datatype(input):
        # test with invalid input to check if the TypeError works
        try:
            resolve_datatype(input)  # This should raise a TypeError
        except TypeError:
            pass
        else:
            assert False, "Test with invalid input failed: No TypeError was raised."

    test_resolve_datatype(123)
    test_resolve_datatype(1.23)
    test_resolve_datatype(DataType["BIPOLAR"])
    test_resolve_datatype(DataType["BINARY"])
    test_resolve_datatype(DataType["TERNARY"])
    test_resolve_datatype(DataType["UINT2"])
    test_resolve_datatype(DataType["UINT3"])
    test_resolve_datatype(DataType["UINT4"])
    test_resolve_datatype(DataType["UINT8"])
    test_resolve_datatype(DataType["UINT16"])
    test_resolve_datatype(DataType["UINT32"])
    test_resolve_datatype(DataType["INT2"])
    test_resolve_datatype(DataType["INT3"])
    test_resolve_datatype(DataType["INT4"])
    test_resolve_datatype(DataType["INT8"])
    test_resolve_datatype(DataType["INT16"])
    test_resolve_datatype(DataType["INT32"])
    test_resolve_datatype(DataType["FLOAT32"])

vectorize_details = {
    "BIPOLAR": [
        np.array([
            [-1, +1,  0],
            [ 0, +1, -1],
            [+1,  0, -1]
        ]),
        np.array([
            [True, True, False],
            [False, True, True],
            [True, False, True]
        ], dtype=bool)
    ],
    "BINARY": [
        np.array([
            [-1, +1,  0],
            [ 0, +1, -1],
            [+1,  0, -1]
        ]),
        np.array([
            [False, True, True],
            [True, True, False],
            [True, True, False]
        ], dtype=bool)
    ],
    "TERNARY": [
        np.array([
            [-1, +2, +1,  0],
            [ 0, +1, +2, -1],
            [+2, +1,  0, -1]
        ]),
        np.array([
            [True, False, True, True],
            [True, True, False, True],
            [False, True, True, True]
        ], dtype=bool)
    ],
    "UINT2": [
        np.array([
            [[-1, +2, +1,  0],
            [ 0, +1, +2, -1]],
            [[+2, +1,  0, -1],
            [+4, -1, -2, +3]],
        ]),
        np.array([
            [[False, True, True, True],
            [True, True, True, False]],
            [[True, True, True, False],
            [False, False, False, True]],
        ], dtype=bool)
    ],
    "UINT3": [
        np.array([
            [[+9, -6, +3,  0],
            [-4, +4,  0, +1]],
            [[-1, +3, +10, +4],
            [+2, +6, +7, +8]],
        ]),
        np.array([
            [[False, False, True, True],
            [False, True, True, True]],
            [[False, True, False, True],
            [True, True, True, False]],
        ], dtype=bool)
    ],
    "UINT4": [
        np.array([
            [[-10, -4, +9, +13],
            [+1, +14,  +11, +4]],
            [[+18, -7, +1, +9],
            [-1, -7, +1, -2]],
        ]),
        np.array([
            [[False, False, True, True],
            [True, True, True, True]],
            [[False, False, True, True],
            [False, False, True, False]],
        ], dtype=bool)
    ],
    "UINT8": [
        np.array([
            [[148,  61,  70,  29],
            [244, 213,  10, 135]],
            [[18,  25, 246, 137],
            [236, -31, 220, 359]],
        ]),
        np.array([
            [[True, True, True, True],
            [True, True, True, True]],
            [[True, True, True, True],
            [True, False, True, False]],
        ], dtype=bool)
    ],
    "UINT16": [
        np.array([
            [[35261, 129491,   9136,  18643],
            [128532,   -597,  34768,    248]],
            [[21646,  30778,  71076,  21224],
            [60657,  52854,  -5994,  17295]],
        ]),
        np.array([
            [[True, False, True, True],
            [False, False, True, True]],
            [[True, True, False, True],
            [True, True, False, True]],
        ], dtype=bool)
    ],
    "UINT32": [
        np.array([
            [[-417565331,  3488834022, -1757218812,   591311876],
            [1842515574,  4131239283,  2022242400,  1240578991]],
            [[609779043,   574774725,  4188472937,  3109757181],
            [-767760560, -2100731532,  3794040092,  3223013612]],
        ]),
        np.array([
            [[False, True, False, True],
            [True, True, True, True]],
            [[True, True, True, True],
            [False, False, True, True]],
        ], dtype=bool)
    ],
    "INT2": [
        np.array([
            [[ 0,  2,  2,  3],
            [-4,  2, -1,  2]],
            [[ 1,  2, -4, -1],
            [ 2, -1, -1, -2]],
        ]),
        np.array([
            [[True, False, False, False],
            [False, False, True, False]],
            [[True, False, False, True],
            [False, True, True, True]],
        ], dtype=bool)
    ],
    "INT3": [
        np.array([
            [[-4, -6, -7,  3],
            [ 2, -8, -7,  3]],
            [[-4, -4,  4, -4],
            [ 1, -4,  1, -5]],
        ]),
        np.array([
            [[True, False, False, True],
            [True, False, False, True]],
            [[True, True, False, True],
            [True, True, True, False]],
        ], dtype=bool)
    ],
    "INT4": [
        np.array([
            [[  5,   9,   3,  -6],
            [  1,   5,   9,  10]],
            [[ 10,  10,  -3,   0],
            [ -8,  -5, -12,  -5]],
        ]),
        np.array([
            [[True, False, True, True],
            [True, True, False, False]],
            [[False, False, True, True],
            [True, True, False, True]],
        ], dtype=bool)
    ],
    "INT8": [
        np.array([
            [[-143,  140,   54, -217],
            [  22,  186,   72, -175]],
            [[-126,   -6,  115,  240],
            [-87, -159,  128, -178]],
        ]),
        np.array([
            [[False, False, True, False],
            [True, False, True, False]],
            [[True, True, True, False],
            [True, False, False, False]],
        ], dtype=bool)
    ],
    "INT16": [
        np.array([
            [[ 36863,   2676,   2728, -61500],
            [ 24314,  18040, -39438,  64013]],
            [[ 28824, -38855,  46308, -50728],
            [-50275, -48853, -42034, -44384]],
        ]),
        np.array([
            [[False, True, True, False],
            [True, True, False, False]],
            [[True, False, False, False],
            [False, False, False, False]],
        ], dtype=bool)
    ],
    "FIXED<4,2>": [
        np.array([
            [[1.8, 1.5, -0.25, 0],
            [-1.1, -2, 1.75, 0.1]],
            [[-1.5, 1.6, 0.5, 0.1],
            [0.4, 0.001, 3.03, 1.75]],
        ]),
        np.array([
            [[False, True, True, True],
            [False, True, True, False]],
            [[True, False, True, False],
            [False, False, False, True]],
        ], dtype=bool)
    ],
    "FLOAT<4,3>": [
        np.array([
            [0.0, 0.5, 1.875, -1.5],
            [1.8, -512.0, 0.013671875, 0.0087890625],
            [0.001953125, 0.0009765625, 2.0, 1.25]
        ]),
        np.array([
            [True, True, True, True],
            [False, False, True, False],
            [True, False, True, True]
        ])
    ],
    "FLOAT<4,0>": [
        np.array([
            [0.0, 0.5, 0.75],
            [0.015625, 0.0078125, 0.0625]
        ]),
        np.array([
            [True, True, False],
            [True, False, True]
        ])
    ],
    "FLOAT<4,3,5>": [
        np.array([
            [0.0, 0.5, 1.875],
            [-1.5, 1.8, -512.0]
        ]),
        np.array([
            [True, True, True],
            [True, False, True]
        ])
    ],
    "FLOAT<4,0,5>": [
        np.array([0.0, 0.0625, 0.03125]),
        np.array([True, True, False])
    ]
}

@pytest.mark.parametrize("datatype", vectorize_details.keys())
def test_vectorized_allowed(datatype):
    input_values, golden_out = vectorize_details[datatype]
    produced_out = DataType[datatype].allowed(input_values)
    assert np.all(golden_out == produced_out)
