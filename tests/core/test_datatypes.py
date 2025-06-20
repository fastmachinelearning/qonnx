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

import numpy as np

from qonnx.core.datatype import DataType, resolve_datatype


def test_datatypes():
    assert DataType["BIPOLAR"].allowed(-1)
    assert DataType["BIPOLAR"].allowed(0) is False
    assert DataType["BINARY"].allowed(-1) is False
    assert DataType["BINARY"].allowed(1)
    assert DataType["TERNARY"].allowed(2) is False
    assert DataType["TERNARY"].allowed(-1)
    assert DataType["UINT2"].allowed(2)
    assert DataType["UINT2"].allowed(10) is False
    assert DataType["UINT3"].allowed(5)
    assert DataType["UINT3"].allowed(-7) is False
    assert DataType["UINT4"].allowed(15)
    assert DataType["UINT4"].allowed(150) is False
    assert DataType["UINT8"].allowed(150)
    assert DataType["UINT8"].allowed(777) is False
    assert DataType["UINT8"].to_numpy_dt() == np.uint8
    assert DataType["UINT16"].allowed(14500)
    assert DataType["UINT16"].to_numpy_dt() == np.uint16
    assert DataType["UINT16"].allowed(-1) is False
    assert DataType["UINT32"].allowed(2**10)
    assert DataType["UINT32"].allowed(-1) is False
    assert DataType["UINT32"].to_numpy_dt() == np.uint32
    assert DataType["INT2"].allowed(-1)
    assert DataType["INT2"].allowed(-10) is False
    assert DataType["INT3"].allowed(5) is False
    assert DataType["INT3"].allowed(-2)
    assert DataType["INT4"].allowed(15) is False
    assert DataType["INT4"].allowed(-5)
    assert DataType["INT8"].allowed(150) is False
    assert DataType["INT8"].allowed(-127)
    assert DataType["INT8"].to_numpy_dt() == np.int8
    assert DataType["INT16"].allowed(-1.04) is False
    assert DataType["INT16"].allowed(-7777)
    assert DataType["INT16"].to_numpy_dt() == np.int16
    assert DataType["INT32"].allowed(7.77) is False
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
    assert DataType["FIXED<4,2>"].allowed(1.8) is False
    assert DataType["FIXED<4,2>"].is_integer() is False
    assert DataType["FIXED<4,2>"].is_fixed_point() is True
    assert str(DataType["FIXED<4,2"]) == "FIXED<4,2>"


def test_datatypes_arbprecfloat():
    assert DataType["FLOAT<4,3>"].allowed(0.0)
    assert DataType["FLOAT<4,0>"].allowed(0.0)
    assert DataType["FLOAT<4,3>"].allowed(0.5)
    assert DataType["FLOAT<4,3>"].allowed(1.875)
    assert DataType["FLOAT<4,3>"].allowed(-1.5)
    assert DataType["FLOAT<4,3>"].allowed(1.8) is False
    assert DataType["FLOAT<4,3>"].allowed(-(2.0 * 2**8)) is False
    assert DataType["FLOAT<4,3>"].min() == -1.875 * 2**8
    assert DataType["FLOAT<4,3>"].max() == 1.875 * 2**8
    assert DataType["FLOAT<4,3>"].to_numpy_dt() == np.float32
    assert DataType["FLOAT<4,3>"].signed()
    assert DataType["FLOAT<4,3>"].is_integer() is False
    assert DataType["FLOAT<4,3>"].is_fixed_point() is False
    assert str(DataType["FLOAT<4,3>"]) == "FLOAT<4,3,7>"
    # test denormals
    assert DataType["FLOAT<4,3>"].allowed(0.013671875) is True  # b1.110 * 2**-7
    assert DataType["FLOAT<4,3>"].allowed(0.0087890625) is False  # b1.001 * 2**-7
    assert DataType["FLOAT<4,3>"].allowed(0.001953125) is True  # b1.000 * 2**-9
    assert DataType["FLOAT<4,3>"].allowed(0.0009765625) is False  # b1.000 * 2**-10
    assert DataType["FLOAT<4,0>"].allowed(0.5) is True  # b1.000 * 2**-1
    assert DataType["FLOAT<4,0>"].allowed(0.75) is False  # b1.100 * 2**-1
    assert DataType["FLOAT<4,0>"].allowed(0.015625) is True  # b1.000 * 2**-6
    assert DataType["FLOAT<4,0>"].allowed(0.0078125) is False  # b1.000 * 2**-7
    # test custom exponent bias
    assert DataType["FLOAT<4,3,5>"].allowed(0.0)
    assert DataType["FLOAT<4,0,5>"].allowed(0.0)
    assert DataType["FLOAT<4,3,5>"].allowed(0.5)
    assert DataType["FLOAT<4,3,5>"].allowed(1.875)
    assert DataType["FLOAT<4,3,5>"].allowed(-1.5)
    assert DataType["FLOAT<4,3,5>"].allowed(1.8) is False
    assert DataType["FLOAT<4,3,5>"].allowed(-(2.0 * 2**8)) is True
    assert DataType["FLOAT<4,3,5>"].min() == -1.875 * 2**10
    assert DataType["FLOAT<4,3,5>"].max() == 1.875 * 2**10
    assert str(DataType["FLOAT<4,3,5>"]) == "FLOAT<4,3,5>"
    assert DataType["FLOAT<4,0,5>"].allowed(0.0625) is True  # b1.000 * 2**-4
    assert DataType["FLOAT<4,0,5>"].allowed(0.03125) is False  # b1.000 * 2**-5


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
