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
from abc import ABC, abstractmethod
from enum import Enum, EnumMeta
from typing import Union


class BaseDataType(ABC):
    "Base class for QONNX data types."

    def signed(self) -> bool:
        "Returns whether this DataType can represent negative numbers."
        return self.min() < 0

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BaseDataType):
            return self.get_canonical_name() == other.get_canonical_name()
        elif isinstance(other, str):
            return self.get_canonical_name() == other
        else:
            return NotImplemented

    def __hash__(self) -> int:
        return hash(self.get_canonical_name())

    @property
    def name(self) -> str:
        return self.get_canonical_name()

    def __repr__(self) -> str:
        return self.get_canonical_name()

    def __str__(self) -> str:
        return self.get_canonical_name()

    @abstractmethod
    def bitwidth(self) -> int:
        "Returns the number of bits required for this DataType."
        pass

    @abstractmethod
    def min(self) -> Union[int, float]:
        "Returns the smallest possible value allowed by this DataType."
        pass

    @abstractmethod
    def max(self) -> Union[int, float]:
        "Returns the largest possible value allowed by this DataType."
        pass

    @abstractmethod
    def allowed(self, value: Union[int, float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Check whether given value is allowed for this DataType.

        * value (float32): value to be checked"""
        pass

    @abstractmethod
    def get_num_possible_values(self) -> int:
        """Returns the number of possible values this DataType can take. Only
        implemented for integer types for now."""
        pass

    @abstractmethod
    def is_integer(self) -> bool:
        "Returns whether this DataType represents integer values only."
        pass

    @abstractmethod
    def is_fixed_point(self) -> bool:
        "Returns whether this DataType represent fixed-point values only."
        pass

    @abstractmethod
    def get_hls_datatype_str(self) -> str:
        "Returns the corresponding Vivado HLS datatype name."
        pass

    @abstractmethod
    def to_numpy_dt(self) -> np.dtype:
        "Return an appropriate numpy datatype that can represent this QONNX DataType."
        pass

    @abstractmethod
    def get_canonical_name(self) -> str:
        "Return a canonical string representation of this QONNX DataType."


class FloatType(BaseDataType):
    def bitwidth(self) -> int:
        return 32

    def min(self) -> float:
        return float(np.finfo(np.float32).min)

    def max(self) -> float:
        return float(np.finfo(np.float32).max)

    def allowed(self, value: Union[int, float, np.ndarray]) -> Union[bool, np.ndarray]:
        if isinstance(value, np.ndarray):
            return np.ones(value.shape, dtype=bool)
        return True

    def get_num_possible_values(self) -> int:
        raise Exception("Undefined for FloatType")

    def is_integer(self) -> bool:
        return False

    def is_fixed_point(self) -> bool:
        return False

    def get_hls_datatype_str(self) -> str:
        return "float"

    def to_numpy_dt(self) -> np.dtype:
        return np.dtype(np.float32)

    def get_canonical_name(self) -> str:
        return "FLOAT32"


class ArbPrecFloatType(BaseDataType):
    def __init__(
        self,
        exponent_bits: int,
        mantissa_bits: int,
        exponent_bias: Union[int, float, None] = None,
    ) -> None:
        self._exponent_bits = exponent_bits
        self._mantissa_bits = mantissa_bits

        if not exponent_bias:
            # default (IEEE-style) exponent bias
            exponent_bias = (2.0 ** (exponent_bits - 1)) - 1
        self._exponent_bias = exponent_bias

    def signed(self) -> bool:
        "Returns whether this DataType can represent negative numbers."
        return True

    def bitwidth(self) -> int:
        # sign bit + exponent bits + mantissa bits
        return 1 + self.exponent_bits() + self.mantissa_bits()

    def exponent_bits(self) -> int:
        return self._exponent_bits

    def mantissa_bits(self) -> int:
        return self._mantissa_bits

    def exponent_bias(self) -> Union[int, float]:
        return self._exponent_bias

    def min(self) -> float:
        return -1 * self.max()

    def max(self) -> float:
        # note: assumes no bits reserved for NaN/inf etc.
        exponent_bias = self.exponent_bias()
        exponent_bitwidth = self.exponent_bits()
        mantissa_bitwidth = self.mantissa_bits()
        max_exponent = (2.0**exponent_bitwidth) - 1.0 - exponent_bias
        max_mantissa = np.sum(
            (2.0 ** np.arange(0, -1.0 * mantissa_bitwidth - 1.0, -1.0))
        )
        max_val = max_mantissa * (2**max_exponent)
        return float(max_val)

    def allowed(self, value: Union[int, float, np.ndarray]) -> Union[bool, np.ndarray]:
        # fp32 format parameters
        fp32_exponent_bias = 127
        fp32_mantissa_bitwidth = 23
        fp32_nrm_mantissa_bitwidth = (
            fp32_mantissa_bitwidth + 1
        )  # width of normalized mantissa with implicit 1
        # minifloat format parameters
        exponent_bias = self.exponent_bias()
        min_exponent = (
            -exponent_bias + 1
        )  # minimum exponent if IEEE-style denormals are supported
        mantissa_bitwidth = self.mantissa_bits()
        nrm_mantissa_bitwidth = (
            mantissa_bitwidth + 1
        )  # width of normalized mantissa with implicit 1
        # extract fields from fp32 representation
        bin_val = np.float32(value).view(np.uint32)
        exp = (bin_val & 0b01111111100000000000000000000000) >> fp32_mantissa_bitwidth
        mant = bin_val & 0b00000000011111111111111111111111
        # Cast exp to signed int to avoid overflow when subtracting bias
        exp_biased = (
            exp.astype(np.int32) - fp32_exponent_bias
        )  # bias the extracted raw exponent (assume not denormal)
        # Use np.where to handle the implicit 1 bit element-wise
        mant_normalized = mant + np.where(
            exp != 0, 2**fp32_mantissa_bitwidth, 0
        ).astype(np.int32)
        # for this value to be representable as this ArbPrecFloatType:
        # the value must be within the representable range
        range_ok = (value <= self.max()) & (value >= self.min())
        # the mantissa must be within representable range:
        # no set bits in the mantissa beyond the allowed number of bits (assume value is not denormal in fp32)
        # compute bits of precision lost to tapered precision if denormal, clamp to: 0 <= dnm_shift <= nrm_mantissa_bitwidth
        dnm_shift = np.clip(min_exponent - exp_biased, 0, nrm_mantissa_bitwidth).astype(np.int32)
        available_bits = nrm_mantissa_bitwidth - dnm_shift
        
        # For arrays, we need to compute mantissa_ok element-wise
        if isinstance(value, np.ndarray):
            mantissa_ok = np.zeros(value.shape, dtype=bool)
            for idx in np.ndindex(value.shape):
                ab = int(available_bits[idx]) # type: ignore
                mantissa_mask = "0" * ab + "1" * (fp32_nrm_mantissa_bitwidth - ab)
                mantissa_ok[idx] = (mant_normalized[idx] & int(mantissa_mask, base=2)) == 0
            return mantissa_ok & range_ok
        else:
            # Scalar case
            ab = int(available_bits)
            mantissa_mask = "0" * ab + "1" * (fp32_nrm_mantissa_bitwidth - ab)
            mantissa_ok = (mant_normalized & int(mantissa_mask, base=2)) == 0
            return bool(mantissa_ok and range_ok)

    def is_integer(self) -> bool:
        return False

    def is_fixed_point(self) -> bool:
        return False

    def get_hls_datatype_str(self) -> str:
        assert False, "get_hls_datatype_str() not yet implemented for ArbPrecFloatType"

    def to_numpy_dt(self) -> np.dtype:
        return np.dtype(np.float32)

    def get_canonical_name(self) -> str:
        return "FLOAT<%d,%d,%d>" % (
            self.exponent_bits(),
            self.mantissa_bits(),
            self.exponent_bias(),
        )

    def get_num_possible_values(self) -> int:
        # TODO: consider -0 and +0 as different values?
        # also assumes no special symbols like NaN, inf etc
        return 2 ** self.bitwidth()


class Float16Type(BaseDataType):
    def bitwidth(self) -> int:
        return 16

    def min(self) -> float:
        return float(np.finfo(np.float16).min)

    def max(self) -> float:
        return float(np.finfo(np.float16).max)

    def allowed(self, value: Union[int, float, np.ndarray]) -> Union[bool, np.ndarray]:
        if isinstance(value, np.ndarray):
            return np.ones(value.shape, dtype=bool)
        return True

    def get_num_possible_values(self) -> int:
        raise Exception("Undefined for Float16Type")

    def is_integer(self) -> bool:
        return False

    def is_fixed_point(self) -> bool:
        return False

    def get_hls_datatype_str(self) -> str:
        return "half"

    def to_numpy_dt(self) -> np.dtype:
        return np.dtype(np.float16)

    def get_canonical_name(self) -> str:
        return "FLOAT16"


class IntType(BaseDataType):
    def __init__(self, bitwidth: int, signed: bool) -> None:
        super().__init__()
        self._bitwidth = bitwidth
        self._signed = signed

    def bitwidth(self) -> int:
        return self._bitwidth

    def min(self) -> Union[int, float]:
        unsigned_min = 0
        signed_min = -(2 ** (self.bitwidth() - 1))
        return signed_min if self._signed else unsigned_min

    def max(self) -> Union[int, float]:
        unsigned_max = (2 ** (self.bitwidth())) - 1
        signed_max = (2 ** (self.bitwidth() - 1)) - 1
        return signed_max if self._signed else unsigned_max

    def allowed(self, value: Union[int, float, np.ndarray]) -> Union[bool, np.ndarray]:
        if isinstance(value, np.ndarray):
            return (
                (self.min() <= value)
                & (value <= self.max())
                & np.isclose(value, np.round(value))
            )
        return (
            (self.min() <= value)
            and (value <= self.max())
            and float(value).is_integer()
        )

    def get_num_possible_values(self) -> int:
        return int(abs(self.min()) + abs(self.max()) + 1)

    def is_integer(self) -> bool:
        return True

    def is_fixed_point(self) -> bool:
        return False

    def get_hls_datatype_str(self) -> str:
        if self.signed():
            return "ap_int<%d>" % self.bitwidth()
        else:
            return "ap_uint<%d>" % self.bitwidth()

    def to_numpy_dt(self) -> np.dtype:
        if self.bitwidth() <= 8:
            return np.dtype(np.int8 if self.signed() else np.uint8)
        elif self.bitwidth() <= 16:
            return np.dtype(np.int16 if self.signed() else np.uint16)
        elif self.bitwidth() <= 32:
            return np.dtype(np.int32 if self.signed() else np.uint32)
        elif self.bitwidth() <= 64:
            return np.dtype(np.int64 if self.signed() else np.uint64)
        else:
            raise Exception("Unknown numpy dtype for " + str(self))

    def get_canonical_name(self) -> str:
        if self.bitwidth() == 1 and (not self.signed()):
            return "BINARY"
        else:
            prefix = "INT" if self.signed() else "UINT"
            return prefix + str(self.bitwidth())


class BipolarType(BaseDataType):
    def bitwidth(self) -> int:
        return 1

    def min(self) -> int:
        return -1

    def max(self) -> int:
        return +1

    def allowed(self, value: Union[int, float, np.ndarray]) -> Union[bool, np.ndarray]:
        if isinstance(value, np.ndarray):
            return np.isin(value, [-1, +1])
        return value in [-1, +1]

    def get_num_possible_values(self) -> int:
        return 2

    def is_integer(self) -> bool:
        return True

    def is_fixed_point(self) -> bool:
        return False

    def get_hls_datatype_str(self) -> str:
        return "ap_int<1>"

    def to_numpy_dt(self) -> np.dtype:
        return np.dtype(np.int8)

    def get_canonical_name(self) -> str:
        return "BIPOLAR"


class TernaryType(BaseDataType):
    def bitwidth(self) -> int:
        return 2

    def min(self) -> int:
        return -1

    def max(self) -> int:
        return +1

    def allowed(self, value: Union[int, float, np.ndarray]) -> Union[bool, np.ndarray]:
        if isinstance(value, np.ndarray):
            return np.isin(value, [-1, 0, +1])
        return value in [-1, 0, +1]

    def get_num_possible_values(self) -> int:
        return 3

    def is_integer(self) -> bool:
        return True

    def is_fixed_point(self) -> bool:
        return False

    def get_hls_datatype_str(self) -> str:
        return "ap_int<2>"

    def to_numpy_dt(self) -> np.dtype:
        return np.dtype(np.int8)

    def get_canonical_name(self) -> str:
        return "TERNARY"


class FixedPointType(IntType):
    def __init__(self, bitwidth: int, intwidth: int) -> None:
        super().__init__(bitwidth=bitwidth, signed=True)
        assert intwidth < bitwidth, "FixedPointType violates intwidth < bitwidth"
        self._intwidth = intwidth

    def int_bits(self) -> int:
        return self._intwidth

    def frac_bits(self) -> int:
        return self.bitwidth() - self.int_bits()

    def scale_factor(self) -> float:
        return 2 ** -(self.frac_bits())

    def min(self) -> float:
        return super().min() * self.scale_factor()

    def max(self) -> float:
        return super().max() * self.scale_factor()

    def allowed(self, value: Union[int, float, np.ndarray]) -> Union[bool, np.ndarray]:
        int_value = value / self.scale_factor()
        return IntType(self._bitwidth, True).allowed(int_value)

    def is_integer(self) -> bool:
        return False

    def is_fixed_point(self) -> bool:
        return True

    def get_hls_datatype_str(self) -> str:
        return "ap_fixed<%d, %d>" % (self.bitwidth(), self.int_bits())

    def to_numpy_dt(self) -> np.dtype:
        return np.dtype(np.float32)

    def get_canonical_name(self) -> str:
        return "FIXED<%d,%d>" % (self.bitwidth(), self.int_bits())


class ScaledIntType(IntType):
    # scaled integer datatype, only intended for
    # inference cost calculations, many of the
    # member methods are not implemented
    def __init__(self, bitwidth: int) -> None:
        super().__init__(bitwidth=bitwidth, signed=True)

    def min(self) -> int:
        raise Exception("Undefined for ScaledIntType")

    def max(self) -> int:
        raise Exception("Undefined for ScaledIntType")

    def allowed(self, value: Union[int, float, np.ndarray]) -> Union[bool, np.ndarray]:
        raise Exception("Undefined for ScaledIntType")

    def is_integer(self) -> bool:
        return False

    def is_fixed_point(self) -> bool:
        return False

    def get_hls_datatype_str(self) -> str:
        raise Exception("Undefined for ScaledIntType")

    def to_numpy_dt(self) -> np.dtype:
        return np.dtype(np.float32)

    def signed(self) -> bool:
        "Returns whether this DataType can represent negative numbers."
        return True

    def get_canonical_name(self) -> str:
        return "SCALEDINT<%d>" % (self.bitwidth())


def resolve_datatype(name: str) -> BaseDataType:
    if not isinstance(name, str):
        raise TypeError(
            f"Input 'name' must be of type 'str', but got type '{type(name).__name__}'"
        )

    _special_types = {
        "BINARY": IntType(1, False),
        "BIPOLAR": BipolarType(),
        "TERNARY": TernaryType(),
        "FLOAT32": FloatType(),
        "FLOAT16": Float16Type(),
    }
    if name in _special_types.keys():
        return _special_types[name]
    elif name.startswith("UINT"):
        bitwidth = int(name.replace("UINT", ""))
        return IntType(bitwidth, False)
    elif name.startswith("INT"):
        bitwidth = int(name.replace("INT", ""))
        return IntType(bitwidth, True)
    elif name.startswith("FIXED"):
        name = name.replace("FIXED<", "")
        name = name.replace(">", "")
        nums = name.split(",")
        bitwidth = int(nums[0].strip())
        intwidth = int(nums[1].strip())
        return FixedPointType(bitwidth, intwidth)
    elif name.startswith("SCALEDINT"):
        name = name.replace("SCALEDINT<", "")
        name = name.replace(">", "")
        nums = name.split(",")
        bitwidth = int(nums[0].strip())
        return ScaledIntType(bitwidth)
    elif name.startswith("FLOAT<"):
        name = name.replace("FLOAT<", "")
        name = name.replace(">", "")
        nums = name.split(",")
        if len(nums) == 2:
            exp_bits = int(nums[0].strip())
            mant_bits = int(nums[1].strip())
            return ArbPrecFloatType(exp_bits, mant_bits)
        elif len(nums) == 3:
            exp_bits = int(nums[0].strip())
            mant_bits = int(nums[1].strip())
            exp_bias = int(nums[2].strip())
            return ArbPrecFloatType(exp_bits, mant_bits, exp_bias)
        else:
            raise KeyError("Could not resolve DataType " + name)
    else:
        raise KeyError("Could not resolve DataType " + name)


class DataTypeMeta(EnumMeta):
    def __getitem__(self, name: str) -> BaseDataType:
        return resolve_datatype(name)


class DataType(Enum, metaclass=DataTypeMeta):
    """Enum class that contains QONNX data types to set the quantization annotation.
    ONNX does not support data types smaller than 8-bit integers, whereas in QONNX we are
    interested in smaller integers down to ternary and bipolar."""

    @staticmethod
    def get_accumulator_dt_cands() -> list[str]:
        cands = ["BINARY"]
        cands += ["UINT%d" % (x + 1) for x in range(64)]
        cands += ["BIPOLAR", "TERNARY"]
        cands += ["INT%d" % (x + 1) for x in range(64)]
        return cands

    @staticmethod
    def get_smallest_possible(value: Union[int, float]) -> BaseDataType:
        """Returns smallest (fewest bits) possible DataType that can represent
        value. Prefers unsigned integers where possible."""
        if not int(value) == value:
            return DataType["FLOAT32"]
        cands = DataType.get_accumulator_dt_cands()
        for cand in cands:
            dt = DataType[cand]
            if (dt.min() <= value) and (value <= dt.max()):
                return dt
        raise Exception("Could not find a suitable int datatype for " + str(value))
