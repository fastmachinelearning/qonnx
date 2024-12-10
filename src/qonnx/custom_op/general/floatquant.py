# Copyright (c) 2024 Nicolo Ghielmetti
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

from qonnx.custom_op.general.quant import resolve_rounding_mode


def compute_default_exponent_bias(exponent_bitwidth):
    return (2.0 ** (exponent_bitwidth - 1)) - 1


def compute_max_val(exponent_bitwidth, mantissa_bitwidth, exponent_bias=None):
    if exponent_bias is None:
        exponent_bias = compute_default_exponent_bias(exponent_bitwidth)
    max_exponent = (2.0**exponent_bitwidth) - 1.0 - exponent_bias
    max_mantissa = np.sum((2.0 ** np.arange(0, -1.0 * mantissa_bitwidth - 1.0, -1.0)))
    max_val = max_mantissa * (2**max_exponent)
    return max_val


def float_quantize(X, scale, exponent_bitwidth, mantissa_bitwidth, exponent_bias=None, max_val=None, rounding_mode="ROUND"):
    """Quantize a given floating point array to minifloat format by specifying the desired minifloat quantization"""
    if exponent_bias is None:
        exponent_bias = compute_default_exponent_bias(exponent_bitwidth)
    if max_val is None:
        max_val = compute_max_val(exponent_bitwidth, mantissa_bitwidth, exponent_bias)
    # copy the sign of the input
    sign = np.sign(X)
    # compute the mask of the values equal to 0 - it will always be zero at the output
    zero_mask = np.where(X == 0)
    # copy the input in order to not modify it
    X = X.copy()
    # set the zeros to 1.0 - but could be any random value
    X[zero_mask] = 1.0
    # apply the scale to the input
    X /= scale
    # get input exponents from the floats - no need to use eps since the zeros have been already removed
    e_inp = np.floor(np.log2(np.abs(X)))
    # compute the max exponent given the exponent bitwidth.
    # Note: inf/NaN representation is included and it is clipped at the end of this function
    e_max = np.maximum(2.0 ** (exponent_bitwidth), 1.0)
    # compute exponent range given the max exponent. e_low represent the subnormals of the
    # quantized representation, e_high the infs/NaNs
    e_low, e_high = -e_max + exponent_bias + 1, e_max + exponent_bias
    # limit the value of the exponent given the quantization range
    e_quant = np.clip(e_inp, e_low, e_high)
    # compute the shift to get the quantized value rounded properly. This part basically quantize the mantissa
    # (round the mantissa by setting to 0 the bits not beloging to the quantised representation)
    round_shift = 2.0 ** (e_quant - mantissa_bitwidth)
    # apply the shift
    man = X / round_shift
    # round the mantissa
    man_quant = resolve_rounding_mode(rounding_mode)(man)
    # compute the max value of the mantissa (i.e. all the mantissa bits set to 1)
    man_max = 2.0 ** (mantissa_bitwidth + 1) - 1
    # if the quantised value is a subnormal, remove 1 from the mantissa (i.e. 1 + 2**m => 2**m)
    man_max = np.where(e_quant != e_low, man_max, man_max - 1)
    # make sure the mantissa is in the representable range
    man_clip = np.clip(man_quant, -man_max, man_max)
    # go back to float representation
    qx = man_clip * round_shift
    # if it's inf or nan, saturates to sign*max_val
    qx = np.where(e_quant == e_high, sign * max_val, qx)
    # restore the original zeros
    qx[zero_mask] = 0.0
    # unscale the input
    qx *= scale
    return qx
