### <a name="FloatQuant"></a><a name="abs">**FloatQuant**</a>

Calculates the [arbitrary-precision-float-quantized](https://arxiv.org/abs/2311.12359) values of one input data (Tensor<T>) and produces one output data (Tensor<T>).
Additionally, takes five floats as input, which define the scale, exponent bitwidth, mantissa bitwidth, maximum representable value and exponent bias of the quantization,
all of which may be scalars or tensors with shapes broadcastable to the shape of the input data tensor. This can be used to
control the granularity of the quantization. For instance, a scalar scale operand implies per-tensor scaling, while a scale operand with
the same shape as the input data implies per-element scaling.

*Special (symbolic) values:* Specialized floating point datatype behaviors such as supporting infinity, NaN and subnormals are specified by the attributes of the node to inform backends, but note that they do not affect the behavior of the `FloatQuant` operator. Instead, the `max_val` input is used to account for decreased representational range due
to having to represent special cases.

*Why `max_val` is specified explicitly?* The maximum representable value is derived from a combination of exponent and mantissa bitwidths, but also how many encodings are reserved for
special (symbolic) values. This makes it nontrivial to infer the maximum representable value. For instance, OCP E5M2 reserves three encodings for NaN, whereas E4M3 reserves only one.

*Integer quantization:* This operator is not intended for integer quantization, for this purpose the `IntQuant` custom op exists.

#### Version

This operator is not part of the ONNX standard and is not currently versioned.

#### Attributes

<dl>
<dt><tt>has_infinity</tt> : int (default is 0)</dt>
<dd>Integer value interpreted as boolean, defines whether the representation  supports infinity values. The ability to represent infinity values will  decrease the representable numerical range. This attribute has no effect on the execution of this operation and is intended purely to inform backends.</dd>

<dt><tt>has_nan</tt> : int (default is 0)</dt>
<dd>Integer value interpreted as boolean, defines whether the representation  supports not-a-number (NaN) values. The ability to represent NaN values will   decrease the representable numerical range. This attribute has no effect on the execution of this operation and is intended purely to inform backends.</dd>

<dt><tt>has_subnormal</tt> : int (default is 1)</dt>
<dd>Integer value interpreted as boolean, defines whether the representation  supports subnormal values. Subnormal values have an exponent value of 0 and are interpreted to have a leading significand digit of zero rather than one. Supporting subnormals will increase the complexity of the required arithmetic datapath. This attribute has no effect on the execution of this operation and is intended purely to inform backends.</dd>

<dt><tt>saturation</tt> : int (default is 1)</dt>
<dd>Integer value interpreted as boolean, defines whether the representation  will saturate during arithmetic. This attribute has no effect on the execution of this operation and is intended purely to inform backends.</dd>

<dt><tt>rounding_mode</tt> : string (default is "ROUND")</dt>
<dd>Defines how rounding should be applied during quantization. Currently available modes are: "ROUND", "CEIL" and "FLOOR". Here "ROUND" implies a round-to-even operation. Lowercase variants for the rounding mode string are also supported: "round", "ceil", "floor".</dd>

</dl>

#### Inputs

<dl>
<dt><tt>X</tt> : tensor(float32)</dt>
<dd>input tensor to quantize</dd>
<dt><tt>scale</tt> : tensor(float32)</dt>
<dd>The scale factor, either as a global scalar or with a broadcastable shape matching the number of dimensions of the X tensor</dd>
<dt><tt>exponent_bitwidth</tt> : tensor(float32)</dt>
<dd>The number of bits for the exponent used by the quantization, either as a global scalar or with a broadcastable shape matching the number of dimensions of the X tensor. Must be a positive integer.</dd>
<dt><tt>mantissa_bitwidth</tt> : tensor(float32)</dt>
<dd>The number of bits for the mantissa used by the quantization, either as a global scalar or with a broadcastable shape matching the number of dimensions of the X tensor. Must be a positive integer.</dd>
<dt><tt>exponent_bias</tt> : tensor(float32)</dt>
<dd>The exponent bias used by the quantization, either as a global scalar or with a broadcastable shape matching the number of dimensions of the X tensor. Must be a positive integer.</dd>
<dt><tt>max_val</tt> : tensor(float32)</dt>
<dd>Maximum possible representable value, either as a global scalar or with a broadcastable shape matching the number of dimensions of the X tensor. </dd>
</dl>


#### Outputs

<dl>
<dt><tt>Y</tt> : tensor(float32)</dt>
<dd>Output tensor</dd>
</dl>

#### Examples
```python
def compute_max_val(exponent_bit_width, mantissa_bit_width, exponent_bias):
    max_exponent = (2. ** exponent_bit_width) - 1. - exponent_bias
    max_mantissa = np.sum((
        2. ** np.arange(
            0,
            -1. * mantissa_bit_width - 1.,
            -1.
            )))
    max_val = max_mantissa * (2 ** max_exponent)
    return max_val

import numpy as np
x = np.random.rand(100).astype(np.float32)
scale = 1
exponent_bitwidth = 4
mantissa_bitwidth = 3
exponent_bias = 0
max_val = compute_max_val(exponent_bitwidth, mantissa_bitwidth, exponent_bias)
rounding_mode = 'ROUND'
signed = True
xq = float_quantize(x, scale, exponent_bitwidth, mantissa_bitwidth, exponent_bias, max_val, rounding_mode)
```


#### Sample Implementation
```python
def float_quantize(X, scale, exponent_bitwidth, mantissa_bitwidth, exponent_bias, max_val, rounding_mode):
    """Quantize a given floating point array to minifloat format by specifying the desired minifloat quantization"""

    def resolve_rounding_mode(mode_string):
        """Resolve the rounding mode string to the corresponding numpy functions."""
        mode_string = mode_string.upper()
        if mode_string == "ROUND":
            return np.round
        elif mode_string == "CEIL":
            return np.ceil
        elif mode_string == "FLOOR":
            return np.floor
        else:
            raise ValueError(f"Could not resolve rounding mode called: {mode_string}")

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
    e_max = np.maximum(2.**(exponent_bitwidth), 1.)
    # compute exponent range given the max exponent. e_low represent the subnormals of the quantized representation, e_high the infs/NaNs
    e_low, e_high = -e_max + exponent_bias + 1, e_max + exponent_bias
    # limit the value of the exponent given the quantization range
    e_quant = np.clip(e_inp, e_low, e_high)
    # compute the shift to get the quantized value rounded properly. This part basically quantize the mantissa
    # (round the mantissa by setting to 0 the bits not beloging to the quantised representation)
    round_shift = 2.**(e_quant - mantissa_bitwidth)
    # apply the shift
    man = X / round_shift
    # round the mantissa
    man_quant = resolve_rounding_mode(rounding_mode)(man)
    # compute the max value of the mantissa (i.e. all the mantissa bits set to 1)
    man_max = 2.**(mantissa_bitwidth + 1) - 1
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
```
