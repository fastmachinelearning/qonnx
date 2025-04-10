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
# see src/qonnx/custom_op/general/floatquant.py for up-to-date implementation
def float_quant(
    X,
    scale,
    exponent_bitwidth,
    mantissa_bitwidth,
    exponent_bias,
    signed,
    max_val=None,
    has_inf=False,
    has_nan=False,
    has_subnormal=False,
    rounding_mode="ROUND",
    saturation=True
):
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
    # the comments are left to track the correspondence with the brevitas code
    # np version of brevitas function
    def inf_nan_clamp(X, inf_mask, p_max_val_mask, n_max_val_mask):
        if has_inf:
            X[p_max_val_mask] = np.inf
            X[n_max_val_mask] = -np.inf
        elif has_nan:
            full_max_val_mask = np.logical_or(p_max_val_mask, n_max_val_mask)
            X[full_max_val_mask] = np.nan
            X[inf_mask] = np.nan
        else:
            raise RuntimeError(
                "Clamping is not saturating, but neither `inf_values` nor `nan_values` is specified"
            )
        return X

    # consistency check
    # if bit_width != exponent_bitwidth + mantissa_bitwidth + int(signed):
    #         raise RuntimeError("Mismatch between total bit-width, exponent, mantissa and sign.")

    # x = self.input_view_impl(x) # assuming input_view_impl is Identity

    # the following lines (up to max_value assignment) implements the float_internal_scale function from brevitas using numpy
    # internal_scale = float_internal_scale(
    #     scaled_x, self.mantissa_bit_width(), self.fp_internal_scale_min(), self.eps)

    X = X / scale

    eps = np.finfo(X.dtype).tiny # the datatype used here and in brevitas must be the same to have the same eps
    fp_internal_scale_min = 1. - exponent_bias - mantissa_bitwidth

    internal_scale = np.floor(np.log2(np.abs(X) + eps)) - mantissa_bitwidth
    internal_scale = np.maximum(internal_scale, fp_internal_scale_min) # np version of: internal_scale = torch.ok(internal_scale, fp_internal_scale_min)
    internal_scale = np.exp2(internal_scale)

    x_q = internal_scale * resolve_rounding_mode(rounding_mode)(X / internal_scale) # self.float_to_int_impl(x / internal_scale)

    max_value = compute_max_val(exponent_bitwidth, mantissa_bitwidth, exponent_bias)
    max_value = max_value if max_val is None else np.minimum(max_value, max_val)
    min_value = 0. if not signed else -max_value

    # Compute masks
    inf_mask = np.isinf(x_q)
    p_max_val_mask = x_q > max_value
    n_max_val_mask = x_q < min_value

    # first clamp everything to  [min_value,max_value], basically the saturating case
    x_q = np.clip(x_q, min_value, max_value) # self.saturating_clamp(x_q, max_value, min_value)

    if not saturation:
        x_q = inf_nan_clamp(x_q, inf_mask, p_max_val_mask, n_max_val_mask)

    return x_q * scale #, self.saturating, self.inf_values, self.nan_values
