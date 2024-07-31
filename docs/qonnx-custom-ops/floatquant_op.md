### <a name="FloatQuant"></a><a name="abs">**FloatQuant**</a>

Calculates the [minifloat-quantized](https://arxiv.org/abs/2311.12359) values of one input data (Tensor<T>) and produces one output data (Tensor<T>).
Additionally, takes three floats as input, which define the scale, exponent bitwidth and mantissa bitwidth of the quantization,
all of which may be scalars or tensors with shapes broadcastable to the shape of the input data tensor. This can be used to
control the granularity of the quantization. For instance, a scalar scale operand implies per-tensor scaling, while a scale operand with
the same shape as the input data implies per-element scaling.
TODO add comment about attributes when clarified.


Note: This operator is not intended for integer quantization, for this purpose the `IntQuant` custom op exists.

#### Version

This operator is not part of the ONNX standard and is not currently versioned.

#### Attributes

<dl>
<dt><tt>float_mode</tt> : string (default is "")</dt>
<dd>Defines the floating point mode used by the quantizer, which defines behaviors such as whether infinities and NaN are represented.
See the "Float Mode" section for more details.</dd>
<dt><tt>subnormal</tt> : int (default is 1)</dt>
<dd>Defines whether subnormal values are supported. Subnormal values have an exponent value of 1 and are interpreted to have a leading
significand digit of zero rather than one.</dd>
<dt><tt>rounding_mode</tt> : string (default is TODO)</dt>
<dd>TODO.</dd>
<dt><tt>saturation</tt> : int (default is 1)</dt>
<dd>TODO.</dd>
</dl>

#### Inputs

<dl>
<dt><tt>X</tt> (differentiable) : tensor(float32)</dt>
<dd>input tensor to quantize</dd>
<dt><tt>scale</tt> : float32, tensor(float32)</dt>
<dd>The scale factor, either as a global scalar or with a broadcastable shape matching the number of dimensions of the X tensor</dd>
<dt><tt>mantissa_bitwidth</tt> : int32, float32</dt>
<dd>The number of bits for the mantissa used by the quantization, must be a positive integer. If float32 dtype is used for convenience, it must still represent an positive integer number of bits.</dd>
<dt><tt>mantissa_bitwidth</tt> : int32, float32</dt>
<dd>The number of bits for the exponent used by the quantization, must be a positive integer. If float32 dtype is used for convenience, it must still represent an positive integer number of bits.</dd>

</dl>


#### Outputs

<dl>
<dt><tt>Y</tt> (differentiable) : tensor(float32)</dt>
<dd>Output tensor</dd>
</dl>


#### Float Mode
TODO FNUZ etc

#### Examples
TODO


#### Sample Implementation
TODO
