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
TODO


#### Sample Implementation
TODO
