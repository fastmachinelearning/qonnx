### <a name="FloatQuant"></a><a name="abs">**FloatQuant**</a>

Calculates the [minifloat-quantized](https://arxiv.org/abs/2311.12359) values of one input data (Tensor<T>) and produces one output data (Tensor<T>).
Additionally, takes four floats as input, which define the scale, exponent bitwidth, mantissa bitwidth and exponent bias of the quantization,
all of which may be scalars or tensors with shapes broadcastable to the shape of the input data tensor. This can be used to
control the granularity of the quantization. For instance, a scalar scale operand implies per-tensor scaling, while a scale operand with
the same shape as the input data implies per-element scaling.
Specialized behaviors such as supporting infinity, NaN and subnormals are controlled by the attributes of the node.

Note: This operator is not intended for integer quantization, for this purpose the `IntQuant` custom op exists.

#### Version

This operator is not part of the ONNX standard and is not currently versioned.

#### Attributes

<dl>
<dt><tt>has_infinity</tt> : int (default is 0)</dt>
<dd>Integer value interpreted as boolean, defines whether the representation  supports infinity values. The ability to represent infinity values will   decrease the representable numerical range.</dd>
  
<dt><tt>has_nan</tt> : int (default is 0)</dt>
<dd>Integer value interpreted as boolean, defines whether the representation  supports not-a-number (NaN) values. The ability to represent NaN values will   decrease the representable numerical range.</dd>

<dt><tt>has_subnormal</tt> : int (default is 1)</dt>
<dd>Integer value interpreted as boolean, defines whether the representation  supports subnormal values. Subnormal values have an exponent value of 0 and are interpreted to have a leading significand digit of zero rather than one. Supporting subnormals will increase the complexity of the required arithmetic datapath.</dd>

<dt><tt>saturation</tt> : int (default is 1)</dt>
<dd>Integer value interpreted as boolean, defines whether the representation  will saturate during arithmetic.</dd>

<dt><tt>max_val</tt> : float (default is 0.0)</dt>
<dd>Maximum possible representable value, which is part of the quantization equation. If specified to be 0.0, the implementation is responsible for computing the maximum possible representable value. Otherwise, this specified value will be used.</dd>

<dt><tt>rounding_mode</tt> : string (default is "ROUND")</dt>
<dd>Defines how rounding should be applied during quantization. Currently available modes are: "ROUND", "CEIL" and "FLOOR". Here "ROUND" implies a round-to-even operation. Lowercase variants for the rounding mode string are also supported: "round", "ceil", "floor".</dd>

</dl>

#### Inputs

<dl>
<dt><tt>X</tt> (differentiable) : tensor(float32)</dt>
<dd>input tensor to quantize</dd>
<dt><tt>scale</tt> : float32, tensor(float32)</dt>
<dd>The scale factor, either as a global scalar or with a broadcastable shape matching the number of dimensions of the X tensor</dd>
<dt><tt>exponent_bitwidth</tt> : int32, float32</dt>
<dd>The number of bits for the exponent used by the quantization, must be a positive integer. If float32 dtype is used for convenience, it must still represent an positive integer number of bits.</dd>
<dt><tt>mantissa_bitwidth</tt> : int32, float32</dt>
<dd>The number of bits for the mantissa used by the quantization, must be a positive integer. If float32 dtype is used for convenience, it must still represent an positive integer number of bits.</dd>
<dt><tt>exponent_bias</tt> : int32, float32</dt>
<dd>The exponent bias used by the quantization, must be a positive integer. If float32 dtype is used for convenience, it must still represent an positive integer number of bits.</dd>
</dl>


#### Outputs

<dl>
<dt><tt>Y</tt> (differentiable) : tensor(float32)</dt>
<dd>Output tensor</dd>
</dl>

#### Examples
TODO


#### Sample Implementation
TODO
