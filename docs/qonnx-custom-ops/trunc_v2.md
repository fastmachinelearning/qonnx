### <a name="Trunc"></a><a name="abs">**Trunc**</a>

Truncates the values of one input data (Tensor<T>) at a specified bitwidth and produces one output data (Tensor<T>).
Additionally, takes four float tensors as input, which define the scale, zero-point, input bit-width and output bit-width of the quantization.
The attribute rounding_mode defines how truncated values are rounded.

#### Version

This operator is not part of the ONNX standard.
The description of this operator in this document corresponds to `qonnx.custom_ops.general` opset version 2.

#### Attributes

<dl>
<dt><tt>rounding_mode</tt> : string (default is "FLOOR")</dt>
<dd>Defines how rounding should be applied during truncation. Currently available modes are: "ROUND", "CEIL" and "FLOOR". Here "ROUND" implies a round-to-even operation. Lowercase variants for the rounding mode string are also supported: "round", "ceil", "floor".</dd>
<dt><tt>signed</tt> : int (default is 1)</dt>
<dd>Defines if the quantization includes a signed bit. E.g. at 8b unsigned=[0, 255] vs signed=[-128, 127].</dd>
<dt><tt>narrow</tt> : int (default is 0)</dt>
<dd>Defines if the value range should be interpreted as narrow, when signed=1. E.g. at 8b regular=[-128, 127] vs narrow=[-127, 127].</dd>
</dl>

#### Inputs

<dl>
<dt><tt>X</tt> (differentiable) : tensor(float32)</dt>
<dd>input tensor to truncate</dd>
<dt><tt>scale</tt> : float32</dt>
<dd>The scale factor at the input of the truncation</dd>
<dt><tt>zeropt</tt> : float32</dt>
<dd>The zero-point at the input of the truncation</dd>
<dt><tt>in_bitwidth</tt> : int32</dt>
<dd>The number of bits used at the input of the truncation</dd>
<dt><tt>out_scale</tt> : float32</dt>
<dd>The scale factor of the output of the truncation</dd>
<dt><tt>out_bitwidth</tt> : int32</dt>
<dd>The number of bits used at the output of the truncation</dd>
</dl>


#### Outputs

<dl>
<dt><tt>Y</tt> (differentiable) : tensor(float32)</dt>
<dd>Output tensor</dd>
</dl>


#### Examples
<details>
<summary>Trunc</summary>

```python
from onnx import helper
import numpy as np

# Define node settings and input
x = np.random.randn(100).astype(np.float32)*10.
scale = np.array(1.)
zeropt = np.array(0.)
in_bitwidth = np.array(10)
out_bitwidth = np.array(4)
rounding_mode = "ROUND"

# Create node
node = helper.make_node(
    'Trunc',
    domain='finn.custom_op.general',
    inputs=['x', 'scale', 'zeropt', 'in_bitwidth', 'out_bitwidth'],
    outputs=['y'],
    rounding_mode=rounding_mode,
)

# Execute the same settings with the reference implementation (trunc)
# See the sample implementation for more details on trunc.
output_ref = trunc(inp_tensor, scale, zeropt, in_bitwidth, out_bitwidth, rounding_mode)

# Execute node and compare
expect(node, inputs=[x, scale, zeropt, bitwidth], outputs=[output_ref], name='test_trunc')

```

</details>


#### Sample Implementation

<details>
<summary>Trunc</summary>

```python
# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

def trunc(inp_tensor, scale, zeropt, input_bit_width, narrow, signed, output_scale, output_bit_width, rounding_mode):

    # Scaling
    y = inp_tensor / scale
    y = y + zeropt
    # Rounding
    y = np.round(y)
    # Rescale
    trunc_scale = 2 ** np.round(
        np.log2(output_scale / scale)
    )  # Trunc scale should be a power-of-two - ensure that is the case
    y = y / trunc_scale

    # Clamping
    min_int_val = min_int(signed, narrow, output_bit_width)
    max_int_val = max_int(signed, narrow, output_bit_width)
    y = np.where(y > max_int_val, max_int_val.astype(y.dtype), y)
    y = np.where(y < min_int_val, min_int_val.astype(y.dtype), y)
    # To int (truncate)
    rounding_fx = resolve_rounding_mode(rounding_mode)
    y = rounding_fx(y)

    # Rescale
    output_zeropt = zeropt / trunc_scale  # Rescale zero-point
    y = y - output_zeropt
    y = y * output_scale

    return y

def resolve_rounding_mode(mode_string):
    """Resolve the rounding mode string of Quant and Trunc ops
    to the corresponding numpy functions."""
    if mode_string == "ROUND":
        return np.round
    elif mode_string == "CEIL":
        return np.ceil
    elif mode_string == "FLOOR":
        return np.floor
    else:
        raise ValueError(f"Could not resolve rounding mode called: {mode_string}")

```

</details>
