### <a name="Trunc"></a><a name="abs">**Trunc**</a>

Truncates the values of one input data (Tensor<T>) at a specified bitwidth and produces one output data (Tensor<T>).
Additionally, takes four float tensors as input, which define the scale, zero-point, input bit-width and output bit-width of the quantization.
The attribute rounding_mode defines how truncated values are rounded.

#### Version

This operator is not part of the ONNX standard and is not currently versioned.

#### Attributes

<dl>
<dt><tt>rounding_mode</tt> : string (default is "FLOOR")</dt>
<dd>Defines how rounding should be applied during truncation. Currently available modes are: "ROUND", "CEIL" and "FLOOR". Here "ROUND" implies a round-to-even operation. Lowercase variants for the rounding mode string are also supported: "round", "ceil", "floor".</dd>
</dl>

#### Inputs

<dl>
<dt><tt>X</tt> (differentiable) : tensor(float32)</dt>
<dd>input tensor to truncate</dd>
<dt><tt>scale</tt> : float32</dt>
<dd>The scale factor</dd>
<dt><tt>zeropt</tt> : float32</dt>
<dd>The zero-point</dd>
<dt><tt>in_bitwidth</tt> : int32</dt>
<dd>The number of bits used at the input of the truncation</dd>
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

def trunc(inp_tensor, scale, zeropt, input_bit_width, output_bit_width, rounding_mode):
    # Port of TruncIntQuant class from Brevitas: https://bit.ly/3wzIpTR

    # Scaling
    y = inp_tensor / scale
    y = y + zeropt
    # Rounding
    y = np.round(y)
    # Truncate
    trunc_bit_width = input_bit_width - output_bit_width
    trunc_scale = 2.0 ** trunc_bit_width
    y = y / trunc_scale

    # To int
    rounding_fx = resolve_rounding_mode(rounding_mode)
    y = rounding_fx(y)

    # Rescale
    y = y - zeropt
    y = y * scale

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
