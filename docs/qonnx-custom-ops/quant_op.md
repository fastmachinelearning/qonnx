### <a name="Quant"></a><a name="abs">**Quant**</a>

Calculates the quantized values of one input data (Tensor<T>) and produces one output data (Tensor<T>).
Additionally, takes three floats as input, which define the scale, zero-point and bit-width of the quantization.
The attributes narrow and signed define how the bits of the quantization are interpreted, while the attribute
rounding_mode defines how quantized values are rounded.

Note: This operator does not work for binary or bipolar quantization, for this purpose the simpler BipolarQuant node exists.

#### Version

This operator is not part of the ONNX standard and is not currently versioned.

#### Attributes

<dl>
<dt><tt>signed</tt> : int (default is 1)</dt>
<dd>Defines if the quantization includes a signed bit. E.g. at 8b unsigned=[0, 255] vs signed=[-128, 127].</dd>
<dt><tt>narrow</tt> : int (default is 0)</dt>
<dd>Defines if the value range should be interpreted as narrow, when signed=1. E.g. at 8b regular=[-128, 127] vs narrow=[-127, 127].</dd>
<dt><tt>rounding_mode</tt> : string (default is "ROUND")</dt>
<dd>Defines how rounding should be applied during quantization. Currently available modes are: "ROUND", "CEIL" and "FLOOR". Here "ROUND" implies a round-to-even operation. Lowercase variants for the rounding mode string are also supported: "round", "ceil", "floor".</dd>
</dl>

#### Inputs

<dl>
<dt><tt>X</tt> (differentiable) : tensor(float32)</dt>
<dd>input tensor to quantize</dd>
<dt><tt>scale</tt> : float32</dt>
<dd>The scale factor</dd>
<dt><tt>zeropt</tt> : float32</dt>
<dd>The zero-point</dd>
<dt><tt>bitwidth</tt> : int32</dt>
<dd>The number of bits used by the quantization</dd>
</dl>


#### Outputs

<dl>
<dt><tt>Y</tt> (differentiable) : tensor(float32)</dt>
<dd>Output tensor</dd>
</dl>


#### Examples
<details>
<summary>Quant</summary>

```python
from onnx import helper
import numpy as np

# Define node settings and input
x = np.random.randn(100).astype(np.float32)*10.
scale = np.array(1.)
zeropt = np.array(0.)
bitwidth = np.array(4)
signed = 1
narrow = 0
rounding_mode = "ROUND"

# Create node
node = helper.make_node(
    'Quant',
    domain='finn.custom_op.general',
    inputs=['x', 'scale', 'zeropt', 'bitwidth'],
    outputs=['y'],
    narrow=narrow,
    signed=signed,
    rounding_mode=rounding_mode,
)

# Execute the same settings with the reference implementation (quant)
# See the sample implementation for more details on quant.
output_ref = quant(x, scale, zeropt, bitwidth, signed, narrow, rounding_mode)

# Execute node and compare
expect(node, inputs=[x, scale, zeropt, bitwidth], outputs=[output_ref], name='test_quant')

```

</details>


#### Sample Implementation

<details>
<summary>Quant</summary>

```python
# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

def quant(inp_tensor, scale, zeropt, bitwidth, signed, narrow, rounding_mode):
    # Port of IntQuant class from Brevitas: https://bit.ly/2S6qvZJ
    # Scaling
    y_int = inp_tensor / scale
    y_int = y_int + zeropt
    # Clamping
    min_int_val = min_int(signed, narrow, bitwidth)
    max_int_val = max_int(signed, narrow, bitwidth)
    y_int = np.where(y_int > max_int_val, max_int_val.astype(y_int.dtype), y_int)
    y_int = np.where(y_int < min_int_val, min_int_val.astype(y_int.dtype), y_int)
    # Rounding
    rounding_fx = resolve_rounding_mode(rounding_mode)
    y_int = rounding_fx(y_int)

    # Re-scaling
    out_tensor = y_int - zeropt
    out_tensor = out_tensor * scale

    return out_tensor

def min_int(signed: bool, narrow_range: bool, bit_width: int) -> int:
    """Compute the minimum integer representable by a given number of bits.
    Args:
        signed (bool): Indicates whether the represented integer is signed or not.
        narrow_range (bool): Indicates whether to narrow the minimum value
        represented by 1.
        bit_width (int): Number of bits available for the representation.
    Returns:
        int: Maximum unsigned integer that can be represented according to
        the input arguments.
    Examples:
        >>> min_int(signed=True, narrow_range=True, bit_width=8)
        int(-127)
        >>> min_int(signed=False, narrow_range=True, bit_width=8)
        int(0)
        >>> min_int(signed=True, narrow_range=False, bit_width=8)
        int(-128)
        >>> min_int(signed=False, narrow_range=False, bit_width=8)
        int(0)
    """
    if signed and narrow_range:
        value = -(2 ** (bit_width - 1)) + 1
    elif signed and not narrow_range:
        value = -(2 ** (bit_width - 1))
    else:
        value = 0 * bit_width
    return value


def max_int(signed: bool, narrow_range: bool, bit_width: int) -> int:
    """Compute the maximum integer representable by a given number of bits.
    Args:
        signed (bool): Indicates whether the represented integer is signed or not.
        narrow_range (bool): Indicates whether to narrow the maximum unsigned value
        represented by 1.
        bit_width (int): Number of bits available for the representation.
    Returns:
        Tensor: Maximum integer that can be represented according to
        the input arguments.
    Examples:
        >>> max_int(signed=True, narrow_range=True, bit_width=8)
        int(127)
        >>> max_int(signed=False, narrow_range=True, bit_width=8)
        int(254)
        >>> max_int(signed=True, narrow_range=False, bit_width=8)
        int(127)
        >>> max_int(signed=False, narrow_range=False, bit_width=8)
        int(255)
    """
    if not signed and not narrow_range:
        value = (2 ** bit_width) - 1
    elif not signed and narrow_range:
        value = (2 ** bit_width) - 2
    else:
        value = (2 ** (bit_width - 1)) - 1
    return value

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
