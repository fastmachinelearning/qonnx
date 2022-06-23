### <a name="BipolarQuant"></a><a name="abs">**BipolarQuant**</a>

Calculates the binary quantized values of one input data (Tensor<T>) and produces one output data (Tensor<T>).
Additionally, takes one float as input, which define the scaling.

#### Version

This operator is not part of the ONNX standard and is not currently versioned.

#### Attributes

<dl>
</dl>

#### Inputs

<dl>
<dt><tt>X</tt> (differentiable) : tensor(float32)</dt>
<dd>input tensor to quantize</dd>
<dt><tt>scale</tt> : float32</dt>
<dd>The scale factor</dd>
</dl>


#### Outputs

<dl>
<dt><tt>Y</tt> (differentiable) : tensor(float32)</dt>
<dd>Output tensor</dd>
</dl>


#### Examples
<details>
<summary>BipolarQuant</summary>

```python
from onnx import helper
import numpy as np

# Define node settings and input
x = np.random.randn(100).astype(np.float32)*10.
scale = np.array(1.)

# Create node
node = helper.make_node(
    'BipolarQuant',
    domain='finn.custom_op.general',
    inputs=['x', 'scale'],
    outputs=['y'],
)

# Execute the same settings with the reference implementation (quant)
# See the sample implementation for more details on quant.
output_ref = binary_quant(x, scale)

# Execute node and compare
expect(node, inputs=[x, scale], outputs=[output_ref], name='test_binary_quant')

```

</details>


#### Sample Implementation

<details>
<summary>BipolarQuant</summary>

```python
# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

def binary_quant(inp_tensor, scale):
    # Quantizing
    y_int = inp_tensor
    y_ones = np.ones(y_int.shape, dtype=y_int.dtype)
    y_int = np.where(y_int >= 0.0, y_ones, -y_ones)
    # Scaling
    out_tensor = y_int * scale

    return out_tensor

```

</details>
