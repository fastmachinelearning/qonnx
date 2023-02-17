### <a name="Qkeras to Qonnx"></a><a name="abs">**Qkeras to Qonnx**</a>

The converter works by (1) strip QKeras model of quantization attributes and store in a dictionary; (2) convert (as if plain Keras model) using tf2onnx; (3) Insert “Quant” nodes at appropriate locations based on a dictionary of quantization attributes.

The current version has few issues given how tf2onnx inserts the quant nodes. These problems have suitable workarounds detailed below.

### Quantized-Relu
The quantized-relu quantization inserts a redundant quantization node when used as output activation of Dense/Conv2D layer.

Workaround: Only use quantized-relu activation in a seperate QActivation layers.

<img src="https://user-images.githubusercontent.com/31563706/209125992-e03078e4-ec92-4796-982f-2a31292687d6.png"  width="300" height="500">

### Quantized-Bits
The quantized-bits quantization node is not added to the model when used in QActivation layers.

Workaround: Use quantized-bits only at the output of a Dense/Conv2D layers.

<img src="https://user-images.githubusercontent.com/31563706/209126623-9956ecea-748e-4d7c-930c-d46d06ab6a14.png"  width="350" height="350">

### Ternary Quantization
A threshold of 0.5 must be used when using ternary quantization.
(This is sometimes unstable even with t=0.5)
