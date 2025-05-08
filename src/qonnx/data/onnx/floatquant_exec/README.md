Sample model for testing FloatQuant execution with exported graph. Generated with Brevitas (Commit: 904bbeaafaae5adb5c965af8d6b95120b7d1589a), using the code below.

```python
# Create the Brevitas model
brevitas_model = qnn.QuantLinear(
    3, 16, weight_quant=Fp8e4m3OCPWeightPerTensorFloat, input_quant=Fp8e4m3OCPActPerTensorFloat
)
# important to put into eval mode before export
brevitas_model.eval()
# Export the Brevitas model to QONNX format
export_path = "qonnx_act_weight_fp8.onnx"
input_shape = (1, 3)  # Example input shape, adjust as needed
dummy_input = torch.randn(input_shape)
export_qonnx(brevitas_model, dummy_input, export_path)

input_values = np.random.rand(*input_shape).astype(np.float32)
np.save("input.npy", input_values)

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach().value.numpy()

    return hook

brevitas_model.input_quant.register_forward_hook(get_activation("input_quant"))
brevitas_model.weight_quant.register_forward_hook(get_activation("weight_quant"))

# Get the output from the Brevitas model
brevitas_output = brevitas_model(torch.tensor(input_values)).detach().numpy()
np.save("output.npy", brevitas_output)
np.savez("activation.npz", **activation)
```
