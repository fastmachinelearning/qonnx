import pytest

import onnx

# from numpy.testing import assert_allclose
from qkeras import QActivation, QConv2D, QDense, binary, quantized_bits, quantized_relu, ternary

# from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
import tensorflow as tf

import numpy as np

import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_shapes import InferShapes

import qonnx
from qonnx.util.to_channels_last import to_channels_last

import onnxruntime as ort
import tf2onnx

def test_qkeras_conversion():
    x = x_in = Input((28, 28, 1), name="input")
    x = QConv2D(
        32,
        (2, 2),
        strides=(2, 2),
        kernel_quantizer=binary(alpha=1.0),
        bias_quantizer=quantized_bits(4, 0, 1),
        activation=quantized_bits(6, 2, 1, alpha=1.0),
        bias_initializer="ones",  # If bias tensor is all zeros, tf2onnx removes it
        name="conv2d_0_m",
    )(x)
    x = QActivation("quantized_relu(6,2,1)", name="act0_m")(x)
    x = QConv2D(
        64,
        (3, 3),
        strides=(2, 2),
        kernel_quantizer=ternary(alpha=1.0),
        bias_quantizer=quantized_bits(4, 0, 1),
        bias_initializer="ones",
        name="conv2d_1_m",
        activation=quantized_relu(6, 3, 1),
    )(x)
    x = QConv2D(
        64,
        (2, 2),
        strides=(2, 2),
        kernel_quantizer=quantized_bits(6, 2, 1, alpha=1.0),
        use_bias=False,  # Lets try this one without bias to see if that trips up the converter
        name="conv2d_2_m",
    )(x)
    x = QActivation("quantized_relu(6,4,1)", name="act2_m")(x)
    x = Flatten(name="flatten")(x)
    x = QDense(
        10,
        kernel_quantizer=quantized_bits(6, 2, 1, alpha=1.0),
        bias_quantizer=quantized_bits(4, 0, 1),
        bias_initializer="ones",
        name="dense",
    )(x)
    x = Activation("softmax", name="softmax")(x)

    model = Model(inputs=[x_in], outputs=[x])

    x_test = np.random.uniform(low=-1.0, high=1.0, size=(1, 28, 28, 1)).astype(dtype=np.float32)
    y_qkeras = model.predict(x_test)

    onnx_model, external_storage = qonnx.converters.from_keras(model, "test_qkeras_conversion", opset=9)
    assert external_storage is None
    onnx.save(onnx_model, "model_test_qkeras_conversion.onnx")

    onnx_model = ModelWrapper("model_test_qkeras_conversion.onnx")
    onnx_model = onnx_model.transform(InferShapes())

    idict = {onnx_model.graph.input[0].name: x_test}
    odict = oxe.execute_onnx(onnx_model, idict, True)
    y_qonnx = odict[onnx_model.graph.output[0].name]

    np.testing.assert_array_equal(y_qkeras, y_qonnx)

quantizers = [(quantized_bits(8,4,1,alpha=1), quantized_bits(8,4,1,alpha=1)),
              (quantized_bits(8,8,1,alpha=1), quantized_bits(8,8,1,alpha=1)),
              (quantized_bits(4,4,1,alpha=1), quantized_bits(8,8,1,alpha=1)),
              (quantized_bits(4,0,1,alpha=1), quantized_bits(8,0,1,alpha=1)),
              (quantized_bits(2,2,1,alpha=1), quantized_bits(2,2,1,alpha=1)),
              (ternary(), quantized_bits(4,4)),
              (binary(), quantized_bits(4,4))]
quantizers_ids = list(range(len(quantizers)))
@pytest.mark.parametrize('quantizers', quantizers, ids=quantizers_ids)
def test_qkeras_qconv2d(quantizers, request):
    kq, bq = quantizers
    print(kq, bq)
    x = x_in = Input((28, 28, 3), name="input")
    x = QConv2D(
        32,
        (2, 2),
        strides=(2, 2),
        kernel_quantizer=kq,
        bias_quantizer=bq,
        bias_initializer="ones",  # If bias tensor is all zeros, tf2onnx removes it
        name="conv2d_0",
    )(x)
    model = Model(inputs=[x_in], outputs=[x])

    x_test = np.random.uniform(low=-1.0, high=1.0, size=(1, 28, 28, 3)).astype(dtype=np.float32)
    y_qkeras = model.predict(x_test)
    
    onnx_model, external_storage = qonnx.converters.from_keras(model, "test_qkeras_conversion", opset=9)
    assert external_storage is None
    model_path = f"model_test_qkeras_qconv2d_{request.node.callspec.id}.onnx"
    onnx.save(onnx_model, model_path)
 
    onnx_model = ModelWrapper(model_path)
    onnx_model = onnx_model.transform(InferShapes())

    idict = {onnx_model.graph.input[0].name: x_test}
    odict = oxe.execute_onnx(onnx_model, idict, True)
    y_qonnx = odict[onnx_model.graph.output[0].name]

    np.testing.assert_array_equal(y_qkeras, y_qonnx)

quantizers = [(quantized_bits(8,4,1,alpha=1), quantized_bits(8,4,1,alpha=1)),
              (quantized_bits(8,8,1,alpha=1), quantized_bits(8,8,1,alpha=1)),
              (quantized_bits(4,4,1,alpha=1), quantized_bits(8,8,1,alpha=1)),
              (quantized_bits(4,0,1,alpha=1), quantized_bits(8,0,1,alpha=1)),
              (quantized_bits(2,2,1,alpha=1), quantized_bits(2,2,1,alpha=1)),
              (ternary(), quantized_bits(4,4)),
              (binary(), quantized_bits(4,4))]
quantizers_ids = list(range(len(quantizers)))
@pytest.mark.parametrize('quantizers', quantizers, ids=quantizers_ids)
def test_qkeras_qdense(quantizers, request):
    kq, bq = quantizers
    print(kq, bq)
    x = x_in = Input((16), name="input")
    x = QDense(
        32,
        kernel_quantizer=kq,
        bias_quantizer=bq,
        bias_initializer="ones",  # If bias tensor is all zeros, tf2onnx removes it
        name="dense_0",
    )(x)
    model = Model(inputs=[x_in], outputs=[x])
    x_test = np.random.uniform(low=-1.0, high=1.0, size=(1, 16)).astype(dtype=np.float32)
    y_qkeras = model.predict(x_test)
    
    onnx_model, external_storage = qonnx.converters.from_keras(model, "test_qkeras_conversion", opset=9)
    assert external_storage is None
    model_path = f"model_test_qkeras_qdense_{request.node.callspec.id}.onnx"
    onnx.save(onnx_model, model_path)
    model.save(model_path.replace('onnx', 'h5'))

    onnx_model = ModelWrapper(model_path)
    onnx_model = onnx_model.transform(InferShapes())

    idict = {onnx_model.graph.input[0].name: x_test}
    odict = oxe.execute_onnx(onnx_model, idict, True)
    y_qonnx = odict[onnx_model.graph.output[0].name]

    np.testing.assert_array_equal(y_qkeras, y_qonnx)

def test_keras_conversion():
    x = x_in = Input((28, 28, 1), name="input")
    x = Conv2D(32, (2, 2), strides=(2, 2), name="conv2d_0_m")(x)
    x = Activation("relu", name="act0_m")(x)
    x = Conv2D(64, (3, 3), strides=(2, 2), name="conv2d_1_m", activation="relu")(x)
    x = Conv2D(64, (2, 2), strides=(2, 2), name="conv2d_2_m")(x)
    x = Activation("relu", name="act2_m")(x)
    x = Flatten(name="flatten")(x)
    x = Dense(10, bias_initializer="ones", name="dense")(x)
    x = Activation("softmax", name="softmax")(x)

    model = Model(inputs=[x_in], outputs=[x])

    x_test = np.random.uniform(low=-1.0, high=1.0, size=(1, 28, 28, 1)).astype(dtype=np.float32)
    y_qkeras = model.predict(x_test)

    onnx_model, external_storage = qonnx.converters.from_keras(model, "test_keras_dense_conversion")
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    assert external_storage is None
    onnx.save(onnx_model, "model_test_keras_conversion.onnx")

    onnx_model = ModelWrapper("model_test_keras_conversion.onnx")
    onnx_model = onnx_model.transform(InferShapes())

    idict = {onnx_model.graph.input[0].name: x_test}
    odict = oxe.execute_onnx(onnx_model, idict, True)
    y_qonnx = odict[onnx_model.graph.output[0].name]

    np.testing.assert_allclose(y_qkeras, y_qonnx, rtol=1e-5, atol=1e-5)


def test_keras_dense_conversion():
    x = x_in = Input((15), name="input")
    x = Dense(10, kernel_initializer="ones", bias_initializer="ones", name="dense1")(x)
    x = Activation("relu", name="act0_m")(x)
    x = Dense(10, kernel_initializer="ones", bias_initializer="ones", activation="relu", name="dense2")(x)
    x = Dense(10, kernel_initializer="ones", bias_initializer="ones", name="dense3")(x)
    x = Activation("softmax", name="softmax")(x)

    model = Model(inputs=[x_in], outputs=[x])

    x_test = np.random.uniform(low=-1.0, high=1.0, size=(1, 15)).astype(dtype=np.float32)
    y_qkeras = model.predict(x_test)

    onnx_model, external_storage = qonnx.converters.from_keras(model, "test_keras_dense_conversion")
    assert external_storage is None
    onnx.save(onnx_model, "model_test_keras_dense_conversion.onnx")

    onnx_model = ModelWrapper("model_test_keras_dense_conversion.onnx")
    onnx_model = onnx_model.transform(InferShapes())

    idict = {onnx_model.graph.input[0].name: x_test}
    odict = oxe.execute_onnx(onnx_model, idict, True)
    y_qonnx = odict[onnx_model.graph.output[0].name]

    np.testing.assert_allclose(y_qkeras, y_qonnx, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__])
