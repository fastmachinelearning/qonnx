import pytest

import onnx
from numpy.testing import assert_allclose
from qkeras import QActivation, QConv2D, QDense, binary, quantized_bits, quantized_relu, ternary
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model

import qonnx


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

    onnx_model, external_storage = qonnx.converters.from_keras(model)
    assert external_storage is None
    onnx.save(onnx_model, "model.onnx")

    # TODO add some useful test with the model


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

    onnx_model, external_storage = qonnx.converters.from_keras(model)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    assert external_storage is None
    onnx.save(onnx_model, "model.onnx")


def test_keras_dense_conversion():
    x = x_in = Input((15,), name="input")
    x = Dense(10, kernel_initializer="ones", bias_initializer="ones", name="dense1")(x)
    x = Activation("relu", name="act0_m")(x)
    x = Dense(10, kernel_initializer="ones", bias_initializer="ones", activation="relu", name="dense2")(x)
    x = Dense(10, kernel_initializer="ones", bias_initializer="ones", name="dense3")(x)
    x = Activation("softmax", name="softmax")(x)

    model = Model(inputs=[x_in], outputs=[x])

    onnx_model, external_storage = qonnx.converters.from_keras(model)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    assert external_storage is None
    onnx.save(onnx_model, "model.onnx")


if __name__ == "__main__":
    pytest.main([__file__])
