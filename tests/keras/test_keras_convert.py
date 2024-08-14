import pytest

import numpy as np
import onnx
import os
import tensorflow as tf
tf.keras.utils.set_random_seed(42)
np.random.seed(42)
from qkeras import QActivation, QConv2D, QDense, binary, quantized_bits, quantized_relu, ternary
from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model

import qonnx.core.onnx_exec as oxe
from qonnx.converters import from_keras
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes

act_quantizers = [
    quantized_bits(8, 4, 0, alpha=1),
    quantized_bits(8, 4, 1, alpha=1),
    quantized_bits(8, 8, 0, alpha=1),
    quantized_bits(8, 8, 1, alpha=1),
    quantized_bits(4, 4, 0, alpha=1),
    quantized_bits(4, 4, 1, alpha=1),
    quantized_bits(4, 0, 0, alpha=1),
    quantized_bits(4, 0, 1, alpha=1),
    quantized_bits(4, 2, 0, alpha=1),
    quantized_bits(2, 2, 1, alpha=1),
    quantized_bits(2, 1, 1, alpha=1),
    #   ternary(alpha=1, threshold=0.5), Not stable
    binary(alpha=1),
]
act_quantizers_ids = list(range(len(act_quantizers)))

act_quantizers_relu = [
    quantized_relu(8),
    quantized_relu(8, 4),
    quantized_relu(4),
    quantized_relu(4, 4),
    quantized_relu(6, 2),
]
act_quantizers_relu_ids = list(range(len(act_quantizers_relu)))


@pytest.mark.parametrize("quantizer", act_quantizers_relu, ids=act_quantizers_relu_ids)
def test_qkeras_qactivation(quantizer, request):
    x = x_in = Input((16), name="input")
    x = QActivation(activation=quantizer, name="act_0")(x)
    model = Model(inputs=[x_in], outputs=[x])
    x_test = np.random.uniform(low=-1.0, high=1.0, size=(1, 16)).astype(dtype=np.float32)
    y_qkeras = model.predict(x_test)

    onnx_model, external_storage = from_keras(model, "test_qkeras_conversion", opset=9)
    assert external_storage is None
    model_path = f"model_test_qkeras_qactivation_{request.node.callspec.id}.onnx"
    onnx.save(onnx_model, model_path)

    onnx_model = ModelWrapper(model_path)
    onnx_model = onnx_model.transform(InferShapes())

    idict = {onnx_model.graph.input[0].name: x_test}
    odict = oxe.execute_onnx(onnx_model, idict, True)
    y_qonnx = odict[onnx_model.graph.output[0].name]

    np.testing.assert_allclose(y_qkeras, y_qonnx, rtol=1e-5, atol=1e-5)
    os.remove(model_path)

@pytest.mark.parametrize("quantizer", [
    quantized_relu(bits=4, integer=4), 
    quantized_bits(bits=4, integer=4, keep_negative=False, alpha=1),
    ])
def test_qkeras_quantizers_rounding_modes(quantizer, request):  
    x = x_in = Input((10,), name="input")
    x = QActivation(activation=quantizer)(x)
    model = Model(inputs=[x_in], outputs=[x])
    model.compile()
    
    onnx_model, _ = from_keras(model)
    model_path = f"model_test_qkeras_quantizers_rounding_modes_{request.node.callspec.id}.onnx"
    onnx.save(onnx_model, model_path)
    onnx_model = ModelWrapper(onnx_model)

    x_test = np.array([[5.5, 2.5, 1.6, 1.1, 1.0, -1.0, -1.1, -1.6, -2.5, -5.5]]).astype(np.float32)
    idict = {onnx_model.graph.input[0].name: x_test}
    odict = oxe.execute_onnx(onnx_model, idict, True)
    y_qonnx = odict[onnx_model.graph.output[0].name]
    y_qkeras = model.predict(x_test)
    assert np.array_equal(y_qkeras, y_qonnx)
    os.remove(model_path)

@pytest.mark.parametrize("bias", [5.5, 2.5, 1.6, 1.1, 1.0, -1.0, -1.1, -1.6, -2.5, -5.5])
def test_qkeras_quantizers_autopo2_rounding_modes(bias, request):
    kq = bq = quantized_bits(4, 4, 1, alpha='auto_po2')
    # Initialize the kernel & bias to RandonUniform within the range of the quantizers
    x = x_in = Input((10), name="input")
    x = QDense(
        1,
        kernel_quantizer=kq,
        bias_quantizer=bq,
        kernel_initializer=tf.keras.initializers.Constant(1.0),
        bias_initializer=tf.keras.initializers.Constant(bias),
        name="dense_0",
    )(x)
    model = Model(inputs=[x_in], outputs=[x])
    x_test = np.random.uniform(low=-1.0, high=1.0, size=(1, 10)).astype(dtype=np.float32)
    _ = model.predict(x_test, verbose=0)

    onnx_model, _ = from_keras(model)
    model_path = f"model_test_qkeras_quantizers_auto_rounding_modes_{request.node.callspec.id}.onnx"
    onnx.save(onnx_model, model_path)
    onnx_model = ModelWrapper(onnx_model)

    x_test = np.zeros(shape=(1, 10), dtype=np.float32)
    idict = {onnx_model.graph.input[0].name: x_test}
    odict = oxe.execute_onnx(onnx_model, idict, True)
    y_qonnx = odict[onnx_model.graph.output[0].name]
    y_qkeras = model.predict(x_test, verbose=0)
    assert np.array_equal(y_qkeras, y_qonnx)
    os.remove(model_path)

# pairs of quantizers for kernel and bias
kb_quantizers = [
    (quantized_bits(8, 4, 0, alpha=1), quantized_bits(8, 4, 0, alpha=1)),
    (quantized_bits(8, 4, 1, alpha=1), quantized_bits(8, 4, 1, alpha=1)),
    (quantized_bits(8, 8, 0, alpha=1), quantized_bits(8, 8, 0, alpha=1)),
    (quantized_bits(8, 8, 1, alpha=1), quantized_bits(8, 8, 1, alpha=1)),
    (quantized_bits(4, 4, 0, alpha=1), quantized_bits(8, 8, 0, alpha=1)),
    (quantized_bits(4, 4, 1, alpha=1), quantized_bits(8, 8, 1, alpha=1)),
    (quantized_bits(4, 0, 0, alpha=1), quantized_bits(8, 0, 0, alpha=1)),
    (quantized_bits(4, 0, 1, alpha=1), quantized_bits(8, 0, 1, alpha=1)),
    (quantized_bits(4, 2, 0, alpha=1), quantized_bits(8, 2, 0, alpha=1)),
    (quantized_bits(2, 2, 1, alpha=1), quantized_bits(2, 2, 1, alpha=1)),
    (quantized_bits(2, 1, 1, alpha=1), quantized_bits(2, 1, 1, alpha=1)),
    (ternary(alpha=1, threshold=0.5), quantized_bits(4, 4)),
    (binary(alpha=1), quantized_bits(4, 4)),
]
kb_quantizers_ids = list(range(len(kb_quantizers)))


def test_keras_conv2d_conversion():
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

    onnx_model, external_storage = from_keras(model, "test_keras_conv2d_conversion")
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    assert external_storage is None
    onnx.save(onnx_model, "model_test_keras_conv2d_conversion.onnx")

    onnx_model = ModelWrapper("model_test_keras_conv2d_conversion.onnx")
    onnx_model = onnx_model.transform(InferShapes())

    idict = {onnx_model.graph.input[0].name: x_test}
    odict = oxe.execute_onnx(onnx_model, idict, True)
    y_qonnx = odict[onnx_model.graph.output[0].name]

    np.testing.assert_allclose(y_qkeras, y_qonnx, rtol=1e-5, atol=1e-5)
    os.remove("model_test_keras_conv2d_conversion.onnx")


def test_keras_dense_conversion():
    ini = tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)
    x = x_in = Input((15), name="input")
    x = Dense(10, kernel_initializer=ini, bias_initializer=ini, name="dense1")(x)
    x = Activation("relu", name="act0_m")(x)
    x = Dense(10, kernel_initializer=ini, bias_initializer=ini, activation="relu", name="dense2")(x)
    x = Dense(10, kernel_initializer=ini, bias_initializer=ini, name="dense3")(x)
    x = Activation("softmax", name="softmax")(x)

    model = Model(inputs=[x_in], outputs=[x])

    x_test = np.random.uniform(low=-1.0, high=1.0, size=(1, 15)).astype(dtype=np.float32)
    y_qkeras = model.predict(x_test)

    onnx_model, external_storage = from_keras(model, "test_keras_dense_conversion")
    assert external_storage is None
    onnx.save(onnx_model, "model_test_keras_dense_conversion.onnx")

    onnx_model = ModelWrapper("model_test_keras_dense_conversion.onnx")
    onnx_model = onnx_model.transform(InferShapes())

    idict = {onnx_model.graph.input[0].name: x_test}
    odict = oxe.execute_onnx(onnx_model, idict, True)
    y_qonnx = odict[onnx_model.graph.output[0].name]

    np.testing.assert_allclose(y_qkeras, y_qonnx, rtol=1e-5, atol=1e-5)
    os.remove("model_test_keras_dense_conversion.onnx")


@pytest.mark.parametrize("quantizers", kb_quantizers, ids=kb_quantizers_ids)
def test_qkeras_qdense_1(quantizers, request):
    kq, bq = quantizers
    # Initialize the kernel & bias to RandonUniform within the range of the quantizers
    k_ini = tf.keras.initializers.RandomUniform(minval=kq.min(), maxval=kq.max())
    b_ini = tf.keras.initializers.RandomUniform(minval=bq.min(), maxval=bq.max())
    x = x_in = Input((16), name="input")
    x = QDense(
        32,
        kernel_quantizer=kq,
        bias_quantizer=bq,
        kernel_initializer=k_ini,
        bias_initializer=b_ini,
        name="dense_0",
    )(x)
    model = Model(inputs=[x_in], outputs=[x])
    x_test = np.random.uniform(low=-1.0, high=1.0, size=(1, 16)).astype(dtype=np.float32)
    y_qkeras = model.predict(x_test)

    onnx_model, external_storage = from_keras(model, "test_qkeras_conversion", opset=9)
    assert external_storage is None
    model_path = f"model_test_qkeras_qdense1_{request.node.callspec.id}.onnx"
    onnx.save(onnx_model, model_path)

    onnx_model = ModelWrapper(model_path)
    onnx_model = onnx_model.transform(InferShapes())

    idict = {onnx_model.graph.input[0].name: x_test}
    odict = oxe.execute_onnx(onnx_model, idict, True)
    y_qonnx = odict[onnx_model.graph.output[0].name]

    np.testing.assert_allclose(y_qkeras, y_qonnx, rtol=1e-4, atol=1e-4)
    os.remove(model_path)


@pytest.mark.parametrize("quantizers", kb_quantizers, ids=kb_quantizers_ids)
def test_qkeras_qdense_2(quantizers, request):
    kq, bq = quantizers
    # Initialize the kernel & bias to RandonUniform within the range of the quantizers
    k_ini = tf.keras.initializers.RandomUniform(minval=kq.min(), maxval=kq.max())
    b_ini = tf.keras.initializers.RandomUniform(minval=bq.min(), maxval=bq.max())
    x = x_in = Input((16), name="input")
    x = QDense(
        32,
        kernel_quantizer=kq,
        bias_quantizer=bq,
        kernel_initializer=k_ini,
        bias_initializer=b_ini,
        name="dense_0",
    )(x)
    x = QActivation("quantized_relu(6,2)", name="act1")(x)
    x = QDense(
        32,
        kernel_quantizer=kq,
        bias_quantizer=bq,
        kernel_initializer=k_ini,
        bias_initializer=b_ini,
        name="dense_1",
    )(x)
    model = Model(inputs=[x_in], outputs=[x])
    x_test = np.random.uniform(low=-1.0, high=1.0, size=(1, 16)).astype(dtype=np.float32)
    y_qkeras = model.predict(x_test)

    onnx_model, external_storage = from_keras(model, "test_qkeras_conversion", opset=9)
    assert external_storage is None
    model_path = f"model_test_qkeras_qdense2_{request.node.callspec.id}.onnx"
    onnx.save(onnx_model, model_path)

    onnx_model = ModelWrapper(model_path)
    onnx_model = onnx_model.transform(InferShapes())

    idict = {onnx_model.graph.input[0].name: x_test}
    odict = oxe.execute_onnx(onnx_model, idict, True)
    y_qonnx = odict[onnx_model.graph.output[0].name]
    np.testing.assert_allclose(y_qkeras, y_qonnx, rtol=1e-4, atol=1e-4)
    os.remove(model_path)


@pytest.mark.parametrize("quantizers", kb_quantizers, ids=kb_quantizers_ids)
def test_qkeras_qdense_3(quantizers, request):
    kq, bq = quantizers
    # Initialize the kernel & bias to RandonUniform within the range of the quantizers
    k_ini = tf.keras.initializers.RandomUniform(minval=kq.min(), maxval=kq.max())
    b_ini = tf.keras.initializers.RandomUniform(minval=bq.min(), maxval=bq.max())
    x = x_in = Input((16), name="input")
    x = QDense(
        32,
        kernel_quantizer=kq,
        bias_quantizer=bq,
        activation=quantized_bits(6, 2, 1, alpha=1.0),
        kernel_initializer=k_ini,
        bias_initializer=b_ini,
        name="dense_0",
    )(x)
    x = QActivation("quantized_relu(6,4,0)", name="act2_m")(x)
    x = QDense(
        10,
        kernel_quantizer=kq,
        bias_quantizer=bq,
        kernel_initializer=k_ini,
        bias_initializer=b_ini,
        name="dense_1",
    )(x)
    x = Activation("softmax", name="softmax")(x)
    model = Model(inputs=[x_in], outputs=[x])
    x_test = np.random.uniform(low=-1.0, high=1.0, size=(1, 16)).astype(dtype=np.float32)
    y_qkeras = model.predict(x_test)

    onnx_model, external_storage = from_keras(model, "test_qkeras_conversion", opset=9)
    assert external_storage is None
    model_path = f"model_test_qkeras_qdense3_{request.node.callspec.id}.onnx"
    onnx.save(onnx_model, model_path)

    onnx_model = ModelWrapper(model_path)
    onnx_model = onnx_model.transform(InferShapes())

    idict = {onnx_model.graph.input[0].name: x_test}
    odict = oxe.execute_onnx(onnx_model, idict, True)
    y_qonnx = odict[onnx_model.graph.output[0].name]
    np.testing.assert_allclose(y_qkeras, y_qonnx, rtol=1e-4, atol=1e-4)
    os.remove(model_path)


@pytest.mark.parametrize("quantizers", act_quantizers_relu, ids=act_quantizers_relu_ids)
def test_qkeras_qdense_4(quantizers, request):
    kq, bq = (quantized_bits(4, 0, 1, alpha=1), quantized_bits(8, 0, 1, alpha=1))
    # Initialize the kernel & bias to RandonUniform within the range of the quantizers
    k_ini = tf.keras.initializers.RandomUniform(minval=kq.min(), maxval=kq.max())
    b_ini = tf.keras.initializers.RandomUniform(minval=bq.min(), maxval=bq.max())
    x = x_in = Input((16), name="input")
    x = QDense(
        32,
        kernel_quantizer=kq,
        bias_quantizer=bq,
        activation=quantized_bits(8, 8, 1, alpha=1.0),
        kernel_initializer=k_ini,
        bias_initializer=b_ini,
        name="dense_0",
    )(x)
    x = QActivation(activation=quantizers, name="act_1")(x)
    x = QDense(
        32,
        kernel_quantizer=kq,
        bias_quantizer=bq,
        kernel_initializer=k_ini,
        bias_initializer=b_ini,
        name="dense_1",
    )(x)
    x = QActivation(activation=quantizers, name="act_2")(x)
    x = QDense(
        10,
        kernel_quantizer=kq,
        bias_quantizer=bq,
        activation=quantized_bits(4, 4, 0, alpha=1.0),
        use_bias=False,
        kernel_initializer=k_ini,
        bias_initializer=b_ini,
        name="dense_2",
    )(x)
    x = Activation("softmax", name="softmax")(x)
    model = Model(inputs=[x_in], outputs=[x])
    x_test = np.random.uniform(low=-1.0, high=1.0, size=(1, 16)).astype(dtype=np.float32)
    y_qkeras = model.predict(x_test)

    onnx_model, external_storage = from_keras(model, "test_qkeras_conversion", opset=9)
    assert external_storage is None
    model_path = f"model_test_qkeras_qdense4_{request.node.callspec.id}.onnx"
    onnx.save(onnx_model, model_path)

    onnx_model = ModelWrapper(model_path)
    onnx_model = onnx_model.transform(InferShapes())

    idict = {onnx_model.graph.input[0].name: x_test}
    odict = oxe.execute_onnx(onnx_model, idict, True)
    y_qonnx = odict[onnx_model.graph.output[0].name]
    np.testing.assert_allclose(y_qkeras, y_qonnx, rtol=1e-4, atol=1e-4)
    os.remove(model_path)

@pytest.mark.parametrize("bits,signed,alpha",[
    (8, True,  [1.000, 1.000, 1.000, 1.000]),
    (8, False, [1.000, 1.000, 1.000, 1.000]),
    (4, True,  [1.000, 1.000, 1.000, 1.000]),
    (4, False, [1.000, 1.000, 1.000, 1.000]),
    (8, True,  [0.125, 0.250, 0.500, 1.000]),
    (8, False, [0.125, 0.250, 0.500, 1.000]),
    (5, True,  [0.250, 0.250, 0.125, 0.125]),
    (5, False, [0.250, 0.250, 0.125, 0.125]),
    (4, True,  [0.125, 0.250, 0.500, 1.000]),
    (4, False, [0.125, 0.250, 0.500, 1.000]),
    (3, True,  [0.125, 0.125, 0.250, 0.125]),
    (3, False, [0.125, 0.125, 0.250, 0.125])
])
def test_qkeras_tensor_alpha(bits, signed, alpha, request):
    random_state = np.random.RandomState(seed=42)
    max_val = np.array(alpha) * 2**(bits-signed)
    min_val = -(max_val + 1)
    w1 = random_state.randint(low=min_val, high=max_val, size=(3, 4))
    b1 = np.array([0.0, 0.0, 0.0, 0.0])
    x = x_in = tf.keras.layers.Input(shape=3)
    x = QActivation(
        quantized_bits(bits=4, integer=3, keep_negative=True)
    )(x)
    x = QDense(
        4,
        kernel_quantizer=quantized_bits(
            bits=bits, integer=(bits-signed), keep_negative=signed, alpha=alpha
        ),
    )(x)
    x = QActivation(quantized_relu(bits=3, integer=3))(x)
    model = tf.keras.Model(inputs=[x_in], outputs=[x])
    model.compile()
    model.layers[2].set_weights([w1, b1])
    onnx_model, _ = from_keras(model)
    model_path = f"model_test_qkeras_tensor_alpha_{request.node.callspec.id}.onnx"
    onnx.save(onnx_model, model_path)
    onnx_model = ModelWrapper(onnx_model)

    data = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[0.0, 1.0, 2.0]],
            [[2.0, 1.0, 0.0]],
            [[4.0, 4.0, 4.0]],
            [[7.0, 7.0, 7.0]],
            [[6.0, 0.0, 7.0]],
            [[3.0, 3.0, 3.0]],
            [[7.0, 0.0, 0.0]],
            [[0.0, 7.0, 0.0]],
            [[0.0, 0.0, 7.0]],
        ]
    ).astype(np.float32)
    for x in data:
        y_qkeras = model.predict(x, verbose=0)
        idict = {onnx_model.graph.input[0].name: x}
        odict = oxe.execute_onnx(onnx_model, idict, True)
        y_qonnx = odict[onnx_model.graph.output[0].name]
        assert np.array_equal(y_qkeras, y_qonnx)
    os.remove(model_path)
    
@pytest.mark.parametrize("quantizers", kb_quantizers, ids=kb_quantizers_ids)
def test_qkeras_qconv2d_1(quantizers, request):
    kq, bq = quantizers
    k_ini = tf.keras.initializers.RandomUniform(minval=kq.min(), maxval=kq.max())
    b_ini = tf.keras.initializers.RandomUniform(minval=bq.min(), maxval=bq.max())
    x = x_in = Input((28, 28, 3), name="input")
    x = QConv2D(
        32,
        (2, 2),
        strides=(2, 2),
        kernel_quantizer=kq,
        bias_quantizer=bq,
        activation=quantized_bits(4, 4, 1, alpha=1.0),
        kernel_initializer=k_ini,
        bias_initializer=b_ini,
        name="conv2d_0",
    )(x)
    x = QActivation("quantized_relu(6,2)", name="act1")(x)
    x = QConv2D(
        64,
        (3, 3),
        strides=(2, 2),
        kernel_quantizer=kq,
        bias_quantizer=bq,
        use_bias=False,
        kernel_initializer=k_ini,
        bias_initializer=b_ini,
        name="conv2d_1",
    )(x)
    model = Model(inputs=[x_in], outputs=[x])

    x_test = np.random.uniform(low=-1.0, high=1.0, size=(1, 28, 28, 3)).astype(dtype=np.float32)
    y_qkeras = model.predict(x_test)

    onnx_model, external_storage = from_keras(model, "test_qkeras_conversion", opset=9)
    assert external_storage is None
    model_path = f"model_test_qkeras_qconv2d1_{request.node.callspec.id}.onnx"
    onnx.save(onnx_model, model_path)

    onnx_model = ModelWrapper(model_path)
    onnx_model = onnx_model.transform(InferShapes())

    idict = {onnx_model.graph.input[0].name: x_test}
    odict = oxe.execute_onnx(onnx_model, idict, True)
    y_qonnx = odict[onnx_model.graph.output[0].name]

    np.testing.assert_allclose(y_qkeras, y_qonnx, rtol=1e-4, atol=1e-4)
    os.remove(model_path)


@pytest.mark.parametrize("quantizers", act_quantizers_relu, ids=act_quantizers_relu_ids)
def test_qkeras_qconv2d_2(quantizers, request):
    kq, bq = (quantized_bits(4, 0, 1, alpha=1), quantized_bits(8, 0, 1, alpha=1))
    k_ini = tf.keras.initializers.RandomUniform(minval=kq.min(), maxval=kq.max())
    b_ini = tf.keras.initializers.RandomUniform(minval=bq.min(), maxval=bq.max())
    x = x_in = Input((28, 28, 3), name="input")
    x = QConv2D(
        32,
        (2, 2),
        strides=(2, 2),
        kernel_quantizer=kq,
        bias_quantizer=bq,
        activation=quantized_bits(4, 4, 1, alpha=1.0),
        kernel_initializer=k_ini,
        bias_initializer=b_ini,
        name="conv2d_0",
    )(x)
    x = QActivation(quantizers, name="act1")(x)
    x = QConv2D(
        64,
        (3, 3),
        strides=(2, 2),
        kernel_quantizer=kq,
        bias_quantizer=bq,
        use_bias=False,
        kernel_initializer=k_ini,
        bias_initializer=b_ini,
        name="conv2d_1",
    )(x)
    model = Model(inputs=[x_in], outputs=[x])

    x_test = np.random.uniform(low=-1.0, high=1.0, size=(1, 28, 28, 3)).astype(dtype=np.float32)
    y_qkeras = model.predict(x_test)

    onnx_model, external_storage = from_keras(model, "test_qkeras_conversion", opset=9)
    assert external_storage is None
    model_path = f"model_test_qkeras_qconv2d2_{request.node.callspec.id}.onnx"
    onnx.save(onnx_model, model_path)

    onnx_model = ModelWrapper(model_path)
    onnx_model = onnx_model.transform(InferShapes())

    idict = {onnx_model.graph.input[0].name: x_test}
    odict = oxe.execute_onnx(onnx_model, idict, True)
    y_qonnx = odict[onnx_model.graph.output[0].name]

    np.testing.assert_allclose(y_qkeras, y_qonnx, rtol=1e-4, atol=1e-4)
    os.remove(model_path)


@pytest.mark.parametrize("quantizers", act_quantizers, ids=act_quantizers_ids)
def test_qkeras_qconv2d_3(quantizers, request):
    kq, bq = (quantized_bits(4, 2, 0, alpha=1), quantized_bits(8, 8, 1, alpha=1))
    k_ini = tf.keras.initializers.RandomUniform(minval=kq.min(), maxval=kq.max())
    b_ini = tf.keras.initializers.RandomUniform(minval=bq.min(), maxval=bq.max())
    x = x_in = Input((28, 28, 3), name="input")
    x = QConv2D(
        32,
        (2, 2),
        strides=(2, 2),
        kernel_quantizer=kq,
        bias_quantizer=bq,
        activation=quantizers,
        kernel_initializer=k_ini,
        bias_initializer=b_ini,
        name="conv2d_0",
    )(x)
    x = QConv2D(
        64,
        (3, 3),
        strides=(2, 2),
        kernel_quantizer=kq,
        bias_quantizer=bq,
        activation=quantizers,
        kernel_initializer=k_ini,
        bias_initializer=b_ini,
        name="conv2d_1",
    )(x)
    x = QActivation("quantized_relu(6,2)", name="act1")(x)
    x = QConv2D(
        64,
        (3, 3),
        strides=(2, 2),
        kernel_quantizer=kq,
        bias_quantizer=bq,
        kernel_initializer=k_ini,
        bias_initializer=b_ini,
        name="conv2d_2",
    )(x)
    model = Model(inputs=[x_in], outputs=[x])

    x_test = np.random.uniform(low=-1.0, high=1.0, size=(1, 28, 28, 3)).astype(dtype=np.float32)
    y_qkeras = model.predict(x_test)

    onnx_model, external_storage = from_keras(model, "test_qkeras_conversion", opset=9)
    assert external_storage is None
    model_path = f"model_test_qkeras_qconv2d3_{request.node.callspec.id}.onnx"
    onnx.save(onnx_model, model_path)

    onnx_model = ModelWrapper(model_path)
    onnx_model = onnx_model.transform(InferShapes())

    idict = {onnx_model.graph.input[0].name: x_test}
    odict = oxe.execute_onnx(onnx_model, idict, True)
    y_qonnx = odict[onnx_model.graph.output[0].name]

    np.testing.assert_allclose(y_qkeras, y_qonnx, rtol=1e-4, atol=1e-4)
    os.remove(model_path)


@pytest.mark.parametrize("quantizers", kb_quantizers, ids=kb_quantizers_ids)
def test_qkeras_qconv2d_conversion_1(quantizers, request):
    kq, bq = quantizers
    k_ini = tf.keras.initializers.RandomUniform(minval=kq.min(), maxval=kq.max())
    b_ini = tf.keras.initializers.RandomUniform(minval=bq.min(), maxval=bq.max())
    x = x_in = Input((28, 28, 1), name="input")
    x = QConv2D(
        32,
        (2, 2),
        strides=(2, 2),
        kernel_quantizer=kq,
        bias_quantizer=bq,
        kernel_initializer=k_ini,
        bias_initializer=b_ini,
        name="conv2d_0_m",
    )(x)
    x = QConv2D(
        64,
        (3, 3),
        strides=(2, 2),
        kernel_quantizer=kq,
        bias_quantizer=bq,
        kernel_initializer=k_ini,
        bias_initializer=b_ini,
        name="conv2d_1_m",
    )(x)
    x = QConv2D(
        64,
        (2, 2),
        strides=(2, 2),
        kernel_quantizer=kq,
        kernel_initializer=k_ini,
        use_bias=False,  # Lets try this one without bias to see if that trips up the converter
        name="conv2d_2_m",
    )(x)
    x = QConv2D(
        64,
        (2, 2),
        strides=(2, 2),
        kernel_quantizer=kq,
        kernel_initializer=k_ini,
        use_bias=False,  # Lets try this one without bias to see if that trips up the converter
        name="conv2d_3_m",
    )(x)
    x = QActivation("quantized_relu(6,4)", name="act2_m")(x)
    x = Flatten(name="flatten")(x)
    x = QDense(
        10,
        kernel_quantizer=kq,
        bias_quantizer=bq,
        kernel_initializer=k_ini,
        bias_initializer=b_ini,
        name="dense",
    )(x)
    model = Model(inputs=[x_in], outputs=[x])

    x_test = np.random.uniform(low=-1.0, high=1.0, size=(1, 28, 28, 1)).astype(dtype=np.float32)
    y_qkeras = model.predict(x_test)

    onnx_model, external_storage = from_keras(model, "test_qkeras_qconv2d_conversion", opset=9)
    assert external_storage is None
    model_path = f"model_test_qkeras_qconv2d_conversion1_{request.node.callspec.id}.onnx"
    onnx.save(onnx_model, model_path)

    onnx_model = ModelWrapper(model_path)
    onnx_model = onnx_model.transform(InferShapes())
    idict = {onnx_model.graph.input[0].name: x_test}
    odict = oxe.execute_onnx(onnx_model, idict, True)
    y_qonnx = odict[onnx_model.graph.output[0].name]
    np.testing.assert_allclose(y_qkeras, y_qonnx, rtol=1e-4, atol=1e-4)
    os.remove(model_path)


@pytest.mark.parametrize("quantizers", act_quantizers_relu, ids=act_quantizers_relu_ids)
def test_qkeras_qconv2d_conversion_2(quantizers, request):
    kq, bq = (quantized_bits(4, 4, 0, alpha=1), quantized_bits(8, 8, 0, alpha=1))
    k_ini = tf.keras.initializers.RandomUniform(minval=kq.min(), maxval=kq.max())
    b_ini = tf.keras.initializers.RandomUniform(minval=bq.min(), maxval=bq.max())
    x = x_in = Input((28, 28, 1), name="input")
    x = QConv2D(
        32,
        (2, 2),
        strides=(2, 2),
        kernel_quantizer=kq,
        bias_quantizer=bq,
        activation=quantized_bits(4, 2, 0, alpha=1),
        kernel_initializer=k_ini,
        bias_initializer=b_ini,
        name="conv2d_0_m",
    )(x)
    x = QActivation(quantizers, name="act1_m")(x)
    x = QConv2D(
        64,
        (3, 3),
        strides=(2, 2),
        kernel_quantizer=kq,
        bias_quantizer=bq,
        kernel_initializer=k_ini,
        bias_initializer=b_ini,
        name="conv2d_1_m",
    )(x)
    x = Flatten(name="flatten")(x)
    x = QActivation(quantizers, name="act2_m")(x)
    x = QDense(
        32,
        kernel_quantizer=kq,
        bias_quantizer=bq,
        activation=quantized_bits(8, 2, 1, alpha=1),
        kernel_initializer=k_ini,
        bias_initializer=b_ini,
        name="dense_1",
    )(x)
    x = QDense(
        10,
        kernel_quantizer=kq,
        bias_quantizer=bq,
        kernel_initializer=k_ini,
        bias_initializer=b_ini,
        name="dense_2",
    )(x)
    x = Activation("softmax", name="softmax")(x)

    model = Model(inputs=[x_in], outputs=[x])

    x_test = np.random.uniform(low=-1.0, high=1.0, size=(1, 28, 28, 1)).astype(dtype=np.float32)
    y_qkeras = model.predict(x_test)

    onnx_model, external_storage = from_keras(model, "test_qkeras_qconv2d_conversion", opset=9)
    assert external_storage is None
    model_path = f"model_test_qkeras_qconv2d_conversion2_{request.node.callspec.id}.onnx"
    onnx.save(onnx_model, model_path)

    onnx_model = ModelWrapper(model_path)
    onnx_model = onnx_model.transform(InferShapes())
    idict = {onnx_model.graph.input[0].name: x_test}
    odict = oxe.execute_onnx(onnx_model, idict, True)
    y_qonnx = odict[onnx_model.graph.output[0].name]
    np.testing.assert_allclose(y_qkeras, y_qonnx, rtol=1e-4, atol=1e-4)
    os.remove(model_path)


# quantized_relu should not be used as a layer activation
# def test_qkeras_broken_1(quantizers, request):
#     kq, bq = (quantized_bits(4, 4, 0, alpha=1), quantized_bits(8, 8, 0, alpha=1))
#     # Initialize the kernel & bias to RandonUniform within the range of the quantizers
#     k_ini = tf.keras.initializers.RandomUniform(minval=kq.min(), maxval=kq.max())
#     b_ini = tf.keras.initializers.RandomUniform(minval=bq.min(), maxval=bq.max())
#     x = x_in = Input((16), name="input")
#     x = QDense(
#         32,
#         kernel_quantizer=kq,
#         bias_quantizer=bq,
#         activation="quantized_relu(6,2)",
#         kernel_initializer=k_ini,
#         bias_initializer=b_ini,
#         name="dense_0",
#     )(x)
#     x = QDense(
#         32,
#         kernel_quantizer=kq,
#         bias_quantizer=bq,
#         kernel_initializer=k_ini,
#         bias_initializer=b_ini,
#         name="dense_1",
#     )(x)
#     model = Model(inputs=[x_in], outputs=[x])
#     x_test = np.random.uniform(low=-1.0, high=1.0, size=(1, 16)).astype(dtype=np.float32)
#     y_qkeras = model.predict(x_test)

#     onnx_model, external_storage =from_keras(model, "test_qkeras_conversion", opset=9)
#     assert external_storage is None
#     model_path = f"test_qkeras_broken1{request.node.callspec.id}.onnx"
#     onnx.save(onnx_model, model_path)

#     onnx_model = ModelWrapper(model_path)
#     onnx_model = onnx_model.transform(InferShapes())

#     idict = {onnx_model.graph.input[0].name: x_test}
#     odict = oxe.execute_onnx(onnx_model, idict, True)
#     y_qonnx = odict[onnx_model.graph.output[0].name]
#     np.testing.assert_allclose(y_qkeras, y_qonnx, rtol=1e-4, atol=1e-4)


# quantized_bits should not be used in QActivation
# def test_qkeras_broken_2(quantizers, request):
#     kq, bq = (quantized_bits(4, 4, 0, alpha=1), quantized_bits(8, 8, 0, alpha=1))
#     # Initialize the kernel & bias to RandonUniform within the range of the quantizers
#     k_ini = tf.keras.initializers.RandomUniform(minval=kq.min(), maxval=kq.max())
#     b_ini = tf.keras.initializers.RandomUniform(minval=bq.min(), maxval=bq.max())
#     x = x_in = Input((16), name="input")
#     x = QDense(
#         32,
#         kernel_quantizer=kq,
#         bias_quantizer=bq,
#         kernel_initializer=k_ini,
#         bias_initializer=b_ini,
#         name="dense_0",
#     )(x)
#     x = QActivation("quantized_bits(4,4,1)", name="act2_m")(x)
#     x = QDense(
#         32,
#         kernel_quantizer=kq,
#         bias_quantizer=bq,
#         kernel_initializer=k_ini,
#         bias_initializer=b_ini,
#         name="dense_1",
#     )(x)
#     model = Model(inputs=[x_in], outputs=[x])
#     x_test = np.random.uniform(low=-1.0, high=1.0, size=(1, 16)).astype(dtype=np.float32)
#     y_qkeras = model.predict(x_test)

#     onnx_model, external_storage =from_keras(model, "test_qkeras_conversion", opset=9)
#     assert external_storage is None
#     model_path = f"test_qkeras_broken2{request.node.callspec.id}.onnx"
#     onnx.save(onnx_model, model_path)

#     onnx_model = ModelWrapper(model_path)
#     onnx_model = onnx_model.transform(InferShapes())

#     idict = {onnx_model.graph.input[0].name: x_test}
#     odict = oxe.execute_onnx(onnx_model, idict, True)
#     y_qonnx = odict[onnx_model.graph.output[0].name]
#     np.testing.assert_allclose(y_qkeras, y_qonnx, rtol=1e-4, atol=1e-4)

if __name__ == "__main__":
    pytest.main([__file__])
