import keras, onnx
import numpy as np
from HGQ.layers import HDense, HConv2D, PMaxPooling2D, PFlatten, PReshape, HQuantize
from HGQ import ResetMinMax, FreeBOPs
from HGQ import trace_minmax, to_proxy_model
from qonnx.converters.keras import from_keras
from qonnx.util.exec_qonnx import exec_qonnx


def test_convert_HGQ_two_conv2d_to_QONNX():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    # NOTE: we just test a few samples
    np.save("/tmp/x_test.npy", x_test[:100])

    model = keras.models.Sequential([
        HQuantize(beta=3e-5),
        PReshape((28, 28, 1)),
        PMaxPooling2D((2, 2)),
        HConv2D(1, (3, 3), activation='relu', beta=3e-5, parallel_factor=144),
        PMaxPooling2D((2, 2)),
        HConv2D(1, (3, 3), activation='relu', beta=3e-5, parallel_factor=16),
        PMaxPooling2D((2, 2)),
        PFlatten(),
        HDense(10, beta=3e-5)
    ])

    opt = keras.optimizers.Adam(learning_rate=0.001)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    callbacks = [ResetMinMax(), FreeBOPs()]

    model.fit(x_train, y_train, epochs=1, batch_size=32, callbacks=callbacks)

    trace_minmax(model, x_train, cover_factor=1.0)
    proxy = to_proxy_model(model, aggressive=True)

    onnx_model, external_storage = from_keras(proxy, "test_qkeras_conversion", opset=9)
    onnx.save(onnx_model, '/tmp/hgq.onnx')

    qonnx_out = exec_qonnx('/tmp/hgq.onnx', "/tmp/x_test.npy")
    hgq_out = proxy.predict(x_test[:100])
    assert np.isclose(
        qonnx_out, hgq_out
    ).all(), "Output of HGQ proxy model and converted QONNX model should match."
