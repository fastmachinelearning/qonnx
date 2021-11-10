import tensorflow as tf
import tf2onnx
from qkeras.utils import REGISTERED_LAYERS as QKERAS_LAYERS

from finn.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model

from .qkeras.onnx import get_qkeras_onnx_handlers
from .qkeras.qlayers import extract_quantizers_from_layer

_unsupported_layers = [
    # These require some extra work
    "QBatchNormalization",
    "QConv2DBatchnorm",
    "QDepthwiseConv2DBatchnorm",
]


def _is_qkeras_model(model):
    def iterate_model(model):
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                found_qkeras = iterate_model(layer)
                if found_qkeras:
                    return True
            elif layer.__class__.__name__ in QKERAS_LAYERS:
                return True

        return False

    return iterate_model(model)


def _check_supported_layers(model):
    def iterate_model(model):
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                iterate_model(layer)
            elif layer.__class__.__name__ in _unsupported_layers:
                raise Exception("Currently unsupported layer found in QKeras model: {}".format(layer.__class__.__name__))

    iterate_model(model)


def _strip_qkeras_model(model):
    quantizers = {}

    def extract_quantizers(layer):
        keras_cls_name, layer_cfg, layer_quantizers = extract_quantizers_from_layer(layer)
        if layer_quantizers:
            layer_quantizers = {
                k: None if v == "None" else v for k, v in layer_quantizers.items()
            }  # Get rid of 'None' strings
            quantizers[layer.name] = layer_quantizers

        layer_class = tf.keras.layers.__dict__.get(keras_cls_name, None)
        if layer_class is None:
            raise Exception("Cannot create Keras layer from QKeras class {}".format(keras_cls_name))

        return layer_class.from_config(layer_cfg)

    stripped_model = tf.keras.models.clone_model(model, clone_function=extract_quantizers)

    return stripped_model, quantizers


def _convert_quantizers_to_nodes(onnx_model, quantizers_dict):

    for node_name, quantizers in quantizers_dict.items():
        print(node_name, quantizers)

    for n in onnx_model.graph.node:
        print(n)

    return onnx_model.model


def from_keras(
    model,
    input_signature=None,
    opset=None,
    custom_ops=None,
    custom_op_handlers=None,
    custom_rewriter=None,
    inputs_as_nchw=None,
    extra_opset=None,
    shape_override=None,
    target=None,
    large_model=False,
    output_path=None,
):
    """Convert a keras model to QONNX. The API follows the `from_keras` function of tf2onnx.

    Args:
        model: the tf.keras model we want to convert
        input_signature: a tf.TensorSpec or a numpy array defining the shape/dtype of the input
        opset: the opset to be used for the ONNX model, default is the latest
        custom_ops: if a model contains ops not recognized by onnx runtime,
            you can tag these ops with a custom op domain so that the
            runtime can still open the model. Type is a dictionary `{op name: domain}`.
        target: list of workarounds applied to help certain platforms
        custom_op_handlers: dictionary of custom ops handlers
        custom_rewriter: list of custom graph rewriters
        extra_opset: list of extra opset's, for example the opset's used by custom ops
        shape_override: dict with inputs that override the shapes given by tensorflow
        inputs_as_nchw: transpose inputs in list from nchw to nhwc
        large_model: use the ONNX external tensor storage format
        output_path: save model to output_path

    Returns:
        An ONNX model_proto and an external_tensor_storage dict.
    """

    assert not large_model  # TODO for now, let's focus only on models that don't store tensors externally

    if _is_qkeras_model(model):
        _check_supported_layers(model)
        keras_model, quantizers = _strip_qkeras_model(model)
    else:
        keras_model, quantizers = model, {}

    qkeras_op_handlers = get_qkeras_onnx_handlers(quantizers)

    if custom_op_handlers is not None:
        qkeras_op_handlers.update(custom_op_handlers)

    model_proto, external_storage = tf2onnx.convert.from_keras(
        keras_model,
        input_signature=input_signature,
        opset=opset,
        custom_ops=qkeras_op_handlers,
        custom_op_handlers=qkeras_op_handlers,
        custom_rewriter=custom_rewriter,
        inputs_as_nchw=inputs_as_nchw,
        extra_opset=extra_opset,
        shape_override=shape_override,
        target=target,
        large_model=large_model,
        output_path=None,
    )

    onnx_model = ModelWrapper(model_proto)
    cleanup_model(onnx_model)

    if output_path is not None:
        onnx_model.save(output_path)

    return onnx_model.model, external_storage
