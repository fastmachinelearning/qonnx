import tensorflow as tf


class HGQIdentity(tf.keras.layers.Layer):
    def __init__(self, name, dtype):
        super(HGQIdentity, self).__init__(name=name, dtype=dtype)

    def call(self, inputs):
        return inputs


# TODO: this should be implemented in HGQ so we can import it here
# from HGQ.utils import REGISTERED_LAYERS as HGQ_LAYERS
HGQ_LAYERS = ["FixedPointQuantizer"]


def extract_quantizers_from_hgq_layer(layer, model):
    """ """
    layer_class = layer.__class__.__name__
    if layer_class in HGQ_LAYERS:
        handler = handler_map.get(layer_class, None)
        if handler:
            return handler_map[layer_class](layer, model)
        else:
            return layer_class, layer.get_config(), None
    else:
        return layer_class, layer.get_config(), None


def extract_FixedPointQuantizer(layer, model):
    quantizers = layer.get_config()

    if "overrides" not in quantizers:
        # TODO: add support for FixedPointQuantizer which dont override
        raise ValueError("Not supported: FixedPointQuantizer has no layers to override")

    quantizers["inputs"] = {
        "keep_negative": layer.keep_negative.numpy(),
        "bits": layer.bits.numpy(),
        "integer_bits": layer.integers.numpy(),
    }
    quantizers["keras_layer"] = "FixedPointQuantizer"
    keras_config = {"name": quantizers["name"], "dtype": "float32"}

    return "HGQIdentity", keras_config, quantizers


handler_map = {"FixedPointQuantizer": extract_FixedPointQuantizer}
