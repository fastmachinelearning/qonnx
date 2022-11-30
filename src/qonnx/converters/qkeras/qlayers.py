import qkeras

# import tensorflow as tf
from qkeras.quantizers import BaseQuantizer
from qkeras.utils import REGISTERED_LAYERS as QKERAS_LAYERS


def extract_quantizers_from_layer(layer):
    """ """
    layer_class = layer.__class__.__name__
    if layer_class in QKERAS_LAYERS:
        handler = handler_map.get(layer_class, None)
        if handler:
            return handler_map[layer_class](layer)
        else:
            return extract_generic(layer)
    else:
        return layer_class, layer.get_config(), None


def _is_keras_quantizer(quant):
    """Check if the quantizer is a qkeras quantizer

    Args:
        quant: The quantizer node we need this information for

    Returns:
        True if the quantizer is a qkeras quantizer
    """
    try:
        # If we can deserialize the quantizer, it means it belongs to qkeras
        # TODO Since quantizer can be any callable, this should be more robust
        quant_obj = qkeras.get_quantizer(quant)
        return isinstance(quant_obj, BaseQuantizer)
    except ValueError:
        return False


def _extract_initializers(layer_cfg):
    """Return the initializers for the layer

    Args:
        layer_cfg: The layer configuration

    Returns:
        Initializers for the given layer
    """
    initializers = {}
    for key, value in layer_cfg.items():
        if value is None:
            continue
        if "initializer" in key:
            if value["class_name"] == "QInitializer":
                # Save the quantized initializer and it's replacement
                initializers[key] = (value, value["config"]["initializer"])

    return initializers


def _extract_constraints(layer_cfg):
    """Returns the constraints for the layer

    Args:
        layer_cfg: The layer configuration

    Returns:
        Initializers for the given layer
    """
    constraints = {}
    for key, value in layer_cfg.items():
        if value is None:
            continue
        if "constraint" in key:
            if value["class_name"] == "Clip":
                # QKeras doesn't keep the original constraint in the config (bug?)
                constraints[key] = (value, None)

    return constraints


_activation_map = {
    "quantized_bits": "linear",
    "quantized_relu": "relu",
    "binary": "linear",
    "ternary": "linear",
    "quantized_tanh": "tanh",
    "quantized_sigmoid": "sigmoid",
    "quantized_po2": "linear",
    "quantized_relu_po2": "relu",
}


def _replace_activation(quant_act):
    if quant_act is not None:
        for act_key, act_val in _activation_map.items():
            if act_key in quant_act:
                return act_val

    return "linear"


def extract_qlayer(layer):
    quantizers = layer.get_quantization_config()

    keras_config = layer.get_config()

    keras_config.pop("kernel_quantizer", None)
    keras_config.pop("bias_quantizer", None)
    keras_config.pop("kernel_range", None)
    keras_config.pop("bias_range", None)

    # Check if activation is quantized
    if _is_keras_quantizer(keras_config["activation"]):
        keras_config["activation"] = _replace_activation(quantizers["activation"])
    else:
        quantizers["activation"] = None

    quant_init = _extract_initializers(keras_config)
    for key, (quant, non_quant) in quant_init.items():
        quantizers[key] = quant
        keras_config[key] = non_quant

    quant_const = _extract_constraints(keras_config)
    for key, (quant, non_quant) in quant_const.items():
        quantizers[key] = quant
        keras_config[key] = non_quant

    return layer.__class__.__name__[1:], keras_config, quantizers


def extract_qact(layer):
    # As of version 0.9.0, QKeras actiations store quantizer config as a plain string, not dict
    # TODO complain to Hao about this inconsistency
    quantizers = {"activation": layer.get_quantization_config()}

    keras_config = layer.get_config()
    keras_config["activation"] = _replace_activation(quantizers["activation"])

    return "Activation", keras_config, quantizers


# This is a remnant of a first attempt at creating an universal parser
# however it got too complicated so portions were extracted to separate
# parsers. In the future, this should be removed.
def extract_generic(layer):
    get_quant_op = getattr(layer, "get_quantization_config", None)
    if callable(get_quant_op):
        quantizers = get_quant_op()
    else:
        quantizers = {}

    keras_cls_name = layer.__class__.__name__[1:]  # Drop the 'Q' from the layer name

    layer_cfg = layer.get_config()
    # Try to remove quantizers from the config
    non_quant = []
    if layer.name in quantizers:
        for quant_key, quant in quantizers.items():
            try:
                # If we can deserialize the quantizer, it means it belongs to qkeras
                qkeras.get_quantizer(quant)
                layer_cfg.pop(quant_key)
            except ValueError:
                # Otherwise it is not a quantizer (an activation, filter config, etc)
                non_quant.append(quant_key)

    for quant_key in non_quant:
        quantizers.pop(quant_key)

    # TODO Put proper activation in layer config

    # TODO extract initializers and constraints

    # Also remove deprecated 'kernel_range' and 'bias_range' From QConv1D/2D
    layer_cfg.pop("kernel_range", None)
    layer_cfg.pop("bias_range", None)

    return keras_cls_name, layer_cfg, quantizers


handler_map = {
    "QConv1D": extract_qlayer,
    "QConv2D": extract_qlayer,
    "QDense": extract_qlayer,
    "QActivation": extract_qact,
}
