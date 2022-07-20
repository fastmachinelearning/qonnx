import qkeras
import six


def get_quant_params(tensor, qkeras_quantizer):
    if isinstance(qkeras_quantizer, str):
        qkeras_quantizer = qkeras.get_quantizer(qkeras_quantizer)

    return handler_map[qkeras_quantizer.__class__.__name__](tensor, qkeras_quantizer)


def _get_scale_from_alpha(tensor, quantizer):
    alpha = quantizer.get_config()["alpha"]

    if alpha is None:
        return 1
    elif isinstance(alpha, six.string_types):
        raise Exception(f"Cannot parse alpha = {alpha}.")
        return 1
    else:
        return alpha


def _get_quantizer_scale(tensor, quantizer):
    # call the quantizer on the tensor to get its scale
    import numpy as np

    quantizer(np.array(tensor).astype(np.float32))
    return quantizer.scale


def convert_quantized_bits(tensor, quantizer):
    config = quantizer.get_config()
    signed = int(config["keep_negative"])
    narrow = int(config["symmetric"])
    qscale = _get_quantizer_scale(tensor, quantizer)
    assert qscale == 1, "Non-unity alpha is not yet supported"
    scale = 1.0 / 2 ** (int(config["bits"]) - int(config["integer"] + signed))
    zero_point = 0
    bit_width = int(config["bits"])
    rounding_mode = "ROUND"

    settings = {
        "attributes": {"signed": signed, "narrow": narrow, "rounding_mode": rounding_mode},
        "inputs": {"scale": scale, "zero_point": zero_point, "bit_width": bit_width},
    }
    return settings


def convert_quantized_relu(tensor, quantizer):
    config = quantizer.get_config()

    signed = int(config["negative_slope"] != 0.0)
    narrow = int(False)
    scale = 1.0 / 2 ** (int(config["bits"]) - int(config["integer"] + signed))
    zero_point = 0
    bit_width = int(config["bits"])
    rounding_mode = "ROUND"

    settings = {
        "attributes": {"signed": signed, "narrow": narrow, "rounding_mode": rounding_mode},
        "inputs": {"scale": scale, "zero_point": zero_point, "bit_width": bit_width},
    }
    return settings


def convert_binary(tensor, quantizer):
    signed = 1
    narrow = 1
    qscale = _get_quantizer_scale(tensor, quantizer)
    assert qscale == 1, "binary - non-unity alpha is not yet supported"
    scale = 1
    zero_point = 0
    bit_width = 1
    rounding_mode = "ROUND"

    settings = {
        "attributes": {"signed": signed, "narrow": narrow, "rounding_mode": rounding_mode},
        "inputs": {"scale": scale, "zero_point": zero_point, "bit_width": bit_width},
    }
    return settings


def convert_ternary(tensor, quantizer):
    config = quantizer.get_config()
    signed = 1
    narrow = 1
    qscale = _get_quantizer_scale(tensor, quantizer)
    assert qscale == 1, "ternary - non-unity alpha is not yet supported"
    # qkeras ternary quantizer has threshold parameter to change rounding point
    # here we could scale such that normal 'ROUND' op gives the same result, but doesn't work with re-scaling
    t = config["threshold"]
    if t is None:
        ternary = qkeras.ternary()
        t = ternary.default_threshold
    assert t == 0.5, "ternary - only threshold 0.5 is supported"
    # note that if assertions fail, Quant node is not inserted, but model is still converted
    # this seems to be unexpected behavior
    scale = 1.0
    zero_point = 0
    bit_width = 2
    rounding_mode = "ROUND"

    settings = {
        "attributes": {"signed": signed, "narrow": narrow, "rounding_mode": rounding_mode},
        "inputs": {"scale": scale, "zero_point": zero_point, "bit_width": bit_width},
    }
    return settings


handler_map = {
    "quantized_bits": convert_quantized_bits,
    "quantized_relu": convert_quantized_relu,
    "binary": convert_binary,
    "ternary": convert_ternary,
}
