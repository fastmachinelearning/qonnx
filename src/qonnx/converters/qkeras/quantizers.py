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


def convert_quantized_bits(tensor, quantizer):
    config = quantizer.get_config()
    # Let's assume the quantizer is symmetric
    assert config["symmetric"] == 1

    signed = int(config["keep_negative"])
    narrow = int(False)
    scale = _get_scale_from_alpha(tensor, quantizer)
    zero_point = 0
    bit_width = int(config["bits"])

    return {"scale": scale, "zero_point": zero_point, "bit_width": bit_width, "signed": signed, "narrow": narrow}


def convert_quantized_relu(tensor, quantizer):
    config = quantizer.get_config()

    signed = int(True)
    narrow = int(False)
    scale = 1
    zero_point = 0
    bit_width = int(config["bits"])

    return {"scale": scale, "zero_point": zero_point, "bit_width": bit_width, "signed": signed, "narrow": narrow}


def convert_binary(tensor, quantizer):
    # TODO This is a stub, not a correct implementation
    config = quantizer.get_config()

    signed = 0
    narrow = int(False)
    scale = 1
    zero_point = 0
    bit_width = 1

    return {"scale": scale, "zero_point": zero_point, "bit_width": bit_width, "signed": signed, "narrow": narrow}


def convert_ternary(tensor, quantizer):
    # TODO This is a stub, not a correct implementation
    config = quantizer.get_config()

    signed = 0
    narrow = int(False)
    scale = 1
    zero_point = 0
    bit_width = 2

    return {"scale": scale, "zero_point": zero_point, "bit_width": bit_width, "signed": signed, "narrow": narrow}


handler_map = {
    "quantized_bits": convert_quantized_bits,
    "quantized_relu": convert_quantized_relu,
    "binary": convert_binary,
    "ternary": convert_ternary,
}
