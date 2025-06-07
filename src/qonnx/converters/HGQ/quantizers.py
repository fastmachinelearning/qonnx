def get_quant_params(tensor, hgq_quantizer):
    return handler_map[hgq_quantizer["keras_layer"]](tensor, hgq_quantizer)


def convert_quantized_bits(tensor, quantizer):
    settings = {
        "attributes": {
            "RND": quantizer["RND"],
            "SAT": quantizer["SAT"],
        },
        "inputs": {
            "integer_bits": quantizer["inputs"]["integer_bits"],
            "keep_negative": quantizer["inputs"]["keep_negative"],
            "bits": quantizer["inputs"]["bits"],
        },
    }

    return settings


handler_map = {
    "FixedPointQuantizer": convert_quantized_bits,
}
