from tf2onnx.onnx_opset.math import DirectOp, MatMul
from tf2onnx.onnx_opset.nn import BiasAdd, ConvOp

from .quantizers import get_quant_params


def get_qkeras_onnx_handlers(all_quantizers):
    return {
        "Conv2D": (conv2d_handler, ["Conv2D", all_quantizers]),
        "MatMul": (dense_handler, ["MatMul", all_quantizers]),
        "BiasAdd": (bias_handler, ["BiasAdd", all_quantizers]),
        "Relu": (relu_handler, ["Relu", all_quantizers]),
    }


def _extract_node_name(onnx_name, keras_names):
    for keras_name in keras_names:
        match = "/" + keras_name + "/"
        if match in onnx_name:
            return keras_name

    return None


def qlayer_handler(ctx, node, name, args):
    all_quantizers = args[0]
    keras_name = _extract_node_name(name, all_quantizers.keys())
    if not keras_name:
        return  # Not found in quantizers, nothing to do
    quantizers = all_quantizers[keras_name]

    if quantizers.get("kernel_quantizer"):
        weights = node.inputs[1].get_tensor_value(as_list=True)
        kernel_quant_params = get_quant_params(weights, quantizers["kernel_quantizer"])
        ctx.insert_new_node_on_input(
            node, "Quant", node.input[1], name=node.name + "_kernel_quantizer", **kernel_quant_params, domain="qonnx"
        )

    if quantizers.get("bias_quantizer") and len(node.input) == 3:
        bias = node.inputs[2].get_tensor_value(as_list=True)
        bias_quant_params = get_quant_params(bias, quantizers["bias_quantizer"])
        ctx.insert_new_node_on_input(
            node, "Quant", node.input[2], name=node.name + "_bias_quantizer", **bias_quant_params, domain="qonnx"
        )

    if quantizers.get("activation"):
        output_shapes = [ctx.get_shape(node.output[0])]
        dtypes = [ctx.get_dtype(node.output[0])]
        act_quant_params = get_quant_params(None, quantizers["activation"])
        quant_act_node = ctx.make_node(
            "Quant",
            [node.output[0]],
            shapes=output_shapes,
            dtypes=dtypes,
            name=node.name + "_activation_quantizer",
            attr=act_quant_params,
            domain="qonnx",
        )
        ctx.insert_node_on_output(quant_act_node, node.output[0])


def qact_handler(ctx, node, name, args):
    all_quantizers = args[0]
    keras_name = _extract_node_name(name, all_quantizers.keys())
    if not keras_name:
        return  # Not found in quantizers, nothing to do
    quantizers = all_quantizers[keras_name]

    if quantizers.get("activation"):
        output_shapes = [ctx.get_shape(node.output[0])]
        dtypes = [ctx.get_dtype(node.output[0])]
        act_quant_params = get_quant_params(None, quantizers["activation"])
        quant_act_node = ctx.make_node(
            "Quant",
            [node.output[0]],
            shapes=output_shapes,
            dtypes=dtypes,
            name=node.name + "_activation_quantizer",
            attr=act_quant_params,
            domain="qonnx",
        )
        ctx.insert_node_on_output(quant_act_node, node.output[0])


def conv2d_handler(ctx, node, name, args):
    ConvOp.any_version(11, ctx, node)
    qlayer_handler(ctx, node, name, args)


def dense_handler(ctx, node, name, args):
    MatMul.version_1(ctx, node)
    qlayer_handler(ctx, node, name, args)


def bias_handler(ctx, node, name, args):
    BiasAdd.version_1(ctx, node)

    all_quantizers = args[0]
    keras_name = _extract_node_name(name, all_quantizers.keys())
    if not keras_name:
        return  # Not found in quantizers, nothing to do
    quantizers = all_quantizers[keras_name]

    if quantizers.get("bias_quantizer"):
        bias = node.inputs[1].get_tensor_value(as_list=True)
        bias_quant_params = get_quant_params(bias, quantizers["bias_quantizer"])
        ctx.insert_new_node_on_input(
            node, "Quant", node.input[1], name=node.name + "_bias_quantizer", **bias_quant_params, domain="qonnx"
        )


def relu_handler(ctx, node, name, args):
    DirectOp.version_1(ctx, node)
    # qact_handler(ctx, node, name, args)
