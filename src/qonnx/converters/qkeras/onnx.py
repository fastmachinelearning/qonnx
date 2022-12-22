import numpy as np
from tf2onnx.late_rewriters import channel_order_rewriters
from tf2onnx.onnx_opset.math import DirectOp, MatMul
from tf2onnx.onnx_opset.nn import BiasAdd, ConvOp

from .quantizers import get_quant_params


def get_qkeras_onnx_handlers(all_quantizers):
    """Returns the handlers for each kind of layer

    Args:
        all_quantizers: All the quantizers of the model in dictionary format *check

    Returns:
        Dictionary containing the handler information for every type of layer
    """
    return {
        "Conv2D": (conv2d_handler, ["Conv2D", all_quantizers]),
        "MatMul": (dense_handler, ["MatMul", all_quantizers]),
        "BiasAdd": (bias_handler, ["BiasAdd", all_quantizers]),
        "Relu": (relu_handler, ["Relu", all_quantizers]),
        "Identity": (identity_handler, ["Identity", all_quantizers]),
    }


def _extract_node_name(onnx_node, keras_quantizers):
    """

    Args:
        onnx_node: The onnx node to get the information from
        keras_quantizers: The dictionary of all the keras quantizers

    """
    onnx_name = onnx_node.name
    keras_names = keras_quantizers.keys()
    for keras_name in keras_names:
        match = "/" + keras_name + "/"
        if match in onnx_name:
            return keras_name
        elif "Identity" in onnx_name:
            onnx_input = onnx_node.input[0]
            keras_input = keras_quantizers[keras_name]["input"]
            if keras_input in onnx_input:
                return keras_name

    return None


def qlayer_handler(ctx, node, name, args):
    all_quantizers = args[0]
    keras_name = _extract_node_name(node, all_quantizers)
    if not keras_name:
        return  # Not found in quantizers, nothing to do
    quantizers = all_quantizers[keras_name]
    if quantizers.get("kernel_quantizer"):
        weights = node.inputs[1].get_tensor_value(as_list=True)
        quant_params = get_quant_params(weights, quantizers["kernel_quantizer"])
        attr = quant_params["attributes"]
        input_nodes = [node.input[1]]
        for key in quant_params["inputs"].keys():
            name = f"{node.name}_kernel_quantizer_{key}"
            np_val = np.asarray(quant_params["inputs"][key])
            ctx.make_const(name, np_val)
            input_nodes.append(name)
        ctx.insert_new_node_on_input(
            node, "Quant", input_nodes, name=node.name + "_kernel_quantizer", **attr, domain="qonnx"
        )

    if quantizers.get("bias_quantizer") and len(node.input) == 3:
        bias = node.inputs[2].get_tensor_value(as_list=True)
        quant_params = get_quant_params(bias, quantizers["bias_quantizer"])
        attr = quant_params["attributes"]
        input_nodes = [node.input[2]]
        for key in quant_params["inputs"].keys():
            name = f"{node.name}_bias_quantizer_{key}"
            np_val = np.asarray(quant_params["inputs"][key])
            ctx.make_const(name, np_val)
            input_nodes.append(name)
        ctx.insert_new_node_on_input(node, "Quant", input_nodes, name=node.name + "_bias_quantizer", **attr, domain="qonnx")

    if quantizers.get("activation"):
        dtypes = [ctx.get_dtype(node.output[0])]
        quant_params = get_quant_params(None, quantizers["activation"])
        attr = quant_params["attributes"]
        input_nodes = [node.output[0]]
        for key in quant_params["inputs"].keys():
            name = f"{node.name}_activation_quantizer_{key}"
            np_val = np.asarray(quant_params["inputs"][key])
            ctx.make_const(name, np_val)
            input_nodes.append(name)
        quant_act_node = ctx.make_node(
            "Quant",
            input_nodes,
            dtypes=dtypes,
            name=node.name + "_activation_quantizer",
            attr=attr,
            domain="qonnx",
        )
        ctx.insert_node_on_output(quant_act_node, node.output[0])
        ctx.set_shape(quant_act_node.output[0], ctx.get_shape(node.output[0]))


def qact_handler(ctx, node, name, args):
    all_quantizers = args[0]
    keras_name = _extract_node_name(node, all_quantizers)
    if not keras_name:
        return  # Not found in quantizers, nothing to do
    quantizers = all_quantizers[keras_name]
    if quantizers.get("activation"):
        dtypes = [ctx.get_dtype(node.output[0])]
        quant_params = get_quant_params(None, quantizers["activation"])
        attr = quant_params["attributes"]
        input_nodes = [node.output[0]]
        for key in quant_params["inputs"].keys():
            name = f"{node.name}_activation_quantizer_{key}"
            np_val = np.asarray(quant_params["inputs"][key])
            ctx.make_const(name, np_val)
            input_nodes.append(name)
        quant_act_node = ctx.make_node(
            "Quant",
            input_nodes,
            dtypes=dtypes,
            name=node.name + "_activation_quantizer",
            attr=attr,
            domain="qonnx",
        )
        ctx.insert_node_on_output(quant_act_node, node.output[0])
        ctx.set_shape(quant_act_node.output[0], ctx.get_shape(node.output[0]))
        channel_order_rewriters._to_channel_first_handler(ctx, quant_act_node)


def conv2d_handler(ctx, node, name, args):
    ConvOp.any_version(11, ctx, node)
    qlayer_handler(ctx, node, name, args)


def dense_handler(ctx, node, name, args):
    MatMul.version_1(ctx, node)
    qlayer_handler(ctx, node, name, args)


def bias_handler(ctx, node, name, args):
    BiasAdd.version_1(ctx, node)
    all_quantizers = args[0]
    keras_name = _extract_node_name(node, all_quantizers)
    if not keras_name:
        return  # Not found in quantizers, nothing to do
    quantizers = all_quantizers[keras_name]

    if quantizers.get("bias_quantizer"):
        bias = node.inputs[1].get_tensor_value(as_list=True)
        quant_params = get_quant_params(bias, quantizers["bias_quantizer"])
        attr = quant_params["attributes"]
        input_nodes = [node.input[1]]
        for key in quant_params["inputs"].keys():
            name = f"{node.name}_bias_quantizer_{key}"
            np_val = np.asarray(quant_params["inputs"][key])
            ctx.make_const(name, np_val)
            input_nodes.append(name)
        ctx.insert_new_node_on_input(node, "Quant", input_nodes, name=node.name + "_bias_quantizer", **attr, domain="qonnx")

    if quantizers.get("activation"):
        # removes node if added earlier
        remove_node_id = node.input[0]
        remove_node = ctx.get_node_by_output(remove_node_id)
        ctx.replace_all_inputs(node.input[0], remove_node.input[0], ops=None)
        dtypes = [ctx.get_dtype(node.output[0])]
        quant_params = get_quant_params(None, quantizers["activation"])
        attr = quant_params["attributes"]
        input_nodes = [node.output[0]]
        for key in quant_params["inputs"].keys():
            name = f"{node.name}_activation_quantizer_{key}"
            np_val = np.asarray(quant_params["inputs"][key])
            ctx.make_const(name, np_val)
            input_nodes.append(name)
        quant_act_node = ctx.make_node(
            "Quant",
            input_nodes,
            dtypes=dtypes,
            name=node.input[0] + "_activation_quantizer",
            attr=attr,
            domain="qonnx",
        )
        ctx.insert_node_on_output(quant_act_node, node.output[0])
        ctx.set_shape(quant_act_node.output[0], ctx.get_shape(node.output[0]))


def relu_handler(ctx, node, name, args):
    DirectOp.version_1(ctx, node)
    qact_handler(ctx, node, name, args)


def identity_handler(ctx, node, name, args):
    DirectOp.version_1(ctx, node)
    qact_handler(ctx, node, name, args)
