import logging
import numpy as np
from tf2onnx.late_rewriters import channel_order_rewriters
from tf2onnx.onnx_opset.math import DirectOp, MatMul
from tf2onnx.onnx_opset.nn import BiasAdd, ConvOp

from qonnx.custom_op.general.quant import quant

from .quantizers import get_quant_params

logger = logging.getLogger(__name__)


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


def check_tensor_is_representable(tensor, quant_params, node):
    "Gives a Warning iftensor is not representable with the providede quantization settings"
    qtensor = quant(
        inp_tensor=np.array(tensor),
        scale=np.array(quant_params["inputs"]["scale"]),
        zeropt=np.array(quant_params["inputs"]["zero_point"]),
        bitwidth=np.array(quant_params["inputs"]["bit_width"]),
        signed=quant_params["attributes"]["signed"],
        narrow=quant_params["attributes"]["narrow"],
        rounding_mode=quant_params["attributes"]["rounding_mode"],
    )
    if not np.array_equal(tensor, qtensor):
        logger.warn(
            f"Tensor of node: {node.name} is not representable with the provided quantization settings: {quant_params}"
        )


def _add_quant_node_on_input(ctx, node, quantizer_cfg, input_ind):
    weights = node.inputs[input_ind].get_tensor_value(as_list=True)
    quant_params = get_quant_params(weights, quantizer_cfg)
    check_tensor_is_representable(weights, quant_params, node)
    attr = quant_params["attributes"]
    input_nodes = [node.input[input_ind]]
    for key in quant_params["inputs"].keys():
        name = f"{node.name}_{input_ind}_quantizer_{key}"
        np_val = np.asarray(quant_params["inputs"][key])
        ctx.make_const(name, np_val)
        input_nodes.append(name)
    if quant_params["inputs"]["bit_width"] == 1 and attr["signed"] == 1:
        quant_node = ctx.insert_new_node_on_input(
            node, "BipolarQuant", input_nodes[0:2], name=node.name + f"_{input_ind}_quantizer", **dict(), domain="qonnx"
        )
    else:
        quant_node = ctx.insert_new_node_on_input(
            node, "Quant", input_nodes, name=node.name + f"_{input_ind}_quantizer", **attr, domain="qonnx"
        )
    if quantizer_cfg["class_name"] == "quantized_bits":
        bits = quantizer_cfg["config"]["bits"]
        integer = quantizer_cfg["config"]["integer"]
        keep_negative = quantizer_cfg["config"]["keep_negative"]
        if bits == integer + keep_negative:
            scale_node = ctx.make_const(
                name=node.name + f"_{input_ind}_scale", np_val=quant_params["inputs"]["scale"].astype(np.float32)
            )
            ctx.insert_new_node_on_output(
                op_type="Mul",
                output_name=quant_node.output[0],
                name=node.name + f"_{input_ind}_requantizer",
                inputs=[quant_node.output[0], scale_node.name],
            )


def qlayer_handler(ctx, node, name, args):
    all_quantizers = args[0]
    keras_name = _extract_node_name(node, all_quantizers)
    if not keras_name:
        return  # Not found in quantizers, nothing to do
    quantizers = all_quantizers[keras_name]

    if quantizers.get("kernel_quantizer_cfg"):
        _add_quant_node_on_input(ctx, node, quantizers["kernel_quantizer_cfg"], 1)

    if quantizers.get("bias_quantizer_cfg") and len(node.input) == 3:
        _add_quant_node_on_input(ctx, node, quantizers["bias_quantizer_cfg"], -1)

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
        if "auto" in quantizers["activation"]:
            if not node.graph.get_node_by_output(node.input[0]).is_const():
                raise AttributeError(
                    f"Automatic quantizers (auto/auto_po2) must have a const input. Invalid topology at node: {name}."
                )
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

    if quantizers.get("bias_quantizer_cfg"):
        bias = node.inputs[1].get_tensor_value(as_list=True)
        quant_params = get_quant_params(bias, quantizers["bias_quantizer_cfg"])
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
