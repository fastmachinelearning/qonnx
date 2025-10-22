import numpy as np

from .quantizers import get_quant_params


def get_hgq_onnx_handlers(all_quantizers):
    """Returns the handlers for each kind of layer

    Args:
        all_quantizers: All the quantizers of the model in dictionary format *check

    Returns:
        Dictionary containing the handler information for every type of layer
    """
    return {
        # NOTE: we replace the StatefulPartitionedCall layers with HGQIdentity layers
        # after them we are adding now FixedPoint layers for the quantitzation
        "StatefulPartitionedCall": (FixedPoint, ["FixedPoint", all_quantizers]),
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

    return None


def FixedPoint(ctx, node, name, args):
    all_quantizers = args[0]
    keras_name = _extract_node_name(node, all_quantizers)
    if not keras_name:
        return  # Not found in quantizers, nothing to do
    quantizers = all_quantizers[keras_name]
    # if we have overrides we are converting a FixedPointQuantizer from HGQ
    if quantizers.get("overrides"):
        quant_params = get_quant_params(None, quantizers)
        attr = quant_params["attributes"]
        input_nodes = [node.output[0]]
        for key in quantizers["inputs"].keys():
            name = f"{node.name}_FixedPointQuantizer_quantizer_{key}"
            np_val = np.asarray(quant_params["inputs"][key])
            ctx.make_const(name, np_val)
            input_nodes.append(name)
        quant_fixed_node = ctx.make_node(
            "FixedPoint",
            input_nodes,
            dtypes=None,  # TODO: we have to get the type here
            name=node.name + "_FixedPoint_quantizer",
            attr=attr,
            domain="qonnx",
        )
        ctx.insert_node_on_output(quant_fixed_node, node.output[0])
        ctx.set_shape(quant_fixed_node.output[0], ctx.get_shape(node.output[0]))
