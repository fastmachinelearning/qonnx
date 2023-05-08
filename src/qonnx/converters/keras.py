import onnx
import tensorflow as tf
import tf2onnx
from collections import OrderedDict
from qkeras.utils import REGISTERED_LAYERS as QKERAS_LAYERS

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model

from .qkeras.onnx import get_qkeras_onnx_handlers
from .qkeras.qlayers import extract_quantizers_from_layer

_unsupported_layers = [
    # These require some extra work
    "QBatchNormalization",
    "QConv2DBatchnorm",
    "QDepthwiseConv2DBatchnorm",
]

# Skip remove_identity optimizer
del tf2onnx.optimizer._optimizers["remove_identity"]


def add_value_info_for_constants(model: onnx.ModelProto):
    """
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
    # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
    if model.ir_version < 4:
        return model

    def add_const_value_infos_to_graph(graph: onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.value_info}
        for init in graph.initializer:
            # Check it really is a constant, not an input
            if init.name in inputs:
                continue

            # The details we want to add
            elem_type = init.data_type
            shape = init.dims

            # Get existing or create new value info for this constant
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.value_info.add()
                vi.name = init.name

            # Even though it would be weird, we will not overwrite info even if it doesn't match
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField("shape"):
                # Ensure we set an empty list if the const is scalar (zero dims)
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim

        # Handle subgraphs
        for node in graph.node:
            for attr in node.attribute:
                # Ref attrs refer to other attrs, so we don't need to do anything
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)

    add_const_value_infos_to_graph(model.graph)
    return model


def _is_qkeras_model(model):
    """Check if the model has any qkeras layers, so we can handle the qkeras layers separately

    Args:
        model: the model we want to convert

    Returns:
        True if the model contains any qkeras layer
    """

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
    """Check if all the layers in the model are supported for conversion

    Args:
        model: the tf.keras model we want to convert

    Returns:
        Exception if an unsupported layer is found in the model
    """

    def iterate_model(model):
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                iterate_model(layer)
            elif layer.__class__.__name__ in _unsupported_layers:
                raise Exception("Currently unsupported layer found in QKeras model: {}".format(layer.__class__.__name__))

    iterate_model(model)


def _strip_qkeras_model(model):
    """Strip a qkeras model to obtain the keras model and obtain the quant nodes.

    Args:
        model: the tf.keras model we want to convert

    Returns:
        The stripped model, and the quantizers in a dictionary format
    """
    quantizers = OrderedDict()

    def extract_quantizers(layer):
        keras_cls_name, layer_cfg, layer_quantizers = extract_quantizers_from_layer(layer)
        if layer_quantizers:
            layer_quantizers = {
                k: None if v == "None" else v for k, v in layer_quantizers.items()
            }  # Get rid of 'None' strings
            layer_quantizers["input"] = layer.input.name
            quantizers[layer.name] = layer_quantizers

        layer_class = tf.keras.layers.__dict__.get(keras_cls_name, None)
        if layer_class is None:
            raise Exception("Cannot create Keras layer from QKeras class {}".format(keras_cls_name))

        return layer_class.from_config(layer_cfg)

    stripped_model = tf.keras.models.clone_model(model, clone_function=extract_quantizers)
    stripped_model.set_weights(model.get_weights())
    return stripped_model, quantizers


# tests run without this function
def _convert_quantizers_to_nodes(onnx_model, quantizers_dict):
    for node_name, quantizers in quantizers_dict.items():
        print(node_name, quantizers)

    for n in onnx_model.graph.node:
        print(n)

    return onnx_model.model


def from_keras(
    model,
    name="qkeras_to_qonnx_converted",
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
        custom_ops=custom_ops,
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
    # Set the first value of input/output shape to 1, currently this is set to unknown,
    # because it is technically the batch size
    if not (len(onnx_model.graph.input) == 1 and len(onnx_model.graph.output) == 1):
        raise ValueError("Qkeras to QONNX conversion only supports models with exactly one input and output.")
    inp_shape = onnx_model.get_tensor_shape(onnx_model.graph.input[0].name)
    out_shape = onnx_model.get_tensor_shape(onnx_model.graph.output[0].name)
    inp_shape[0] = 1
    out_shape[0] = 1
    onnx_model.set_tensor_shape(onnx_model.graph.input[0].name, inp_shape)
    onnx_model.set_tensor_shape(onnx_model.graph.output[0].name, out_shape)

    # Set all Quant output tensors to float32 datatype, otherwise they are undefined and crash ONNX execution
    qonnx_domain_ops = ["Quant", "Trunc", "BipolarQuant"]
    for q_op_type in qonnx_domain_ops:
        quant_nodes = onnx_model.get_nodes_by_op_type(q_op_type)
        q_node_outputs = [qn.output[0] for qn in quant_nodes]
        for tensor in onnx_model.graph.value_info:
            if tensor.name in q_node_outputs:
                tensor.type.tensor_type.elem_type = 1

    onnx_model = cleanup_model(onnx_model)
    onnx_model.model = add_value_info_for_constants(onnx_model.model)

    if output_path is not None:
        onnx_model.save(output_path)

    return onnx_model.model, external_storage
