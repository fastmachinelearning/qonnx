# Protobuf onnx graph node type
from onnx import NodeProto
# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper
# QONNX graph transformations base class
from qonnx.transformation.base import Transformation
#  Gets items from protobuf by name
from qonnx.util.basic import get_by_name


# Tests whether a node is a quant-init, i.e., a quantizer with only initializer
# inputs
def is_quant_init(node: NodeProto, model: ModelWrapper):
    # Only handle existing Quant or BipolarQuant type nodes
    if node is not None and node.op_type in {"Quant", "BipolarQuant"}:
        # All inputs must have initializers, otherwise this is just a normal
        # quant, but not a quant-init
        return all(model.get_initializer(i) is not None for i in node.input)
    # Did not match the operator type
    return False


# Refactored version of the transformation properly checking whether the
# quantizer is *actually* QuantInit, i.e., all inputs have initializers
class FoldTransposeIntoQuantInit(Transformation):
    """
    Fuses a Transpose node into the initializers of a Quant node.
    """

    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # This transformation is triggered by finding a Transpose node
            if node.op_type == "Transpose":
                # Get the predecessors feeding into the transpose node
                predecessors = model.find_direct_predecessors(node)
                # The transform applies only to transpose with exactly one input
                if predecessors is None or len(predecessors) != 1:
                    # Note: Softly skip this note, maybe consider a hard failure
                    # at least in case there are multiple inputs?
                    continue
                # Check whether the predecessor is a quantizer with only
                # initializer inputs
                if is_quant_init(predecessors[0], model):
                    # Alias to the single predecessor node
                    quant_init = predecessors[0]
                    # Get the (optional) permutation indices of the transpose in
                    # case it is a multi-axis transpose
                    perm = get_by_name(node.attribute, "perm")
                    # Convert permutation indices to list of integers if it is
                    # given
                    perm = perm.ints if perm is not None else None
                    # Transpose all(!) initializer inputs of the quant node
                    for i in quant_init.input:
                        # Get the initializer tensor
                        # Note: No need to validate the presence of the
                        # initializer here, as we already tested this as the
                        # applicability condition above
                        tensor = model.get_initializer(i)
                        # Skip transposing the initializer if the number of
                        # dimensions do not match
                        if perm is not None and len(perm) != tensor.ndim:
                            # TODO: Soft skip ok or is this an error?
                            continue
                        # Transpose the tensor, optionally according to the
                        # permutation indices (perm might be None)
                        tensor = tensor.transpose(perm)
                        # Reassign the transposed initializer tensor
                        model.set_initializer(i, tensor)
                        # The graph has been modified, this needs to be reported
                        # back to the caller
                        graph_modified = True
                    # Rewire the graph to skip the transpose node
                    quant_init.output[0] = node.output[0]
                    # Remove the now absorbed transpose node
                    graph.node.remove(node)
        # TODO: Finalize by running shape inference as some other transforms do?
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified
