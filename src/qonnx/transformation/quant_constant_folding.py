import warnings

from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name


class FoldTransposeIntoQuantInit(Transformation):
    """
    Fueses a Transpose node into the initalizer of a Quant node.
    """

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        # Find transpose nodes, which have Quant node with initilizer upstream.
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Transpose":
                predecessors = model.find_direct_predecessors(n)
                # Check if we reached the top of the graph
                if predecessors is None:
                    continue
                predecessor = predecessors[0]
                if predecessor.op_type == "Quant" or predecessor.op_type == "BipolarQuant":
                    for inp in predecessor.input:
                        if not isinstance(model.get_initializer(inp), type(None)):
                            # Explicitly apply the transpose to the initializers
                            # of the previous node
                            target_tensor = model.get_initializer(inp)
                            if target_tensor is None:
                                warnings.warn(
                                    f"Cannot fold transpose {n} into Quant/BipolarQuant node {predecessor}, "
                                    f"due to not initialized tensor: {inp}. "
                                    f"Exiting FoldTransposeIntoQuantInit transformation."
                                )
                                return model, False
                            # Make sure the tensor has the correct shape
                            perm = get_by_name(n.attribute, "perm")
                            if perm is None:
                                target_tensor = target_tensor.transpose()
                                model.set_initializer(inp, target_tensor)
                                graph_modified = True
                            elif len(perm.ints) == len(target_tensor.shape):
                                target_tensor = target_tensor.transpose(perm.ints)
                                model.set_initializer(inp, target_tensor)
                                graph_modified = True
                    # Reconnect predecessor and delete transpose node
                    predecessor.output[0] = n.output[0]
                    graph.node.remove(n)

                    return model, graph_modified

        return model, graph_modified
