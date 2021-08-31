from finn.transformation.base import Transformation
from finn.util.basic import get_by_name


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
                perm = get_by_name(n.attribute, "perm")
                predecessors = model.find_direct_predecessors(n)
                # Check if we reached the top of the graph
                if predecessors is None:
                    continue
                assert (
                    len(predecessors) == 1
                ), "Transpose nodes should only have one input, I don't think more than one would even be possible."
                predecessor = predecessors[0]
                if predecessor.op_type == "Quant":
                    inp = predecessor.input[0]
                    if not isinstance(model.get_initializer(inp), type(None)):
                        # Explicitly apply the transpose to the initializer
                        # of the previous node
                        target_tensor = model.get_initializer(inp)
                        target_tensor = target_tensor.transpose(perm.ints)
                        model.set_initializer(inp, target_tensor)
                        # Reconnect predecessor and delete transpose node
                        predecessor.output[0] = n.output[0]
                        graph.node.remove(n)

                        graph_modified = True
                        return model, graph_modified

        return model, graph_modified
