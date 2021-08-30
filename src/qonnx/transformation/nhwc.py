import warnings
from onnx import TensorProto, helper

from finn.transformation.base import Transformation
from finn.transformation.general import RemoveUnusedTensors
from finn.transformation.infer_shapes import InferShapes
from finn.util.basic import get_by_name

# ToDo: Similarly to the ops, this should maybe get moved from finn-base into qonnx.
# ToDo: Should these parameters move into a parent class for all NHWC trafos?
# ToDo: I also need some of these parameters in the nhwc op wrappers, so maybe this should get moved to a location,
#  where both, the ops and the trafos can access it.
# Standard ONNX nodes which require a NHWC data format to function properly
_nchw_node_types = ["Conv", "MaxPool", "BatchNormalization"]
_to_chan_last_args = {
    3: (0, 2, 1),
    4: (0, 2, 3, 1),
}
_to_chan_first_args = {
    3: (0, 2, 1),
    4: (0, 3, 1, 2),
}

# Nodes, which do not modify the shape of the tensor
# And modify all values in the same way.
_move_through_nodes = ["Quant", "Relu"]

# Nodes, which do not modify the shape of the tensor,
# And modify all values in the same way, if the second tensor is a scalar.
_move_through_nodes_if_scalar = ["Mul", "Div", "Sub", "Add"]


class ConvertToNHWCAndClean(Transformation):
    """
    Converts data layout dependent nodes to NHWC nodes and inserts transformations.
    Then it tries to eliminate as many transformations as possible and moves the
    still existing ones as far upstream as possible.
    """

    def apply(self, model):
        model = model.transform(FuseTransposeIntoQuantInit())
        model = model.transform(InsertNHWCDomainsAndTrafos())
        max_tries = 100
        for i in range(max_tries):
            initial_model_string = model.model.SerializeToString()
            # Apply RemoveConsecutiveChanFirstAndChanLastTrafos
            model = model.transform(RemoveConsecutiveChanFirstAndChanLastTrafos())

            # Apply MoveChanLastUpstream
            model = model.transform(MoveChanLastUpstream())

            # Run RemoveConsecutiveChanFirstAndChanLastTrafos again,
            # Technically only required if something changed in the previous trafo
            model = model.transform(RemoveConsecutiveChanFirstAndChanLastTrafos())

            # Apply MoveChanLastDownStream
            model = model.transform(MoveChanFirstDownstream())

            # Run RemoveConsecutiveChanFirstAndChanLastTrafos again,
            # Technically only required if something changed in the previous trafo
            model = model.transform(RemoveConsecutiveChanFirstAndChanLastTrafos())

            # Apply AbsorbChanFirstIntoMatMul
            model = model.transform(AbsorbChanFirstIntoMatMul())

            # Check if the model changed
            new_model_string = model.model.SerializeToString()
            if not (initial_model_string == new_model_string):
                # Do some cleanup
                model = model.transform(RemoveUnusedTensors())
                model = model.transform(InferShapes())
            else:
                break

        return model, False


class InsertNHWCDomainsAndTrafos(Transformation):
    """
    Inserts NHWC domain, where required and also inserts required transposes.
    """

    def apply(self, model):
        # ToDo: Add a check that all tensors have shape settings,
        #  otherwise some of the NHWC shape inference breaks
        graph = model.graph
        node_ind = 0
        graph_modified = False
        # Find nodes, where the domain should be changed
        for n in graph.node:
            node_ind += 1
            if (n.op_type in _nchw_node_types) and (n.domain == ""):
                running_node_index = node_ind
                # Insert transformation nodes for input nodes
                input_tensors = n.input
                # Skip for BatchNorm and 2D input tensors,
                # these contain only channels and need no transpose.
                # ToDo: Also support these BatchNorms
                NCHW_shape = model.get_tensor_shape(input_tensors[0])
                if n.op_type == "BatchNormalization" and len(NCHW_shape) == 2:
                    continue

                for i, inp in enumerate(input_tensors):
                    # Skip higher "order" inputs of the Batch-Norm,
                    # these don't need a transpose.
                    if n.op_type == "BatchNormalization" and i > 0:
                        continue
                    # Get the shape of the input tensor
                    # and convert it to the shape for the intermediate tensor
                    NCHW_shape = model.get_tensor_shape(inp)
                    ndim = len(NCHW_shape)
                    assert ndim == 3 or ndim == 4, "Channels last conversion is only available for 3D and 4D tensors."
                    NHWC_shape = [NCHW_shape[idx] for idx in _to_chan_last_args[ndim]]
                    # Intermediate tensor
                    inp_trans_out = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        NHWC_shape,
                    )
                    graph.value_info.append(inp_trans_out)
                    inp_trans_out = inp_trans_out.name

                    # channels last transpose
                    inp_trans_node = helper.make_node("Transpose", [inp], [inp_trans_out], perm=_to_chan_last_args[ndim])
                    graph.node.insert(running_node_index, inp_trans_node)
                    running_node_index += 1

                    # Attach to original node
                    n.input[i] = inp_trans_out

                # Insert transformation nodes for output nodes
                output_tensors = n.output
                for i, outp in enumerate(output_tensors):
                    NCHW_shape = model.get_tensor_shape(outp)
                    ndim = len(NCHW_shape)
                    assert ndim == 3 or ndim == 4, "Channels last conversion is only available for 3D and 4D tensors."
                    NHWC_shape = [NCHW_shape[idx] for idx in _to_chan_last_args[ndim]]
                    # Intermediat tensor
                    outp_trans_in = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        NHWC_shape,
                    )
                    graph.value_info.append(outp_trans_in)
                    outp_trans_in = outp_trans_in.name

                    # NCHW -> NHWC transpose
                    outp_trans_node = helper.make_node("Transpose", [outp_trans_in], [outp], perm=_to_chan_first_args[ndim])
                    graph.node.insert(running_node_index, outp_trans_node)
                    running_node_index += 1

                    # Attach to original node
                    n.output[i] = outp_trans_in

                # Modify domain
                n.domain = "qonnx.custom_op.nhwc"
                # Set modified flag
                graph_modified = True

        return model, graph_modified


class RemoveConsecutiveChanFirstAndChanLastTrafos(Transformation):
    """
    Remove two consecutive transformations, which would do:
    (NHWC -> NCHW) -> (NCHW -> NHWC)
    Or more concrete, the first converts to channels first and the second to channels last.
    """

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False

        # Find fitting transpose node pairs
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Transpose":

                # Check that this is a "to chan first" trafo
                perm_1 = get_by_name(n.attribute, "perm")
                if list(_to_chan_first_args) == perm_1.ints:

                    successor_nodes = model.find_direct_successors(n)
                    assert len(successor_nodes) == 1, (
                        "Transpose nodes should only have one output," " I don't think more than one would even be possible."
                    )
                    successor_node = successor_nodes[0]

                    if successor_node.op_type == "Transpose":
                        # Check that this is a "to chan last" trafo,
                        # if so both can get removed.
                        perm_2 = get_by_name(successor_node.attribute, "perm")
                        if list(_to_chan_last_args) == perm_2.ints:
                            # Connect original input to new output
                            input_tensor = n.input[0]
                            output_tensor_name = successor_node.output[0]

                            target_nodes = model.find_direct_successors(successor_node)
                            assert len(target_nodes) == 1, (
                                "Transpose nodes should only have one output,"
                                " I don't think more than one would even be possible."
                            )

                            target_node = target_nodes[0]
                            for i, inp in enumerate(target_node.input):
                                if inp == output_tensor_name:
                                    target_node.input[i] = input_tensor

                            # remove old nodes
                            graph.node.remove(n)
                            graph.node.remove(successor_node)

                            graph_modified = True
        return model, graph_modified


class MoveChanLastUpstream(Transformation):
    """
    Moves channel last transformations further upstream.
    """

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        # Find transpose nodes, which are "to chan last" trafos
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Transpose":
                perm = get_by_name(n.attribute, "perm")
                if list(_to_chan_last_args) == perm.ints:
                    predecessors = model.find_direct_predecessors(n)
                    # Check if we reached the top of the graph
                    if predecessors is None:
                        continue
                    assert len(predecessors) == 1, (
                        "Transpose nodes should only have one input, " "I don't think more than one would even be possible."
                    )
                    predecessor = predecessors[0]

                    # Check if we can simply move through the previous node
                    move_through_valid = predecessor.op_type in _move_through_nodes
                    # Check if we have a node, which applies a scalar change,
                    # then we can also move through.
                    if predecessor.op_type in _move_through_nodes_if_scalar:
                        second_inp_shape = model.get_tensor_shape(predecessor.input[1])
                        if second_inp_shape == [1] or second_inp_shape == []:
                            move_through_valid |= True

                    # Apply move through trafo if possible
                    if move_through_valid:
                        # Input tensors are always input 0
                        inp = predecessor.input[0]
                        if isinstance(model.get_initializer(inp), type(None)):
                            # Swap around node "predecessor" and "n"
                            # collect tensors
                            tensor_1 = inp
                            tensor_2 = n.input[0]
                            tensor_3 = n.output[0]
                            # Now connect the tensors to the nodes again,
                            # but in different order
                            n.input[0] = tensor_1
                            n.output[0] = tensor_2
                            predecessor.input[0] = tensor_2
                            predecessor.output[0] = tensor_3

                            # Change the shape of the middle tensor
                            target_shape = model.get_tensor_shape(tensor_3)
                            model.set_tensor_shape(tensor_2, target_shape)

                            graph_modified = True
                        else:
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


class MoveChanFirstDownstream(Transformation):
    """
    Moves channel first transformations further downstream.
    """

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        # Find transpose nodes, which are "to chan first" trafos
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Transpose":
                perm = get_by_name(n.attribute, "perm")
                if list(_to_chan_first_args) == perm.ints:
                    successors = model.find_direct_successors(n)
                    assert len(successors) == 1, "Transpose nodes should only have one output"
                    successor = successors[0]

                    # Check if we can simply move through the next node
                    move_through_valid = successor.op_type in _move_through_nodes
                    # Check if we have a node, which applies a scalar change,
                    # then we can also move through.
                    if successor.op_type in _move_through_nodes_if_scalar:
                        second_inp_shape = model.get_tensor_shape(successor.input[1])
                        if second_inp_shape == [1] or second_inp_shape == []:
                            move_through_valid |= True
                    # Apply move through trafo if possible
                    if move_through_valid:
                        # Collect all tensors connecting n and successor
                        # and surrounding nodes
                        tensor_1 = n.input[0]
                        tensor_2 = n.output[0]
                        tensor_3 = successor.output[0]
                        # Now connect the tensors to the nodes again,
                        # but in different order
                        successor.input[0] = tensor_1
                        successor.output[0] = tensor_2
                        n.input[0] = tensor_2
                        n.output[0] = tensor_3

                        # Change the shape of the middle tensor
                        target_shape = model.get_tensor_shape(tensor_1)
                        model.set_tensor_shape(tensor_2, target_shape)

                        graph_modified = True
                        return model, graph_modified

        return model, graph_modified


class FuseTransposeIntoQuantInit(Transformation):
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
                assert len(predecessors) == 1, (
                    "Transpose nodes should only have one input, " "I don't think more than one would even be possible."
                )
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


class AbsorbChanFirstIntoMatMul(Transformation):
    """
    Removes a transpose to channels first node if it is in front of a Flatten and
    MatMul node.

    The channels first transpose is fused into the initializer of the Quant node acting
    as a weight tensor for the MatMul node.
    Reshape nodes with shape [1, -1] are also supported instead of Flatten nodes.
    Independent of whether the flattening operation was performed by a Flatten node
    or a Resphape node, a Flatten node will be reinserted in-front of the MatMul node.

    Note: This transformation removes some of the tensor shapes on the down-stream path.
     Thus running shape inference afterwards is advised.
    """

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        node_ind = 0
        for n in graph.node:
            node_ind += 1
            # also support implicit flatten via reshape, e.g. reshape(1,-1)
            if n.op_type == "Flatten" or n.op_type == "Reshape":
                ishape = model.get_tensor_shape(n.input[0])
                oshape = model.get_tensor_shape(n.output[0])
                if ishape is None or oshape is None:
                    continue
                if len(oshape) == 2 and ishape[0] == oshape[0]:
                    producer = model.find_producer(n.input[0])
                    if producer.op_type == "Transpose":
                        # transpose + flatten, absorb into following node
                        transp_node = producer
                        # check if transpose converts NHWC to NCHW
                        perms = get_by_name(transp_node.attribute, "perm").ints
                        if list(_to_chan_first_args) == perms:
                            producer = model.find_producer(transp_node.input[0])
                            consumer = model.find_consumer(n.output[0])
                            if consumer.op_type == "MatMul":
                                b_shape = model.get_tensor_shape(consumer.input[1])
                                mw = b_shape[0]
                                mh = b_shape[1]
                                (b, h, w, c) = model.get_tensor_shape(transp_node.input[0])
                                # Get the weight initilizer
                                quant_node = model.find_producer(consumer.input[1])
                                if quant_node.op_type == "Quant":
                                    W = model.get_initializer(quant_node.input[0])
                                else:
                                    warnings.warn(f"Could not find weight initializer for " f"MatMul: {consumer.name}")
                                    continue
                                W_new = W.reshape(c, h, w, mh)
                                W_new = W_new.transpose((1, 2, 0, 3))
                                W_new = W_new.reshape(mw, mh)
                                model.set_initializer(quant_node.input[0], W_new)
                                # remove transpose & flatten nodes
                                consumer.input[0] = transp_node.input[0]
                                graph.node.remove(n)
                                graph.node.remove(transp_node)

                                # Insert a Flatten node in front of the MatMul
                                inp_tensor_name = consumer.input[0]
                                # Intermediate tensor
                                out_tensor = helper.make_tensor_value_info(
                                    model.make_new_valueinfo_name(),
                                    TensorProto.FLOAT,
                                    None,
                                )
                                graph.value_info.append(out_tensor)
                                out_tensor_name = out_tensor.name

                                # Attach to MatMul output
                                consumer.input[0] = out_tensor_name

                                # Flatten node
                                flat_node = helper.make_node(
                                    "Flatten",
                                    [inp_tensor_name],
                                    [out_tensor_name],
                                    axis=1,
                                )
                                graph.node.insert(node_ind, flat_node)

                                graph_modified = True
                            else:
                                warnings.warn(
                                    "Could not absorb transpose->flatten \
                                    into subsequent node"
                                )
        return model, graph_modified
