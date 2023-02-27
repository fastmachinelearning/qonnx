import warnings
from onnx import TensorProto, helper

from qonnx.analysis.topology import is_linear
from qonnx.custom_op import channels_last
from qonnx.custom_op.channels_last.base_wrapped_op import to_channels_first_args, to_channels_last_args
from qonnx.transformation.base import Transformation
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.make_input_chanlast import MakeInputChannelsLast
from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit
from qonnx.util.basic import get_by_name

# Standard ONNX nodes which require a ChannelsLast data format to function properly
_channelsLast_node_types = list(channels_last.custom_op.keys())

# Nodes, which do not modify the shape of the tensor
# And modify all values in the same way.
_move_through_nodes = ["Quant", "Relu"]

# Nodes, which do not modify the shape of the tensor,
# And modify all values in the same way, if the second tensor is a scalar.
_move_through_nodes_if_scalar = ["Mul", "Div", "Sub", "Add"]


class ConvertToChannelsLastAndClean(Transformation):
    """
    Converts data layout dependent nodes to ChannelsLast nodes and inserts transformations.
    Then it tries to eliminate as many transformations as possible and moves the
    still existing ones as far upstream as possible.

    :param make_input_channels_last: Also makes the input of the network channels last,
        otherwise a transpose node will be left at the beginning of the network. Defaults to False
    :type make_input_channels_last: bool

    """

    def __init__(self, make_input_channels_last=False):
        super().__init__()
        self._make_input_channels_last = make_input_channels_last

    def apply(self, model):
        assert model.analysis(is_linear)["is_linear"], "Only linear and non-branching models are supported at this moment."
        assert model.check_all_tensor_shapes_specified(), (
            "All tensor shapes must be specified. " "Consider running InferShapes."
        )
        model = model.transform(InsertChannelsLastDomainsAndTrafos())
        initial_model_string = model.model.SerializeToString()
        # Apply RemoveConsecutiveChanFirstAndChanLastTrafos
        model = model.transform(RemoveConsecutiveChanFirstAndChanLastTrafos())

        # Apply MoveChanLastUpstream and fold into initializers
        model = model.transform(MoveChanLastUpstream())
        model = model.transform(FoldTransposeIntoQuantInit())

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

        if self._make_input_channels_last:
            model = model.transform(MakeInputChannelsLast())
            model = model.transform(RemoveConsecutiveChanFirstAndChanLastTrafos())

        # Check if the model changed
        new_model_string = model.model.SerializeToString()

        # Do small cleanup, which isn't done by the cleanup in the normal transformation
        model = model.transform(InferShapes())
        model = model.transform(FoldConstants())

        # Check if the model changed
        model_changed = initial_model_string != new_model_string

        return model, model_changed


class InsertChannelsLastDomainsAndTrafos(Transformation):
    """
    Inserts ChannelsLast domain, where required and also inserts required transposes.
    """

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        # Find nodes, where the domain should be changed
        for n in graph.node:
            node_ind += 1
            if (n.op_type in _channelsLast_node_types) and (n.domain == ""):
                running_node_index = node_ind
                # Insert transformation nodes for input nodes
                input_tensors = n.input
                # Skip for BatchNorm and 2D input tensors,
                # these contain only channels and need no transpose.
                chanFirst_shape = model.get_tensor_shape(input_tensors[0])
                if n.op_type == "BatchNormalization" and len(chanFirst_shape) == 2:
                    continue

                for i, inp in enumerate(input_tensors):
                    # Skip higher "order" inputs of the Batch-Norm,
                    # these don't need a transpose.
                    if n.op_type == "BatchNormalization" and i > 0:
                        continue
                    # Skip Conv bias since it doesn't need a transpose
                    if n.op_type == "Conv" and i == 2:
                        continue
                    # Get the shape of the input tensor
                    # and convert it to the shape for the intermediate tensor
                    chanFirst_shape = model.get_tensor_shape(inp)
                    ndim = len(chanFirst_shape)
                    assert ndim == 3 or ndim == 4, "Channels last conversion is only available for 3D and 4D tensors."
                    chanLast_shape = [chanFirst_shape[idx] for idx in to_channels_last_args(ndim)]
                    # Intermediate tensor
                    inp_trans_out = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        chanLast_shape,
                    )
                    graph.value_info.append(inp_trans_out)
                    inp_trans_out = inp_trans_out.name

                    # channels last transpose
                    inp_trans_node = helper.make_node("Transpose", [inp], [inp_trans_out], perm=to_channels_last_args(ndim))
                    graph.node.insert(running_node_index, inp_trans_node)
                    running_node_index += 1

                    # Attach to original node
                    n.input[i] = inp_trans_out

                # Insert transformation nodes for output nodes
                output_tensors = n.output
                for i, outp in enumerate(output_tensors):
                    chanFirst_shape = model.get_tensor_shape(outp)
                    ndim = len(chanFirst_shape)
                    assert ndim == 3 or ndim == 4, "Channels last conversion is only available for 3D and 4D tensors."
                    chanLast_shape = [chanFirst_shape[idx] for idx in to_channels_last_args(ndim)]
                    # Intermediat tensor
                    outp_trans_in = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        chanLast_shape,
                    )
                    graph.value_info.append(outp_trans_in)
                    outp_trans_in = outp_trans_in.name

                    # ChannelsFirst -> ChannelsLast transpose
                    outp_trans_node = helper.make_node(
                        "Transpose", [outp_trans_in], [outp], perm=to_channels_first_args(ndim)
                    )
                    graph.node.insert(running_node_index, outp_trans_node)
                    running_node_index += 1

                    # Attach to original node
                    n.output[i] = outp_trans_in

                # Modify domain
                n.domain = "qonnx.custom_op.channels_last"
                # Set modified flag
                graph_modified = True

        return model, graph_modified


class RemoveConsecutiveChanFirstAndChanLastTrafos(Transformation):
    """
    Remove two consecutive transformations, which would do:
    (ChannelsLast -> ChannelsFirst) -> (ChannelsFirst -> ChannelsLast)
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
                # Check the input shape and make sure we support it
                input_shape = model.get_tensor_shape(n.input[0])
                # Check that this is a "to chan first" trafo
                perm_1 = get_by_name(n.attribute, "perm")
                ndim = len(input_shape)
                if list(to_channels_first_args(ndim)) == perm_1.ints:
                    successor_nodes = model.find_direct_successors(n)
                    if successor_nodes is None:
                        continue
                    successor_node = successor_nodes[0]

                    if successor_node.op_type == "Transpose":
                        # Check that this is a "to chan last" trafo,
                        # if so both can get removed.
                        perm_2 = get_by_name(successor_node.attribute, "perm")
                        if list(to_channels_last_args(ndim)) == perm_2.ints:
                            # Connect original input to new output
                            input_tensor = n.input[0]
                            output_tensor_name = successor_node.output[0]

                            target_nodes = model.find_direct_successors(successor_node)
                            if target_nodes is None:
                                continue

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
                # Check the input shape and make sure we support it
                input_shape = model.get_tensor_shape(n.input[0])
                ndim = len(input_shape)
                perm = get_by_name(n.attribute, "perm")
                if list(to_channels_last_args(ndim)) == perm.ints:
                    predecessors = model.find_direct_predecessors(n)
                    # Check if we reached the top of the graph
                    if predecessors is None:
                        continue
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
                            return model, graph_modified
                        # else:
                        #     # Explicitly apply the transpose to the initializer
                        #     # of the previous node
                        #     target_tensor = model.get_initializer(inp)
                        #     target_tensor = target_tensor.transpose(perm.ints)
                        #     model.set_initializer(inp, target_tensor)
                        #     # Reconnect predecessor and delete transpose node
                        #     predecessor.output[0] = n.output[0]
                        #     graph.node.remove(n)
                        #
                        #     graph_modified = True
                        # return model, graph_modified

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
                # Check the input shape and make sure we support it
                input_shape = model.get_tensor_shape(n.input[0])
                ndim = len(input_shape)
                perm = get_by_name(n.attribute, "perm")
                if list(to_channels_first_args(ndim)) == perm.ints:
                    # Do not move the node, if it is at the top of the graph,
                    # this is a strange edge case, for 1D networks, where channels last and channels first trafos
                    # are identical.
                    predecessors = model.find_direct_predecessors(n)
                    if predecessors is None:
                        continue

                    successors = model.find_direct_successors(n)
                    if successors is None:
                        continue
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


class AbsorbChanFirstIntoMatMul(Transformation):
    """
    Removes a transpose to channels first node if it is in front of a Flatten and
    MatMul (or Gemm) node.

    The channels first transpose is fused into the initializer of the Quant node acting
    as a weight tensor for the MatMul/Gemm node.
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
                    if (producer is not None) and (producer.op_type == "Transpose"):
                        # transpose + flatten, absorb into following node
                        transp_node = producer
                        # Check the input shape and make sure we support it
                        input_shape = model.get_tensor_shape(transp_node.input[0])
                        # check if transpose converts ChannelsLast to ChannelsFirst
                        ndim = len(input_shape)
                        perms = get_by_name(transp_node.attribute, "perm").ints
                        if list(to_channels_first_args(ndim)) == perms:
                            producer = model.find_producer(transp_node.input[0])
                            consumer = model.find_consumer(n.output[0])
                            if (consumer is not None) and ((consumer.op_type == "MatMul") or (consumer.op_type == "Gemm")):
                                # Gemm supports transposes on B, so we need to account for that
                                b_transposed = False
                                if consumer.op_type == "Gemm":
                                    b_transposed = bool(get_by_name(consumer.attribute, "transB"))

                                b_shape = model.get_tensor_shape(consumer.input[1])
                                if b_transposed:
                                    b_shape = list(reversed(b_shape))
                                mw = b_shape[0]
                                mh = b_shape[1]
                                if ndim == 4:
                                    (b, h, w, c) = model.get_tensor_shape(transp_node.input[0])
                                elif ndim == 3:
                                    (b, h, c) = model.get_tensor_shape(transp_node.input[0])
                                    w = 1
                                else:
                                    raise ValueError(
                                        f"Inputs of dimensionality ndim={ndim}, are currently not supported "
                                        f"for merging transposes into the matrix multiply weight tensor"
                                    )
                                # Get the weight initilizer
                                quant_node = model.find_producer(consumer.input[1])
                                if quant_node is None:
                                    W = model.get_initializer(consumer.input[1])
                                else:
                                    if quant_node.op_type == "Quant":
                                        W = model.get_initializer(quant_node.input[0])
                                    else:
                                        warnings.warn(
                                            f"Could not find weight initializer for " f"MatMul/Gemm: {consumer.name}"
                                        )
                                        continue
                                if b_transposed:
                                    W = W.T
                                W_new = W.reshape(c, h, w, mh)
                                W_new = W_new.transpose((1, 2, 0, 3))
                                W_new = W_new.reshape(mw, mh)
                                if b_transposed:
                                    W_new = W_new.T
                                if quant_node is None:
                                    model.set_initializer(consumer.input[1], W_new)
                                else:
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
