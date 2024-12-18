# Copyright (c) 2023 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of qonnx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import warnings
from copy import deepcopy
from onnx import TensorProto, helper

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op import channels_last
from qonnx.custom_op.channels_last.base_wrapped_op import to_channels_first_args, to_channels_last_args
from qonnx.transformation.base import Transformation
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import SortGraph
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.make_input_chanlast import MakeInputChannelsLast
from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit
from qonnx.util.basic import get_by_name
from qonnx.util.onnx import is_eltwise_optype

# Standard ONNX nodes which require a ChannelsLast data format to function properly
_channelsLast_node_types = list(channels_last.custom_op.keys())

# Nodes, which do not modify the shape of the tensor
# And modify all values in the same way.
_move_through_nodes = ["Quant", "Relu", "Selu", "LeakyRelu", "Sigmoid", "Tanh"]

# Nodes, which do not modify the shape of the tensor,
# And modify all values in the same way, if the second tensor is a scalar.
_move_through_nodes_if_scalar = ["Mul", "Div", "Sub", "Add"]


def get_transpose_perms(transpose_node, model):
    perm = get_by_name(transpose_node.attribute, "perm")
    ndim = len(model.get_tensor_shape(transpose_node.input[0]))
    if perm is None:
        # default perm is to reverse the dim order
        return list(range(ndim - 1, -1, -1))
    else:
        return list(perm.ints)


def move_transpose_past_eltwise(transpose_node, eltwise_node, model: ModelWrapper):
    t0 = transpose_node.input[0]
    t1 = transpose_node.output[0]
    t2 = eltwise_node.output[0]
    subgraph_inp_shape = model.get_tensor_shape(t0)
    ndim_inp = len(subgraph_inp_shape)
    perm = get_transpose_perms(transpose_node, model)

    # before: t0 -> transpose -> t1 -> eltwise -> t2
    # after: t0 -> eltwise -> t1 -> transpose -> t2
    # find the eltwise inp index fed by transpose
    transpose_in_ind = list(eltwise_node.input).index(t1)
    # check all inputs for the eltwise op:
    # we need to ensure those inputs get inverse-transposed
    # to keep the graph semantics intact
    for ind, eltwise_inp in enumerate(eltwise_node.input):
        if ind == transpose_in_ind:
            # the input that feeds from the original transpose
            # node will be implicitly inverse-transposed, since we'll be
            # moving that transpose node past the eltwise op
            continue
        inp_shape = model.get_tensor_shape(eltwise_inp)
        ndim = len(inp_shape)
        if ndim == 0:
            # scalar input, always broadcastable, no action needed
            continue
        elif ndim == ndim_inp:
            # input with matching dimensions, add inverse transpose
            new_t_inp = model.make_new_valueinfo_name()
            inv_perm = np.argsort(perm)
            new_transpose_node = helper.make_node("Transpose", [eltwise_inp], [new_t_inp], perm=inv_perm)
            t_shape = np.transpose(np.empty(inp_shape), axes=inv_perm).shape
            model.set_tensor_shape(new_t_inp, t_shape)
            eltwise_node.input[ind] = new_t_inp
            model.graph.node.append(new_transpose_node)
        else:
            # input with non-matching dimensions, assume broadcastable
            # first add Unsqueeze node to match number of dimensions
            unsqueeze_param_name = model.make_new_valueinfo_name()
            model.set_initializer(unsqueeze_param_name, np.asarray(list(range(ndim_inp - ndim)), dtype=np.int64))
            unsqueeze_out_name = model.make_new_valueinfo_name()
            new_unsqueeze_node = helper.make_node("Unsqueeze", [eltwise_inp, unsqueeze_param_name], [unsqueeze_out_name])
            unsqueeze_out_shape = np.expand_dims(np.empty(inp_shape), axis=tuple(range(ndim_inp - ndim))).shape
            model.set_tensor_shape(unsqueeze_out_name, unsqueeze_out_shape)
            model.graph.node.append(new_unsqueeze_node)
            # now add inverse transpose
            new_t_inp = model.make_new_valueinfo_name()
            inv_perm = np.argsort(perm)
            new_transpose_node = helper.make_node("Transpose", [unsqueeze_out_name], [new_t_inp], perm=inv_perm)
            t_shape = np.transpose(np.empty(unsqueeze_out_shape), axes=inv_perm).shape
            model.set_tensor_shape(new_t_inp, t_shape)
            eltwise_node.input[ind] = new_t_inp
            model.graph.node.append(new_transpose_node)
    # rewire to swap transpose and eltwise node order
    eltwise_node.input[transpose_in_ind] = t0
    eltwise_node.output[0] = t1
    transpose_node.input[0] = t1
    transpose_node.output[0] = t2
    # t1 tensor shape changes to inp_shape
    model.set_tensor_shape(t1, subgraph_inp_shape)
    model = model.transform(SortGraph())
    model = model.transform(FoldConstants())
    return model


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

    def apply(self, model: ModelWrapper):
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

        # Apply MoveChanLastDownStream and MoveTransposePastFork
        model = model.transform(MoveChanFirstDownstream())
        model = model.transform(MoveTransposePastFork())

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
                perm_1 = get_transpose_perms(n, model)
                ndim = len(input_shape)
                if list(to_channels_first_args(ndim)) == perm_1:
                    successor_nodes = model.find_direct_successors(n)
                    # skip if:
                    # - this Transpose has no successors (nothing to do)
                    # - this Transpose output is forking (cannot remove)
                    if successor_nodes is None or len(successor_nodes) > 1:
                        continue
                    successor_node = successor_nodes[0]

                    if successor_node.op_type == "Transpose":
                        # Check that this is a "to chan last" trafo,
                        # if so both can get removed.
                        perm_2 = get_transpose_perms(successor_node, model)
                        if list(to_channels_last_args(ndim)) == perm_2:
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

    def apply(self, model: ModelWrapper):
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
                perm = get_transpose_perms(n, model)
                if list(to_channels_last_args(ndim)) == perm:
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

                    # don't move through if the predecessor output is a fork
                    if model.is_fork_node(predecessor):
                        move_through_valid = False

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
        for node in graph.node:
            node_ind += 1
            if node.op_type == "Transpose":
                # Check the input shape and make sure we support it
                input_shape = model.get_tensor_shape(node.input[0])
                ndim = len(input_shape)
                perm = get_transpose_perms(node, model)
                if list(to_channels_first_args(ndim)) == perm:
                    # Do not move the node, if it is at the top of the graph,
                    # this is a strange edge case, for 1D networks, where channels last and channels first trafos
                    # are identical.
                    predecessors = model.find_direct_predecessors(node)
                    if predecessors is None:
                        continue

                    successors = model.find_direct_successors(node)
                    if successors is None:
                        continue
                    successor = successors[0]
                    transpose_node = node

                    if is_eltwise_optype(successor.op_type):
                        model = move_transpose_past_eltwise(transpose_node, successor, model)
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
                        perms = get_transpose_perms(transp_node, model)
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


class MoveOpPastFork(Transformation):
    """Move node operations past graph forks. Used when a node before a fork
    can be merged with nodes in the branches
    """

    def __init__(self, op_name_list):
        super().__init__()
        self.ops_to_move = op_name_list

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        nodes = [n for n in graph.node]
        node_ind = 0
        for node in nodes:
            node_ind += 1
            if node.op_type in self.ops_to_move and model.is_fork_node(node) and not model.is_join_node(node):
                # Restrict this transform to operations with constant parameters
                # Assuming parameters is in input 1
                if len(node.input) > 1:
                    op_init_param = model.get_initializer(node.input[1])
                else:
                    op_init_param = None

                # Check case when branches are empty and go
                # to the same node
                consumers = model.find_consumers(node.output[0])
                assert len(consumers) > 1, "Must have >1 consumer"
                unique_consumer = True
                for consum_node in consumers[1:]:
                    if consumers[0] != consum_node:
                        unique_consumer = False
                        break

                if unique_consumer:
                    continue

                for consumer_node in consumers[1:]:
                    # create new node
                    new_output_tensor_name = model.make_new_valueinfo_name()
                    if op_init_param is None:
                        new_inp_list = [node.input[0]]
                    else:
                        new_param_name = model.make_new_valueinfo_name()
                        new_inp_list = [node.input[0], new_param_name]
                        model.set_initializer(new_param_name, op_init_param)
                    new_node = deepcopy(node)
                    new_node.input[:] = new_inp_list
                    new_node.output[:] = [new_output_tensor_name]
                    graph.node.insert(node_ind, new_node)
                    node_ind += 1

                    # change consumer input tensor
                    graph.node.remove(consumer_node)
                    for idx, consumer_input in enumerate(consumer_node.input):
                        if consumer_input == node.output[0]:
                            consumer_node.input[idx] = new_output_tensor_name
                            break
                    else:
                        raise Exception("Consumer should have the current node output as input")

                    graph.node.insert(node_ind, consumer_node)

                graph_modified = True

        model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveAddPastFork(MoveOpPastFork):
    def __init__(self):
        super().__init__(["Add"])


class MoveMulPastFork(MoveOpPastFork):
    def __init__(self):
        super().__init__(["Mul"])


class MoveLinearPastFork(MoveOpPastFork):
    def __init__(self):
        super().__init__(["Add", "Mul"])


class MoveTransposePastFork(MoveOpPastFork):
    def __init__(self):
        super().__init__(["Transpose"])
