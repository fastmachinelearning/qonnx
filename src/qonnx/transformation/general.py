# Copyright (c) 2020 Xilinx, Inc.
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
# * Neither the name of Xilinx nor the names of its
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

import json
import numpy as np
import warnings

# Protobuf onnx graph node type
from onnx import NodeProto  # noqa
from onnx import mapping
from toposort import toposort_flatten

import qonnx.util.basic as util
from qonnx.transformation.base import Transformation


class MovePadAttributeToTensor(Transformation):
    "Move padding info from attribute into input tensor for Pad nodes."

    def apply(self, model):
        run_again = False
        pad_nodes = model.get_nodes_by_op_type("Pad")
        for pad_node in pad_nodes:
            pads = util.get_by_name(pad_node.attribute, "pads")
            if pads is not None:
                assert len(pad_node.input) == 1
                pads_t = np.asarray(pads.ints)
                new_pad_name = model.make_new_valueinfo_name()
                model.set_initializer(new_pad_name, pads_t)
                pad_node.input.append(new_pad_name)
                pad_node.attribute.remove(pads)
            padval = util.get_by_name(pad_node.attribute, "value")
            if padval is not None:
                # non-float types will need type correction here
                input_vi = model.get_tensor_valueinfo(pad_node.input[0])
                pad_dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[input_vi.type.tensor_type.elem_type]
                padval_t = np.asarray(padval.f, pad_dtype)
                new_padval_name = model.make_new_valueinfo_name()
                model.set_initializer(new_padval_name, padval_t)
                pad_node.input.append(new_padval_name)
                pad_node.attribute.remove(padval)

        return (model, run_again)


class RemoveUnusedTensors(Transformation):
    """Remove any unused tensors in the graph by removing any initializers,
    ValueInfo and tensor annotations associated with it. Unused tensors do not
    appear as any input/output for any graph nodes.
    """

    def apply(self, model):
        graph_modified = False
        onnx_graph = model.model.graph
        # build a set of tensors that we actually use in the graph nodes
        used_tensors = set()
        for node in model.graph.node:
            for i in node.input:
                used_tensors.add(i)
            for o in node.output:
                used_tensors.add(o)
        # remove initializers, value_info and annotations that are not in the
        # used set of tensors, as determined by the graph node i/o
        for init in onnx_graph.initializer:
            if init.name not in used_tensors:
                onnx_graph.initializer.remove(init)
                graph_modified = True
        for vi in onnx_graph.value_info:
            if vi.name not in used_tensors:
                onnx_graph.value_info.remove(vi)
                graph_modified = True
        for qa in onnx_graph.quantization_annotation:
            if qa.tensor_name not in used_tensors:
                onnx_graph.quantization_annotation.remove(qa)
                graph_modified = True

        return (model, graph_modified)


class RemoveStaticGraphInputs(Transformation):
    "Remove any top-level graph inputs that have initializers."

    def apply(self, model):
        graph_modified = False
        for i in model.graph.input:
            if model.get_initializer(i.name) is not None:
                # move ValueInfo to internal (value_info) container
                model.graph.value_info.append(i)
                model.graph.input.remove(i)
                graph_modified = True

        return (model, graph_modified)


class GiveUniqueNodeNames(Transformation):
    """Give unique names to each node in the graph using enumeration, starting
    with given prefix (if specified in the constructor)."""

    def __init__(self, prefix=""):
        super().__init__()
        self.prefix = prefix

    def apply(self, model):
        optype_count = {}
        for n in model.graph.node:
            if n.op_type not in optype_count.keys():
                optype_count[n.op_type] = 0
            n.name = "%s%s_%d" % (self.prefix, n.op_type, optype_count[n.op_type])
            optype_count[n.op_type] += 1
        # return model_was_changed = False as single iteration is always enough
        return (model, False)


class GiveRandomTensorNames(Transformation):
    """Give random tensor names to all tensors."""

    def apply(self, model):
        names = model.get_all_tensor_names()
        for name in names:
            model.rename_tensor(name, util.random_string())
        # return model_was_changed = False as single iteration is always enough
        return (model, False)


class GiveUniqueTensorNames(Transformation):
    """Give unique tensor names to all tensors."""

    def apply(self, model):
        names = model.get_all_tensor_names()
        i = 0
        for name in names:
            model.rename_tensor(name, f"tensor_{i}")
            i += 1
        # return model_was_changed = False as single iteration is always enough
        return (model, False)


class GiveReadableTensorNames(Transformation):
    """Give more human-readable names to all internal tensors. You should
    apply GiveUniqueNodeNames prior to this transform to avoid empty node names,
    as the readable names are based on the node names."""

    def apply(self, model):
        # to ensure we can use rename_tensor safely (without renaming existing
        # tensors) we start by giving random names to all tensors
        model = model.transform(GiveUniqueTensorNames())
        graph = model.graph
        for n in graph.node:
            assert n.name != "", "Found empty node name"
            out_num = 0
            for o in n.output:
                model.rename_tensor(o, "%s_out%d" % (n.name, out_num))
                out_num += 1
            init_in_num = 0
            for i in n.input:
                if model.get_initializer(i) is not None:
                    model.rename_tensor(i, "%s_param%d" % (n.name, init_in_num))
                    init_in_num += 1
        # give special names to the model inputs and outputs
        for i, inp in enumerate(model.graph.input):
            iname = "global_in" if i == 0 else "global_in_%d" % i
            model.rename_tensor(inp.name, iname)
        for i, outp in enumerate(model.graph.output):
            oname = "global_out" if i == 0 else "global_out_%d" % i
            model.rename_tensor(outp.name, oname)
        # return model_was_changed = False as single iteration is always enough
        return (model, False)


class GiveUniqueParameterTensors(Transformation):
    """Make every parameter tensor unique. The aim is to avoid affecting
    other nodes apart from the one the system is currently operating on."""

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        seen_parameters = []
        for n in graph.node:
            # copy inputs since they may be modified
            node_inputs_list = [x for x in n.input]
            for input_idx, node_input in enumerate(node_inputs_list):
                # check if it's a parameter
                input_init = model.get_initializer(node_input)
                if input_init is None:
                    # dynamic input
                    continue

                # check if repeated
                if node_input not in seen_parameters:
                    # first occurance
                    seen_parameters += [node_input]
                    continue

                new_param_name = model.make_new_valueinfo_name()

                model.set_initializer(new_param_name, input_init)
                model.set_tensor_datatype(new_param_name, model.get_tensor_datatype(node_input))

                # point node input to new tensor
                n.input[input_idx] = new_param_name

        return (model, graph_modified)


class SortGraph(Transformation):
    """Returns the model with its node list sorted topologically.
    Any ONNX graph to be executed must have a topologically sorted node list,
    as dictated by the ONNX standard.
    """

    # Notes on SortGraph performance:
    # benchmark in  tests/transformation/test_sort_graph.py
    # The algorithm doesn't move initializers so its performance should only depend on
    # the number of nodes
    #
    # Relative order of magnitudes for time per step:
    # - Gather graph structure:       base
    # - Sort nodes:                   0.1 of base
    # - Remove and insert in order :  0.001 of base
    #
    # Notes:
    # Remove nodes and insert them in order:
    # Probably this is faster than copying initializers and more robust in general

    def apply(self, model):
        if len(model.graph.node) == 1:
            # single-node graph, nothing to sort
            return (model, False)
        # Gather graph structure
        graph_dependencies = {}
        node_list = [n for n in model.graph.node]  # I also need the list to remove the nodes
        for node_idx, n in enumerate(node_list):
            node_pred = model.find_direct_predecessors(n)
            if node_pred is None:
                if len(n.input) > 0:
                    # check if node inputs are connected to graph inputs or initializers
                    # if so, we can keep the node in the graph
                    for name in n.input:
                        if util.get_by_name(model.graph.initializer, name) or \
                           util.get_by_name(model.graph.input, name):
                            # this node is connected to graph inputs or initializers
                            # so we can keep it in the graph
                            graph_dependencies[node_idx] = set()
                            break
                # Will also eliminate nodes that are floating around for some reason
                continue

            node_dependencies = [node_list.index(pred) for pred in node_pred]
            graph_dependencies[node_idx] = set(node_dependencies)

        # Sort nodes
        sorted_node_indexes = toposort_flatten(graph_dependencies)

        # Remove nodes and insert them in order
        # Can't remove nodes before if I want to use model.find_direct_predecessors()
        for n in node_list:
            model.graph.node.remove(n)

        for new_idx, sorted_idx in enumerate(sorted_node_indexes):
            model.graph.node.insert(new_idx, node_list[sorted_idx])

        return (model, False)


class ConvertSubToAdd(Transformation):
    """Convert subtract-a-constant nodes to add-a-constant nodes."""

    def apply(self, model):
        graph = model.graph
        for n in graph.node:
            if n.op_type == "Sub":
                A = model.get_initializer(n.input[1])
                if A is not None:
                    n.op_type = "Add"
                    model.set_initializer(n.input[1], -A)
        # return model_was_changed = False as single iteration is always enough
        return (model, False)


class ConvertDivToMul(Transformation):
    """Convert divide by constant nodes to multiply by constant nodes."""

    def apply(self, model):
        graph = model.graph
        for n in graph.node:
            if n.op_type == "Div":
                A = model.get_initializer(n.input[1])
                if A is not None:
                    n.op_type = "Mul"
                    model.set_initializer(n.input[1], (1.0 / A).astype(A.dtype))
        # return model_was_changed = False as single iteration is always enough
        return (model, False)


class ApplyConfig(Transformation):
    """Applies node properties (attributes) from either a config dict or its JSON
    representation given as a filename.
    The JSON file can specify default values for particular op_types, as well
    as values for nodes with particular names. Example dict::

        {
        # set kernel_size = 3 for all nodes with op_type=Im2Col
        "Defaults" : {"kernel_size" : [3, ["Im2Col"]]},
        # set kernel_size = 7 for the particular node with name Im2Col_0
        "Im2Col_0" : {"kernel_size" : 7}
        }

    """

    def __init__(self, config, node_filter=lambda x: True):
        super().__init__()
        self.config = config
        self.node_filter = node_filter

    def apply(self, model):
        if isinstance(self.config, dict):
            model_config = self.config
        else:
            with open(self.config, "r") as f:
                model_config = json.load(f)

        used_configurations = ["Defaults"]
        missing_configurations = []

        # Configure network
        for node_idx, node in enumerate(model.graph.node):
            if not self.node_filter(node):
                continue
            try:
                node_config = model_config[node.name]
            except KeyError:
                missing_configurations += [node.name]
                node_config = {}

            from qonnx.custom_op.registry import getCustomOp

            try:
                inst = getCustomOp(node)
            except Exception:
                continue
            used_configurations += [node.name]

            # set specified defaults
            default_values = []
            for key, value in model_config["Defaults"].items():
                assert len(value) % 2 == 0
                if key not in model_config:
                    for val, op in zip(value[::2], value[1::2]):
                        default_values.append((key, val, op))
                        assert not (op == "all" and len(value) > 2)
            default_configs = {key: val for key, val, op in default_values if op == "all" or node.op_type in op}
            for attr, value in default_configs.items():
                inst.set_nodeattr(attr, value)

            # set node attributes from specified configuration
            for attr, value in node_config.items():
                inst.set_nodeattr(attr, value)

        # Configuration verification
        if len(missing_configurations) > 0:
            warnings.warn("\nNo HW configuration for nodes: " + ", ".join(missing_configurations))

        unused_configs = [x for x in model_config if x not in used_configurations]
        if len(unused_configs) > 0:
            warnings.warn("\nUnused HW configurations: " + ", ".join(unused_configs))

        # one iteration is enough
        return (model, False)


# Groups inputs by categories, i.e., groups dynamic inputs first, followed by
# initializers. Keeps order of inputs in each category.
def group_inputs_by_category(node: NodeProto, model):  # noqa
    # Select all dynamic inputs, which are those without initializer tensor
    dynamics = [i for i in node.input if model.get_initializer(i) is None]
    # Select all input which are initializers, which, by exclusion, are all
    # those not among the dynamic inputs
    initializers = [i for i in node.input if i not in dynamics]
    # Return lists of dynamic anc initializer inputs
    return dynamics, initializers


# Tidy-Up transformation sorting the inputs to all commutative operations to
# have initializer inputs last
class SortCommutativeInputsInitializerLast(Transformation):
    """
    Sorts inputs of nodes describing commutative operations to have initializer
    inputs last. This order of inputs is assumed by many other transformations.
    """

    # Set of supported commutative operations
    #   TODO: There might be more valid operations
    SUPPORTED_COMMUTATIVE_OPS = {"Add", "Mul", "And", "Or", "Xor", "Sum"}

    # Applies the transform to a whole model graph
    def apply(self, model):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Check whether this node is among the supported
            if node.op_type in self.SUPPORTED_COMMUTATIVE_OPS:
                # Group node inputs by category
                dynamics, initializers = group_inputs_by_category(node, model)
                # Flatten the grouped input list
                inputs = [*dynamics, *initializers]
                # Length of sorted and original input list must match
                assert len(inputs) == len(node.input)
                # Reassigned inputs from sorted categories
                for i, name in enumerate(inputs):
                    # The graph has been modified if any input is reordered
                    if node.input[i] != name:
                        # Note: This is never reset back to False
                        graph_modified = True
                    # Reassign input name at the new index
                    node.input[i] = name
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified
