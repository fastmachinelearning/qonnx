# Copyright (c) 2022 - 2025 Advanced Micro Devices, Inc.
# Copyright (c) 2020 - 2022 Xilinx, Inc.
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

import copy
import inspect
import onnx
import onnx.helper as oh
import onnx.numpy_helper as np_helper
import os
import warnings
from onnx import TensorProto

import qonnx.util.basic as util
import qonnx.util.onnx as onnxutil
from qonnx.core.datatype import DataType
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.general import (
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
    SortCommutativeInputsInitializerLast,
    SortGraph,
)


class ModelWrapper:
    """A wrapper around ONNX ModelProto that exposes some useful utility
    functions for graph manipulation and exploration."""

    def __init__(self, onnx_model_proto, make_deepcopy=False, fix_float64=False, fix_missing_initializer_valueinfo=True):
        """Creates a ModelWrapper instance.
        onnx_model_proto can be either a ModelProto instance, or a string
        with the path to a stored .onnx file on disk, or serialized bytes.

        make_deepcopy: controls whether a deep copy of the ModelProto
        is made internally.
        fix_float64 : DoubleToSingleFloat correction before applying any
        transformations on this model.
        fix_missing_initializer_valueinfo: add ValueInfoProto fields for
        initializers that are missing theirs.
        """
        if isinstance(onnx_model_proto, str):
            assert os.path.isfile(onnx_model_proto), f"File not found: {onnx_model_proto}"
            self._model_proto = onnx.load(onnx_model_proto)
        elif isinstance(onnx_model_proto, bytes):
            self._model_proto = onnx.load_from_string(onnx_model_proto)
        else:
            if make_deepcopy:
                self._model_proto = copy.deepcopy(onnx_model_proto)
            else:
                self._model_proto = onnx_model_proto
        self.temporary_fix_oldstyle_domain()
        if fix_missing_initializer_valueinfo:
            self.check_all_tensor_shapes_specified(fix_missing_init_shape=True)
        self.fix_float64 = fix_float64

    def temporary_fix_oldstyle_domain(self):
        found_oldstyle = False
        for n in self.graph.node:
            if n.domain == "finn":
                n_backend = util.get_by_name(n.attribute, "backend")
                if n_backend is not None:
                    backend_value = n_backend.s.decode("UTF-8")
                    if backend_value == "fpgadataflow":
                        n.domain = "finn.custom_op.fpgadataflow"
                    else:
                        warnings.warn("Can't fix domain for node " + str(n))
                else:
                    n.domain = "qonnx.custom_op.general"
                found_oldstyle = True
            elif n.domain == "finn.custom_op.general":
                n.domain = "qonnx.custom_op.general"
                found_oldstyle = True
        if found_oldstyle:
            warnings.warn(
                """Some old-style domain attributes were automatically converted to new-style,
                i.e. domain=finn to domain=qonnx.custom_op.<general|fpgadataflow|...>"""
            )

    @property
    def graph(self):
        """Returns the graph of the model."""
        return self._model_proto.graph

    @graph.setter
    def graph(self, value):
        """Sets the graph of the model according to value"""
        self._model_proto.graph = value

    @property
    def model(self):
        """Returns the model."""
        return self._model_proto

    @model.setter
    def model(self, value):
        """Sets the model according to value."""
        self._model_proto = value

    def save(self, filename):
        """Saves the wrapper ONNX ModelProto into a file with given name."""
        onnx.save(self._model_proto, filename)

    def analysis(self, analysis_fxn, apply_to_subgraphs=False):
        """Runs given anaylsis_fxn on this model and return resulting dict."""
        if apply_to_subgraphs == True:
            assert "apply_to_subgraphs" in inspect.signature(analysis_fxn), "analysis_fxn must have 'apply_to_subgraphs' argument when apply_to_subgraphs == True"
            return analysis_fxn(self, apply_to_subgraphs)
        else:
            return analysis_fxn(self)

    def transform_subgraphs(self, transformation, make_deepcopy=True, cleanup=True, apply_to_subgraphs=False, use_preorder_traversal=True):
        """Applies given Transformation to all subgraphs of this ModelWrapper instance.

        - make_deepcopy : operates on a new (deep)copy of model.
        - cleanup : execute cleanup transformations before returning
        - apply_to_subgraphs : if True, transformation is applied to all subgraphs of the model
        - use_preorder_traversal : if True, uses preorder traversal for subgraph transformation,
          otherwise postorder traversal is used.
        """
        for node in self.model.graph.node:
                transformed_subgraph_attrs = []
                for idx, attr in enumerate(node.attribute):
                    if attr.type == onnx.AttributeProto.GRAPH:
                        # this is a subgraph, add it to the list
                        subgraph = self.make_subgraph_modelwrapper(attr.g)
                        # apply the transformation to the subgraph
                        subgraph = subgraph.transform(transformation, make_deepcopy, cleanup, apply_to_subgraphs, use_preorder_traversal)
                        # update the new subgraph in the attrubute
                        transformed_subgraph_attrs.append((idx, onnx.helper.make_attribute(attr.name, subgraph.model.graph)))
                # replace the attributes in the node with the transformed subgraph attributes
                for idx, new_attr in transformed_subgraph_attrs:
                    # remove the old attribute
                    node.attribute.pop(idx)
                    # add the new attribute
                    node.attribute.insert(idx, new_attr)

    def transform(self, transformation, make_deepcopy=True, cleanup=True, apply_to_subgraphs=False, use_preorder_traversal=True):
        """Applies given Transformation repeatedly until no more changes can be made
        and returns a transformed ModelWrapper instance.

        - make_deepcopy : operates on a new (deep)copy of model.
        - cleanup : execute cleanup transformations before returning
        - apply_to_subgraphs : if True, transformation is applied to all subgraphs of the model
        """
        transformed_model = self
        if make_deepcopy:
            transformed_model = copy.deepcopy(self)
        if self.fix_float64:
            (transformed_model, model_was_changed) = DoubleToSingleFloat().apply(transformed_model)

        if apply_to_subgraphs and use_preorder_traversal == False:
            transformed_model.transform_subgraphs(transformation, make_deepcopy, cleanup, apply_to_subgraphs, use_preorder_traversal)

        model_was_changed = True
        while model_was_changed:
            (transformed_model, model_was_changed) = transformation.apply(transformed_model)
        if cleanup:
            transformed_model.cleanup()

        if apply_to_subgraphs and use_preorder_traversal:
            transformed_model.transform_subgraphs(transformation, make_deepcopy, cleanup, apply_to_subgraphs, use_preorder_traversal)

        return transformed_model

    def cleanup(self):
        "Run cleanup transformations on the model."
        transformed_model = self
        cleanup_transforms = [
            RemoveUnusedTensors(),
            RemoveStaticGraphInputs(),
            SortGraph(),
            SortCommutativeInputsInitializerLast(),
        ]
        for trn in cleanup_transforms:
            transformed_model = transformed_model.transform(trn, cleanup=False, make_deepcopy=False)
        return transformed_model

    def make_subgraph_modelwrapper(self, subgraph):
        return ModelWrapper(util.qonnx_make_model(subgraph, opset_imports=self._model_proto.opset_import))

    def get_tensor_datatype(self, tensor_name):
        """Returns the QONNX DataType of tensor with given name."""
        graph = self._model_proto.graph
        qnt_annotations = graph.quantization_annotation
        ret = util.get_by_name(qnt_annotations, tensor_name, "tensor_name")
        if ret is not None:
            ret = util.get_by_name(ret.quant_parameter_tensor_names, "finn_datatype", "key")
            if ret is not None:
                return DataType[ret.value]
        onnx_dtype_to_qonnx_dtype = {
            TensorProto.FLOAT: "FLOAT32",
            TensorProto.FLOAT16: "FLOAT16",
            # TODO: dtypes below need testing to ensure they do not break FINN,
            # since it normally assumes float32 containers for these dtypes
            # TensorProto.UINT8 : "UINT8",
            # TensorProto.INT8 : "INT8",
            # TensorProto.UINT16 : "UINT16",
            # TensorProto.INT16 : "INT16",
            # TensorProto.UINT32 : "UINT32",
            # TensorProto.INT32 : "INT32",
            # TensorProto.UINT64 : "UINT64",
            # TensorProto.INT64 : "INT64",
        }
        tensor_vi = self.get_tensor_valueinfo(tensor_name)
        if tensor_vi is None:
            # some initialized tensors don't get ValueInfo even after shape inference
            _, onnx_dtype = self.get_initializer(tensor_name, return_dtype=True)
        else:
            onnx_dtype = tensor_vi.type.tensor_type.elem_type
        if onnx_dtype in onnx_dtype_to_qonnx_dtype.keys():
            return DataType[onnx_dtype_to_qonnx_dtype[onnx_dtype]]
        else:
            return DataType["FLOAT32"]

    def set_tensor_datatype(self, tensor_name, datatype):
        """Sets the QONNX DataType of tensor with given name."""
        graph = self._model_proto.graph
        qnt_annotations = graph.quantization_annotation
        ret = util.get_by_name(qnt_annotations, tensor_name, "tensor_name")
        if ret is not None:
            ret_dt = util.get_by_name(ret.quant_parameter_tensor_names, "finn_datatype", "key")
            if ret_dt is not None:
                if datatype is None:
                    ret_dt.Clear()
                else:
                    ret_dt.value = datatype.name
            elif datatype is not None:
                dt = onnx.StringStringEntryProto()
                dt.key = "finn_datatype"
                dt.value = datatype.name
                ret.quant_parameter_tensor_names.append(dt)
        elif datatype is not None:
            qa = onnx.TensorAnnotation()
            dt = onnx.StringStringEntryProto()
            dt.key = "finn_datatype"
            dt.value = datatype.name
            qa.tensor_name = tensor_name
            qa.quant_parameter_tensor_names.append(dt)
            qnt_annotations.append(qa)

    def get_tensor_valueinfo(self, tensor_name):
        """Returns ValueInfoProto of tensor with given name, if it has one."""
        graph = self._model_proto.graph
        vi_names = [(x.name, x) for x in graph.input]
        vi_names += [(x.name, x) for x in graph.output]
        vi_names += [(x.name, x) for x in graph.value_info]
        try:
            vi_t_names = [x[0] for x in vi_names]
            assert vi_t_names.count(tensor_name) <= 1, "Multiple ValueInfoProto found for " + tensor_name
            vi_ind = vi_t_names.index(tensor_name)
            vi = vi_names[vi_ind][1]
            return vi
        except ValueError:
            return None

    def get_tensor_shape(self, tensor_name, fix_missing_init_shape=False):
        """Returns the shape of tensor with given name, if it has ValueInfoProto.
        If fix_missing_init_shape is specified, it will add a ValueInfoProto for initializers
        that are missing theirs."""
        graph = self._model_proto.graph
        vi_names = [(x.name, x) for x in graph.input]
        vi_names += [(x.name, x) for x in graph.output]
        vi_names += [(x.name, x) for x in graph.value_info]
        try:
            vi_t_names = [x[0] for x in vi_names]
            assert vi_t_names.count(tensor_name) <= 1, "Multiple ValueInfoProto found for " + tensor_name
            vi_ind = vi_t_names.index(tensor_name)
            vi = vi_names[vi_ind][1]
            dims = [x.dim_value for x in vi.type.tensor_type.shape.dim]
            return dims
        except ValueError:
            # no ValueInfo found for tensor, check initializer
            # (see https://github.com/onnx/onnx/issues/2874)
            tensor_init, tensor_init_dtype = self.get_initializer(tensor_name, return_dtype=True)
            if tensor_init is None:
                # no shape defined for this tensor
                return None
            else:
                if fix_missing_init_shape:
                    self.set_tensor_shape(tensor_name, tensor_init.shape, dtype=tensor_init_dtype)
                # use list return type to keep it consistent with ValueInfo case
                return list(tensor_init.shape)

    def set_tensor_shape(self, tensor_name, tensor_shape, dtype=None):
        """Assigns shape in ValueInfoProto for tensor with given name. If override_dtype
        is None, it will try to preserve the existing datatype, otherwise defaults to
        single-precision float."""
        # call get_tensor_valueinfo to raise a warning for multiple ValueInfoProto cases
        old_vi = self.get_tensor_valueinfo(tensor_name)
        # resolve dtype
        if dtype is None:
            # try to preserve old dtype
            if old_vi is None:
                dtype = TensorProto.FLOAT
            else:
                dtype = old_vi.type.tensor_type.elem_type
        new_vi = oh.make_tensor_value_info(tensor_name, dtype, tensor_shape)
        # find what container this tensor's ValueInfo lives in
        # if not found anywhere, we assume it's a new value_info
        target_container = self.graph.value_info
        if util.get_by_name(self.graph.input, tensor_name) is not None:
            target_container = self.graph.input
            # create list from inputs to find index
            inputs = [x.name for x in self.graph.input]
            # save index of input to preserve order
            ind = inputs.index(tensor_name)
        if util.get_by_name(self.graph.output, tensor_name) is not None:
            target_container = self.graph.output
            # create list from inputs to find index
            outputs = [x.name for x in self.graph.output]
            # save index of input to preserve order
            ind = outputs.index(tensor_name)
        # remove from target container and add new
        util.remove_by_name(target_container, tensor_name)
        if target_container == self.graph.value_info:
            target_container.append(new_vi)
        else:
            target_container.insert(ind, new_vi)

    def set_initializer(self, tensor_name, tensor_value):
        """Sets the initializer value for tensor with given name."""
        graph = self._model_proto.graph
        # convert tensor_value (numpy array) into TensorProto w/ correct name
        tensor_init_proto = np_helper.from_array(tensor_value)
        tensor_init_proto.name = tensor_name
        # first, remove if an initializer already exists
        init_names = [x.name for x in graph.initializer]
        try:
            init_ind = init_names.index(tensor_name)
            init_old = graph.initializer[init_ind]
            graph.initializer.remove(init_old)
        except ValueError:
            pass
        # create and insert new initializer
        graph.initializer.append(tensor_init_proto)
        # set shape
        dtype = tensor_init_proto.data_type
        self.set_tensor_shape(tensor_name, list(tensor_value.shape), dtype)

    def rename_tensor(self, old_name, new_name):
        """Renames a tensor from old_name to new_name."""
        graph = self.graph
        # sweep over inputs
        if util.get_by_name(graph.input, old_name) is not None:
            util.get_by_name(graph.input, old_name).name = new_name
        # sweep over outputs
        if util.get_by_name(graph.output, old_name) is not None:
            util.get_by_name(graph.output, old_name).name = new_name
        # sweep over value_info
        if util.get_by_name(graph.value_info, old_name) is not None:
            util.get_by_name(graph.value_info, old_name).name = new_name
        # sweep over initializers
        if util.get_by_name(graph.initializer, old_name) is not None:
            util.get_by_name(graph.initializer, old_name).name = new_name
        # sweep over quantization annotations
        if util.get_by_name(graph.quantization_annotation, old_name, "tensor_name") is not None:
            util.get_by_name(graph.quantization_annotation, old_name, "tensor_name").tensor_name = new_name
        # sweep over node i/o
        for n in graph.node:
            if old_name in n.input:
                n.input[list(n.input).index(old_name)] = new_name
            if old_name in n.output:
                n.output[list(n.output).index(old_name)] = new_name

    def get_initializer(self, tensor_name, return_dtype=False):
        """Gets the initializer value for tensor with given name, if any.
        ret_dtype can be set to True to retrieve the TensorProto.DataType of the
        initializer by returning it as a second element of a tuple."""
        graph = self._model_proto.graph
        init_names = [x.name for x in graph.initializer]
        try:
            init_ind = init_names.index(tensor_name)
            ret = np_helper.to_array(graph.initializer[init_ind])
            ret_dtype = graph.initializer[init_ind].data_type
            if return_dtype:
                return (ret, ret_dtype)
            else:
                return ret
        except ValueError:
            if return_dtype:
                return (None, None)
            else:
                return None

    def del_initializer(self, initializer_name):
        """Deletes an initializer from the model."""
        graph = self._model_proto.graph
        init = util.get_by_name(graph.initializer, initializer_name)
        if not (init is None):
            graph.initializer.remove(init)

    def find_producer(self, tensor_name):
        """Finds and returns the node that produces the tensor with given name."""
        for x in self._model_proto.graph.node:
            if tensor_name in x.output:
                return x
        return None

    def find_upstream(self, tensor_name, finder_fxn, keep_if_not_found=False):
        """Follow the producer chain upstream, calling finder_fxn on each upstream
        node until it returns True or there are no nodes left. Returns the list
        of nodes visited, or None if finder_fxn did not return True. If
        keep_if_not_found is specified, returns the list of nodes visited, even
        if finder_fxn never returned True, i.e., if the search terminated at an
        input or initializer."""
        visit_list = []
        current_tensor = tensor_name
        while True:
            current_producer = self.find_producer(current_tensor)
            if current_producer is None:
                return visit_list if keep_if_not_found else []
            else:
                found = finder_fxn(current_producer)
                visit_list.append(current_producer)
                if found:
                    return visit_list
                elif len(current_producer.input) > 0:
                    current_tensor = current_producer.input[0]
                else:
                    return visit_list if keep_if_not_found else None

    def find_consumer(self, tensor_name):
        """Finds and returns the node that consumes the tensor with given name.
        If there are multiple consumers, only the first one is returned.
        If there are no consumers, returns None."""
        ret = self.find_consumers(tensor_name)
        if (ret is None) or (ret == []):
            return None
        elif len(ret) == 1:
            return ret[0]
        else:
            warnings.warn("find_consumer: found multiple consumers, returning first one")
            return ret[0]

    def find_consumers(self, tensor_name):
        """Finds and returns a list of the nodes that consume tensor with
        given name."""
        consumers = []
        for n in self._model_proto.graph.node:
            for inp_tensor in n.input:
                if inp_tensor == tensor_name:
                    consumers.append(n)
        return consumers

    def find_direct_successors(self, node):
        """Finds and returns a list of the nodes that are successors of
        given node."""
        successors = []
        for outp_tensor in node.output:
            tensor_consumer_list = self.find_consumers(outp_tensor)
            if tensor_consumer_list is not None:
                for consumer in tensor_consumer_list:
                    successors.append(consumer)
        if successors != []:
            return successors
        else:
            return None

    def find_direct_predecessors(self, node):
        """Finds and returns a list of the nodes that are predecessors of
        given node."""
        predecessors = []
        for inp_tensor in node.input:
            producer = self.find_producer(inp_tensor)
            if producer is not None:
                predecessors.append(producer)
        if predecessors != []:
            return predecessors
        else:
            return None

    def is_fork_node(self, node):
        """Checks if the given node is a fork, that is, the node has multiple
        direct successors"""
        direct_successors = self.find_direct_successors(node)
        # if the node output is also wired to a top-level output, it is still
        # a fork with only 1 direct successor
        if node.output[0] in [x.name for x in self.graph.output]:
            is_fork = False if direct_successors is None else (len(direct_successors) > 0)
        else:
            is_fork = False if direct_successors is None else (len(direct_successors) > 1)
        return is_fork

    def is_join_node(self, node):
        """Checks if the given node is a join, that is, the node has multiple
        direct predecessors"""
        direct_predecessors = self.find_direct_predecessors(node)
        # if the node input is also wired to a top-level input, it is still
        # a fork with only 1 direct predecessor
        if node.input[0] in [x.name for x in self.graph.input]:
            is_join = False if direct_predecessors is None else (len(direct_predecessors) > 0)
        else:
            is_join = False if direct_predecessors is None else (len(direct_predecessors) > 1)
        return is_join

    def get_all_tensor_names(self):
        """Returns a list of all (input, output and value_info) tensor names
        in the graph."""
        graph = self.graph
        names = [x.name for x in graph.value_info]
        names += [x.name for x in graph.input]
        names += [x.name for x in graph.output]
        return names

    def make_new_valueinfo_name(self):
        """Returns a name that can be used for a new value_info."""
        names = self.get_all_tensor_names()
        candidate = util.random_string()
        while candidate in names:
            candidate = util.random_string()
        return candidate

    def make_empty_exec_context(self):
        """Creates an empty execution context for this model.

        The execution context is a dictionary of all tensors used for the
        inference computation. Any initializer values will be taken into
        account, all other tensors will be zero."""
        execution_context = dict()
        graph = self._model_proto.graph
        # make empty tensors for all the graph inputs and outputs
        for vi in graph.input:
            new_tensor = onnxutil.valueinfo_to_tensor(vi)
            execution_context[vi.name] = new_tensor
        for vi in graph.output:
            new_tensor = onnxutil.valueinfo_to_tensor(vi)
            execution_context[vi.name] = new_tensor
        # make empty tensors for all intermediate buffers
        for vi in graph.value_info:
            new_tensor = onnxutil.valueinfo_to_tensor(vi)
            execution_context[vi.name] = new_tensor
        # fill in the constants provided by the initializers (TensorProto to npy)
        for t in graph.initializer:
            execution_context[t.name] = np_helper.to_array(t)
        # for nodes that use empty string as input (=default value), create a
        # dummy entry in the context
        execution_context[""] = None
        return execution_context

    def check_all_tensor_shapes_specified(self, fix_missing_init_shape=False):
        """Checks whether all tensors have a specified shape (ValueInfo).
        The ONNX standard allows for intermediate activations to have no
        associated ValueInfo, but QONNX expects this.
        If fix_missing_init_shape is specified, it will add a ValueInfoProto
        for initializers that are missing theirs."""
        graph = self._model_proto.graph
        ret = True
        # note the call to get_tensor_shape needs to be first to avoid early stopping here
        # due to missing tensor shapes
        # see https://github.com/fastmachinelearning/qonnx/issues/33
        for n in graph.node:
            for i in n.input:
                # skip tensor names with empty string (indicates defaults)
                if i != "":
                    ret = (self.get_tensor_shape(i, fix_missing_init_shape=fix_missing_init_shape) is not None) and ret
            for o in n.output:
                ret = (self.get_tensor_shape(o, fix_missing_init_shape=fix_missing_init_shape) is not None) and ret
        return ret

    def get_tensor_fanout(self, tensor_name):
        """Returns the number of nodes for which the tensor with given name is
        as input."""
        graph = self.graph
        fanout = 0
        for n in graph.node:
            if tensor_name in n.input:
                fanout += 1
        return fanout

    def get_metadata_prop(self, key):
        """Returns the value associated with metadata_prop with given key,
        or None otherwise."""
        metadata_prop = util.get_by_name(self.model.graph.metadata_props, key, "key")
        if metadata_prop is None:
            return None
        else:
            return metadata_prop.value

    def set_metadata_prop(self, key, value):
        """Sets metadata property with given key to the given value."""
        metadata_prop = util.get_by_name(self.model.graph.metadata_props, key, "key")
        if metadata_prop is None:
            metadata_prop = onnx.StringStringEntryProto()
            metadata_prop.key = key
            metadata_prop.value = value
            self.model.graph.metadata_props.append(metadata_prop)
        else:
            metadata_prop.value = value

    def get_nodes_by_op_type(self, op_type):
        """Returns a list of nodes with specified op_type."""
        return list(filter(lambda x: x.op_type == op_type, self.graph.node))

    def get_finn_nodes(self):
        """Returns a list of nodes where domain == 'qonnx.*'."""
        return list(filter(lambda x: util.is_finn_op(x.domain), self.graph.node))

    def get_non_finn_nodes(self):
        """Returns a list of nodes where domain != 'qonnx.*'."""
        return list(filter(lambda x: not util.is_finn_op(x.domain), self.graph.node))

    def get_node_index(self, node):
        """Returns current index of given node, or None if not found."""
        n_ind = 0
        try:
            for n in self.graph.node:
                if n == node:
                    return n_ind
                n_ind += 1
        except ValueError:
            return None
        return None

    def get_node_from_name(self, node_name):
        """Returns the node with the specified name, or None if not found."""
        try:
            for node in self.graph.node:
                if node.name == node_name:
                    return node
        except ValueError:
            return None
        return None

    def get_tensor_layout(self, tensor_name):
        """Returns the data layout annotation of tensor with given name.
        The data layout is expressed as a list of strings with as many
        elements as the number of dimensions in the tensor shape. Each
        string annotates what is contained in that dimension. If there is no
        data layout annotation, None will be returned.
        Examples of data layout annotations:
        ["N", "C"] is tensor[batch][channel]
        ["N", "C", "H", "W"] is tensor[batch][channel][height][width]
        ["N", "H", "W", "C"] is tensor[batch][height][width][channel]
        """
        graph = self._model_proto.graph
        qnt_annotations = graph.quantization_annotation
        ret = util.get_by_name(qnt_annotations, tensor_name, "tensor_name")
        if ret is not None:
            ret = util.get_by_name(ret.quant_parameter_tensor_names, "tensor_layout", "key")
            if ret is not None:
                return eval(ret.value)
        return None

    def set_tensor_layout(self, tensor_name, data_layout):
        """Sets the data layout annotation of tensor with given name. See
        get_tensor_layout for examples."""
        assert type(data_layout) == list, "data_layout must be a list"
        graph = self._model_proto.graph
        qnt_annotations = graph.quantization_annotation
        ret = util.get_by_name(qnt_annotations, tensor_name, "tensor_name")
        if ret is not None:
            ret_tl = util.get_by_name(ret.quant_parameter_tensor_names, "tensor_layout", "key")
            if ret_tl is not None:
                ret_tl.value = str(data_layout)
            else:
                tl = onnx.StringStringEntryProto()
                tl.key = "tensor_layout"
                tl.value = str(data_layout)
                ret.quant_parameter_tensor_names.append(tl)
        else:
            qa = onnx.TensorAnnotation()
            dt = onnx.StringStringEntryProto()
            dt.key = "tensor_layout"
            dt.value = str(data_layout)
            qa.tensor_name = tensor_name
            qa.quant_parameter_tensor_names.append(dt)
            qnt_annotations.append(qa)

    def get_tensor_sparsity(self, tensor_name):
        """Returns the sparsity of a given tensor as dictionary."""
        graph = self._model_proto.graph
        qnt_annotations = graph.quantization_annotation
        ret = util.get_by_name(qnt_annotations, tensor_name, "tensor_name")
        if ret is not None:
            ret = util.get_by_name(ret.quant_parameter_tensor_names, "tensor_sparsity", "key")
            if ret is not None:
                return eval(ret.value)
        return None

    def set_tensor_sparsity(self, tensor_name, sparsity_dict):
        """Sets the sparsity annotation of a tensor with given name."""
        graph = self._model_proto.graph
        qnt_annotations = graph.quantization_annotation
        ret = util.get_by_name(qnt_annotations, tensor_name, "tensor_name")
        if ret is not None:
            ret_ts = util.get_by_name(ret.quant_parameter_tensor_names, "tensor_sparsity", "key")
            if ret_ts is not None:
                ret_ts.value = str(sparsity_dict)
            else:
                ts = onnx.StringStringEntryProto()
                ts.key = "tensor_sparsity"
                ts.value = str(sparsity_dict)
                ret.quant_parameter_tensor_names.append(ts)
        else:
            qa = onnx.TensorAnnotation()
            dt = onnx.StringStringEntryProto()
            dt.key = "tensor_sparsity"
            dt.value = str(sparsity_dict)
            qa.tensor_name = tensor_name
            qa.quant_parameter_tensor_names.append(dt)
            qnt_annotations.append(qa)
