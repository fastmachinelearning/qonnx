# Copyright (c) 2024 Advanced Micro Devices, Inc.
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
from functools import partial
from onnx import TensorProto, helper
from warnings import warn

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.util.range_analysis import unbroadcast_tensor


def default_streamline_tensor_filter(model: ModelWrapper, tname: str):
    not_initializer = model.get_initializer(tname) is None
    not_toplevel = not (tname in [x.name for x in model.graph.output])
    consumer_is_quant = all([x.op_type == "Quant" for x in model.find_consumers(tname)])
    return not_initializer and consumer_is_quant and not_toplevel


class Streamline(Transformation):
    """
    Given a scaled-integer range analysis dictionary, call the ExtractAggregateScaleBias
    transformation for every tensor feeding a dynamic quantizer (Quant node).
    """

    def __init__(self, scaledint_range_dict, tensor_filter=default_streamline_tensor_filter, include_toplevel_outs=True):
        super().__init__()
        self.scaledint_range_dict = scaledint_range_dict
        self.tensor_filter = tensor_filter
        self.include_toplevel_outs = include_toplevel_outs

    def apply(self, model: ModelWrapper):
        tensor_filter = partial(self.tensor_filter, model)
        tensor_list = list(filter(tensor_filter, model.get_all_tensor_names()))
        if self.include_toplevel_outs:
            tensor_list += [x.name for x in model.graph.output]
        for tensor_name in tensor_list:
            model = model.transform(ExtractAggregateScaleBias(self.scaledint_range_dict, tensor_name))
        return model, False


class ExtractAggregateScaleBias(Transformation):
    """
    Given a scaled-integer range analysis dictionary and a tensor name, extract the
    aggregate scale and bias for that tensor as standalone Mul and Add nodes, re-setting
    the original tensors that contributed to the aggregate scale and bias to 1 and 0,
    respectively.
    """

    def __init__(self, scaledint_range_dict, target_tensor_name, remove_identity=True, unbroadcast=True):
        super().__init__()
        self.target_tensor_name = target_tensor_name
        self.target_tensor_ri = scaledint_range_dict[self.target_tensor_name]
        self.remove_identity = remove_identity
        self.unbroadcast = unbroadcast

    def apply(self, model: ModelWrapper):
        graph = model.graph
        # transform:
        # head node -> (target tensor) -> tail node
        # into:
        # head node -> (new tensor 0) -> Mul -> (new tensor 1) -> Add -> (target tensor) -> tail node
        # TODO insert Identity nodes around target tensor to always guarantee
        # head node below (needed for for top-level tensors)
        head = model.find_producer(self.target_tensor_name)
        assert head is not None
        tshape = self.target_tensor_ri.shape
        scale = self.target_tensor_ri.scale
        bias = self.target_tensor_ri.bias
        if scale is None or bias is None:
            warn(f"{self.target_tensor_name} has no scaled-int information for ExtractAggregateScaleBias, skipping ")
            return model, False
        if self.unbroadcast:
            scale = unbroadcast_tensor(scale)
            bias = unbroadcast_tensor(bias)
        # create new tensors
        new_tensor0_tname = self.target_tensor_name + "_prescale"
        new_tensor1_tname = self.target_tensor_name + "_prebias"
        new_tensor0_vi = helper.make_tensor_value_info(
            new_tensor0_tname,
            TensorProto.FLOAT,
            tshape,
        )
        graph.value_info.append(new_tensor0_vi)
        new_tensor1_vi = helper.make_tensor_value_info(
            new_tensor1_tname,
            TensorProto.FLOAT,
            tshape,
        )
        graph.value_info.append(new_tensor1_vi)
        scale_tname = self.target_tensor_name + "_aggr_scale"
        model.set_initializer(scale_tname, scale)
        bias_tname = self.target_tensor_name + "_aggr_bias"
        model.set_initializer(bias_tname, bias)
        # create new Mul and Add nodes to apply aggregate scale and bias
        mul_node = helper.make_node(
            "Mul",
            [new_tensor0_tname, scale_tname],
            [new_tensor1_tname],
        )
        graph.node.append(mul_node)
        add_node = helper.make_node(
            "Add",
            [new_tensor1_tname, bias_tname],
            [self.target_tensor_name],
        )
        graph.node.append(add_node)
        # remove old scale/bias tensors from history by setting them to 1 or 0
        for old_scale_tname in self.target_tensor_ri.history_scale:
            if model.get_initializer(old_scale_tname) is None:
                # TODO clearer warning for input scale factors
                warn(f"{old_scale_tname} does not have an initializer!")
            else:
                model.set_initializer(old_scale_tname, np.asarray(1.0, dtype=np.float32))
        for old_bias_tname in self.target_tensor_ri.history_bias:
            if model.get_initializer(old_bias_tname) is None:
                # TODO clearer warning for input biases
                warn(f"{old_bias_tname} does not have an initializer!.")
            else:
                model.set_initializer(old_bias_tname, np.asarray(0.0, dtype=np.float32))
        # rewire the head node
        head_out_target_tensor_ind = list(head.output).index(self.target_tensor_name)
        head.output[head_out_target_tensor_ind] = new_tensor0_tname
        # sort graph topologically (we added the Mul/Add nodes at the end)
        model = model.transform(SortGraph())
        if self.remove_identity:
            model = model.transform(RemoveIdentityOps())
        return model, False
