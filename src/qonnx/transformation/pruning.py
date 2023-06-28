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
from typing import Dict, Tuple

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name

eltwise_ops = ["Add", "Mul", "Sub", "Div", "BatchNormalization", "MultiThreshold", "Quant", "Relu"]


def ensure_masktype_is_set(mask):
    if type(mask) is set:
        # all good, return as is
        return mask
    if mask is None:
        # use empty set instead of no sparsity mask (None)
        return set()
    else:
        raise Exception("Cannot turn %s into set" % str(mask))


def remove_masked_tensor_channels(tensor_or_shape, mask, axis):
    shape_only = False
    if type(mask) is not list:
        mask_list = list(mask)
    else:
        mask_list = mask
    if type(tensor_or_shape) in [list, tuple]:
        shape_only = True
        tensor_or_shape = np.random.rand(*tensor_or_shape)
    assert type(tensor_or_shape) is np.ndarray
    if tensor_or_shape.ndim == 0 or np.prod(tensor_or_shape.shape) == 1:
        # no pruning for scalar properties
        ret = tensor_or_shape
    else:
        ret = np.delete(tensor_or_shape, mask_list, axis=axis)
    if shape_only:
        return ret.shape
    else:
        return ret


def update_node_mask(node, masks_in, masks_out):
    masks_in = [ensure_masktype_is_set(x) for x in masks_in]
    masks_out = [ensure_masktype_is_set(x) for x in masks_out]

    if node.op_type in eltwise_ops:
        # any i/o can mask any/all other i/o
        # so just take union
        ret = set().union(*masks_in).union(*masks_out)
        masks_in = [ret for x in masks_in]
        masks_out = [ret for x in masks_out]
    elif node.op_type in ["MatMul", "Conv"]:
        # input and output are essentially decoupled from
        # each other by means of the weight (except dwise convs)
        if node.op_type == "Conv":
            groups = get_by_name(node.attribute, "group").i
            # TODO smarter check, other kinds of grouped convs out there..
            is_depthwise = groups > 1
        else:
            is_depthwise = False

        # the weight mask is formulated specially via prefixed strings:
        # iX for input channel X
        # oX for output channel X
        w_mask = masks_in[1]
        # convert back to two distinct int sets to be able to use union etc set ops
        w_mask_in = {int(x.replace("i", "")) for x in w_mask if x.startswith("i")}
        w_mask_out = {int(x.replace("o", "")) for x in w_mask if x.startswith("o")}
        # take union with i/o masks to update
        i_mask = masks_in[0]
        o_mask = masks_out[0]
        mask_in = w_mask_in.union(i_mask)
        mask_out = w_mask_out.union(o_mask)
        if is_depthwise:
            # depthwise convs couple i<->o channels directly
            mask_in = mask_in.union(mask_out)
            mask_out = mask_in
            # dw convs to only use output side for weights by convention
            w_mask = {"o%d" % x for x in mask_out}
        else:
            w_mask = {"i%d" % x for x in mask_in}.union({"o%d" % x for x in mask_out})
        masks_in = [mask_in, w_mask]
        masks_out = [mask_out]
    else:
        warnings.warn("Can't propagate sparsity mask through op_type %s" % node.op_type)
    return (masks_in, masks_out)


class ApplyMasks(Transformation):
    def __init__(self, prune_spec: Dict) -> None:
        super().__init__()
        self.prune_spec = prune_spec

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        for key, val in self.prune_spec.items():
            # sanity check: if tensor is a weight tensor for
            # Conv or MatMul nodes it needs to follow the convention
            # for indicating input or output channels
            t_has_init = model.get_initializer(key) is not None
            t_consumer = model.find_consumer(key)
            t_fc_cnv = t_consumer is not None and t_consumer.op_type in ["Conv", "MatMul"]
            t_fc_cnv_w = t_fc_cnv and t_consumer.input[1] == key
            if t_fc_cnv_w and t_has_init:
                val_check = list(val)[0]
                assert type(val_check) is str, "Weight masks must be strings"
                assert val_check.startswith("i") or val_check.startswith("o"), "Weight masks must be formatted iX or oX"
            model.set_tensor_sparsity(key, val)
        return (model, False)


class PropagateMasks(Transformation):
    def __init__(self) -> None:
        super().__init__()

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        need_rerun = False
        # traverse graph from inputs to outputs to propagate
        # sparsity masks via per-layer handlers
        for node in model.graph.node:
            node_masks_in = [model.get_tensor_sparsity(x) for x in node.input]
            node_masks_out = [model.get_tensor_sparsity(x) for x in node.output]
            (new_in, new_out) = update_node_mask(node, node_masks_in, node_masks_out)
            in_changed = new_in != node_masks_in
            out_changed = new_out != node_masks_out
            need_rerun |= in_changed
            need_rerun |= out_changed
            for inp_name, inp_annot in zip(node.input, new_in):
                model.set_tensor_sparsity(inp_name, inp_annot)
            for out_name, out_annot in zip(node.output, new_out):
                model.set_tensor_sparsity(out_name, out_annot)
        return (model, need_rerun)


class RemoveMaskedChannels(Transformation):
    def __init__(self) -> None:
        super().__init__()

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        need_rerun = False
        for node in model.graph.node:
            for ioname in [*node.input, *node.output]:
                io_t = model.get_initializer(ioname)
                io_shp = model.get_tensor_shape(ioname)
                mask = model.get_tensor_sparsity(ioname)
                if mask is None or mask == {}:
                    continue
                # print("[RemoveMaskedChannels] tensor %s mask %s: old shape %s" % (ioname, str(mask), str(io_shp)))
                if io_t is None:
                    # dynamic input/output, no initializer
                    # compute new shape only
                    # TODO proper axis? assumes NCHW
                    axis = 1
                    new_shp = remove_masked_tensor_channels(io_shp, mask, axis=axis)
                    model.set_tensor_shape(ioname, new_shp)
                else:
                    if node.op_type in ["MatMul"]:
                        i_mask = [int(x.replace("i", "")) for x in mask if x.startswith("i")]
                        o_mask = [int(x.replace("o", "")) for x in mask if x.startswith("o")]
                        # for MatMul weights, input axis is 0, output is 1
                        new_t = remove_masked_tensor_channels(io_t, i_mask, axis=0)
                        new_t = remove_masked_tensor_channels(new_t, o_mask, axis=1)
                        model.set_initializer(ioname, new_t)
                    elif node.op_type in ["Conv"]:
                        i_mask = [int(x.replace("i", "")) for x in mask if x.startswith("i")]
                        o_mask = [int(x.replace("o", "")) for x in mask if x.startswith("o")]
                        ifm_w = io_shp[1]
                        groups = get_by_name(node.attribute, "group").i
                        assert groups == 1 or ifm_w == 1, "Unknown grouped conv setting"
                        depthwise = groups > 1
                        # for dense Conv weights, input (ifm) axis is 1, output (ofm) is 0
                        ifm_axis = 1
                        ofm_axis = 0
                        if depthwise:
                            # depthwise convs only use the o_mask by convention
                            new_t = remove_masked_tensor_channels(io_t, o_mask, axis=ofm_axis)
                            # need to update the group attribute to match new n chans
                            get_by_name(node.attribute, "group").i = new_t.shape[0]
                        else:
                            new_t = remove_masked_tensor_channels(io_t, i_mask, axis=ifm_axis)
                            new_t = remove_masked_tensor_channels(new_t, o_mask, axis=ofm_axis)
                        model.set_initializer(ioname, new_t)
                    else:
                        # TODO proper axis? assumes NCHW
                        axis = 1 if io_t.ndim >= 2 else 0
                        new_t = remove_masked_tensor_channels(io_t, mask, axis=axis)
                        model.set_initializer(ioname, new_t)
                # clear sparsity annotation since it's already handled
                # leftover annotations here can lead to erronous removal later
                model.set_tensor_sparsity(ioname, {})
                new_shp = model.get_tensor_shape(ioname)
                # print("[RemoveMaskedChannels] tensor %s : new shape %s" % (ioname, str(new_shp)))
                need_rerun = True
        return (model, need_rerun)


class PruneChannels(Transformation):
    def __init__(self, prune_spec: Dict) -> None:
        super().__init__()
        self.prune_spec = prune_spec

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        model = model.transform(ApplyMasks(self.prune_spec))
        model = model.transform(PropagateMasks())
        model = model.transform(RemoveMaskedChannels())
        return (model, False)
