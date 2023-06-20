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

eltwise_ops = ["Add", "Mul", "Sub", "Div", "BatchNormalization", "MultiThreshold", "Quant"]


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
    elif node.op_type == "MatMul":
        # input and output are essentially decoupled from
        # each other by means of the weight. the weight mask
        # is formulated specially via prefixed strings:
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
                if io_t is None:
                    # dynamic input/output, no initializer
                    # compute new shape only
                    # TODO proper axis? assumes NCHW
                    axis = 1
                    new_shp = remove_masked_tensor_channels(io_shp, mask, axis=axis)
                    model.set_tensor_shape(ioname, new_shp)
                    # clear sparsity annotation
                    model.set_tensor_sparsity(ioname, {})
                else:
                    if node.op_type in ["MatMul"]:
                        i_mask = [int(x.replace("i", "")) for x in mask if x.startswith("i")]
                        o_mask = [int(x.replace("o", "")) for x in mask if x.startswith("o")]
                        new_t = remove_masked_tensor_channels(io_t, i_mask, axis=0)
                        new_t = remove_masked_tensor_channels(new_t, o_mask, axis=1)
                        model.set_initializer(ioname, new_t)
                        # clear sparsity annotation
                        model.set_tensor_sparsity(ioname, {})
                    else:
                        # TODO proper axis? assumes NCHW
                        axis = 1 if io_t.ndim >= 2 else 0
                        new_t = remove_masked_tensor_channels(io_t, mask, axis=axis)
                        model.set_initializer(ioname, new_t)
                        # clear sparsity annotation
                        model.set_tensor_sparsity(ioname, {})
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
