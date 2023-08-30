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
from functools import reduce
from typing import Dict, Tuple

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name

# these ops propagate sparsity masks both forward and backward
# (e.g. a sparse channel on the input creates a sparse channel
# on the output, and vice versa)
# note on Quant and MultiThreshold: only bias-free (no zeropoint)
# versions of these ops are bidirectional
eltwise_ops_bidirectional = [
    "Mul",
    "Div",
    "MultiThreshold",
    "Quant",
    "QuantizeLinear",
    "DequantizeLinear",
    "Clip",
    "Relu",
    "MaxPool",
]

# these ops propagate sparsity masks only backward, not forward
# (e.g. a sparse channel on the output creates a sparse channel
# on the input, but not vice versa)
eltwise_ops_bwdonly = ["Add", "Sub", "BatchNormalization"]

# other ops (MatMul, Conv) have more specialized behavior
# and will be handled by update_node_mask

# mapping of weight tensor axes to input/output channels
optype_to_w_axis = {
    "MatMul": {"in": 0, "out": 1},
    "Conv": {"in": 1, "out": 0},
}


def ensure_masktype_is_dict(mask):
    if isinstance(mask, dict):
        # all good, return as is
        return mask
    if mask is None:
        # use empty dict instead of no sparsity mask (None)
        return dict()
    else:
        raise Exception("Cannot turn %s into dict" % str(mask))


def merge_dicts_of_sets(dict1, dict2):
    ret = deepcopy(dict1)
    for key, val in dict2.items():
        if key in ret.keys():
            ret[key].update(val)
        else:
            ret[key] = val
    return ret


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


def update_node_mask(node, masks_in, masks_out, lossy=True):
    masks_in = [ensure_masktype_is_dict(x) for x in masks_in]
    masks_out = [ensure_masktype_is_dict(x) for x in masks_out]

    if lossy:
        # when in lossy mode, allow propagation sparsity masks
        # in both directions (e.g. Add nodes will also get pruned)
        update_bidirectional = eltwise_ops_bidirectional + eltwise_ops_bwdonly
        update_bwdonly = []

    if node.op_type in update_bidirectional:
        # any i/o can mask any/all other i/o
        # so just take union
        all_masks = [*masks_in] + [*masks_out]
        ret = reduce(merge_dicts_of_sets, all_masks)
        # duplicate the result for each node input and output
        masks_in = [ret for x in masks_in]
        masks_out = [ret for x in masks_out]
    elif node.op_type in update_bwdonly:
        # output can mask input but not other way around
        all_masks = [*masks_in] + [*masks_out]
        ret = reduce(merge_dicts_of_sets, all_masks)
        masks_in = [ret for x in masks_in]
    elif node.op_type in ["MatMul", "Conv"]:
        # input and output are essentially decoupled from
        # each other by means of the weight (except dwise convs)
        if node.op_type == "Conv":
            groups = get_by_name(node.attribute, "group")
            if groups is not None:
                groups = groups.i
            else:
                groups = 1
            # TODO smarter check, other kinds of grouped convs out there..
            is_depthwise = groups > 1
        else:
            is_depthwise = False

        # convert back to two distinct int sets to be able to use union etc set ops
        w_mask = masks_in[1]
        w_axis_in = optype_to_w_axis[node.op_type]["in"]
        w_axis_out = optype_to_w_axis[node.op_type]["out"]
        w_mask_in = w_mask.get(w_axis_in, set())
        w_mask_out = w_mask.get(w_axis_out, set())
        # take union with i/o masks to update
        conv_io_chan_axis = 1
        i_mask = masks_in[0].get(conv_io_chan_axis, set())
        o_mask = masks_out[0].get(conv_io_chan_axis, set())
        mask_in = w_mask_in.union(i_mask)
        mask_out = w_mask_out.union(o_mask)
        if is_depthwise:
            # depthwise convs couple i<->o channels directly
            mask_in = mask_in.union(mask_out)
            mask_out = mask_in
            # dw convs to only use output side for weights by convention
            w_mask[w_axis_out] = mask_out
        else:
            w_mask[w_axis_out] = mask_out
            w_mask[w_axis_in] = mask_in
        masks_in = [{conv_io_chan_axis: mask_in}, w_mask]
        masks_out = [{conv_io_chan_axis: mask_out}]
    else:
        warnings.warn("Can't propagate sparsity mask through op_type %s" % node.op_type)
    return (masks_in, masks_out)


class ApplyMasks(Transformation):
    """Apply the given sparsity masks in prune_spec to the appropriately named
    tensors in the model. These masks are only annotations, no actual pruning
    is performed at this stage."""

    def __init__(self, prune_spec: Dict) -> None:
        super().__init__()
        self.prune_spec = prune_spec

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        # sanity check:
        # - prune spec must be a dict
        assert isinstance(self.prune_spec, dict)
        for key, val in self.prune_spec.items():
            # sanity check:
            # - prune spec keys must be strings (tensor names)
            assert isinstance(key, str)
            # - prune spec vals must also be dicts
            assert isinstance(val, dict)
            model.set_tensor_sparsity(key, val)
        return (model, False)


class PropagateMasks(Transformation):
    """Propagate the sparsity masks in the network to relevant upstream and
    downstream layers. Some inital sparsity masks must have been applied
    either manually or with the ApplyMasks transformation. Note that not all
    layer types are supported; see the update_node_mask function for details."""

    def __init__(self, lossy: bool = True) -> None:
        super().__init__()
        self.lossy = lossy

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        need_rerun = False
        # traverse graph from inputs to outputs to propagate
        # sparsity masks via per-layer handlers
        for node in model.graph.node:
            node_masks_in = [model.get_tensor_sparsity(x) for x in node.input]
            node_masks_out = [model.get_tensor_sparsity(x) for x in node.output]
            # ensure all mask types are considered as sets
            # otherwise we end up comparing None and dict()
            node_masks_in = [ensure_masktype_is_dict(x) for x in node_masks_in]
            node_masks_out = [ensure_masktype_is_dict(x) for x in node_masks_out]
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
    """Remove channels indicated by sparsity masks on the model. The sparsity
    mask annotations will be removed after they have been processed for each
    tensor. Does not perform any shape consistency checking and may result in
    a broken graph."""

    def __init__(self, lossy: bool = True) -> None:
        super().__init__()
        self.lossy = lossy

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
                    for target_axis, axis_mask in mask.items():
                        new_shp = remove_masked_tensor_channels(io_shp, axis_mask, axis=target_axis)
                        model.set_tensor_shape(ioname, new_shp)
                else:
                    if node.op_type in ["MatMul"]:
                        w_axis_in = optype_to_w_axis[node.op_type]["in"]
                        w_axis_out = optype_to_w_axis[node.op_type]["out"]
                        w_mask_in = mask.get(w_axis_in, set())
                        w_mask_out = mask.get(w_axis_out, set())
                        new_t = remove_masked_tensor_channels(io_t, w_mask_in, axis=w_axis_in)
                        new_t = remove_masked_tensor_channels(new_t, w_mask_out, axis=w_axis_out)
                        model.set_initializer(ioname, new_t)
                    elif node.op_type in ["Conv"]:
                        w_axis_in = optype_to_w_axis[node.op_type]["in"]
                        w_axis_out = optype_to_w_axis[node.op_type]["out"]
                        w_mask_in = mask.get(w_axis_in, set())
                        w_mask_out = mask.get(w_axis_out, set())
                        ifm_w = io_shp[1]
                        groups = get_by_name(node.attribute, "group")
                        if groups is not None:
                            groups = groups.i
                        else:
                            groups = 1
                        assert groups == 1 or ifm_w == 1, "Unknown grouped conv setting"
                        depthwise = groups > 1
                        if depthwise:
                            # depthwise convs only use the o_mask by convention
                            new_t = remove_masked_tensor_channels(io_t, w_mask_out, axis=w_axis_out)
                            # need to update the group attribute to match new n chans
                            get_by_name(node.attribute, "group").i = new_t.shape[0]
                        else:
                            new_t = remove_masked_tensor_channels(io_t, w_mask_in, axis=w_axis_in)
                            new_t = remove_masked_tensor_channels(new_t, w_mask_out, axis=w_axis_out)
                        model.set_initializer(ioname, new_t)
                    else:
                        new_t = io_t
                        for target_axis, axis_mask in mask.items():
                            if target_axis >= new_t.ndim:
                                # for layers that use broadcasting, param dims have lower dim than input dims
                                # and the target axis won't exist. try to handle those cases appropriately
                                if new_t.ndim == 1:
                                    new_t = remove_masked_tensor_channels(new_t, axis_mask, axis=0)
                                    model.set_initializer(ioname, new_t)
                                elif new_t.ndim == 0:
                                    # don't prune scalar param
                                    continue
                                else:
                                    assert False, "Cannot prune paramet tensor %s with shape %s using spec %s" % (
                                        ioname,
                                        str(io_shp),
                                        str(mask),
                                    )
                            else:
                                new_t = remove_masked_tensor_channels(new_t, axis_mask, axis=target_axis)
                                model.set_initializer(ioname, new_t)
                # clear sparsity annotation since it's already handled
                # leftover annotations here can lead to erronous removal later
                model.set_tensor_sparsity(ioname, {})
                new_shp = model.get_tensor_shape(ioname)
                # print("[RemoveMaskedChannels] tensor %s : new shape %s" % (ioname, str(new_shp)))
                need_rerun = True
        return (model, need_rerun)


class PruneChannels(Transformation):
    """
    Prune channels from specified tensors and their dependencies from a model, as
    specified by the dictionary given in prune_spec.
    This dictionary must be formatted as {tensor_name : {axis : {channels}}}.
    See test_pruning.py for examples.
    If lossy is True, the transformation will aggresively prune all relevant
    upstream/downstream layers around the specified tensors. This is good for
    maintaining the consistency of layer shapes, but may introduce a larger accuracy
    penalty. If lossy is False, the pruning will be more conservative to preserve the
    numerical ranges (e.g. biases won't be pruned in the downstream layers) but this
    may lead to inconsistent shapes in the network.
    """

    def __init__(self, prune_spec: Dict, lossy: bool = True) -> None:
        super().__init__()
        self.prune_spec = prune_spec
        self.lossy = lossy
        if not lossy:
            warnings.warn("The current implementation for lossless channel pruning will fail for some topologies")

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        # check for known patterns that break the pruning transformation
        conv_nodes = model.get_nodes_by_op_type("Conv")
        matmul_nodes = model.get_nodes_by_op_type("MatMul")
        convs_with_bias = [x for x in conv_nodes if len(x.input) == 3]
        if len(convs_with_bias) > 0:
            assert False, "Found Conv nodes with bias, please use cleanup with extract_conv_bias=True first: %s" % str(
                [x.name for x in convs_with_bias]
            )
        dotprod_nodes = conv_nodes + matmul_nodes
        dotprod_nodes_dyn_w = [x for x in dotprod_nodes if model.get_initializer(x.input[1]) is None]
        if len(dotprod_nodes_dyn_w) > 0:
            assert False, (
                "Found MatMul or Conv nodes with non-static weights. "
                "If this is due to weight quantizer ops, try cleanup with preserve_qnt_ops=False: "
                + str([x.name for x in dotprod_nodes_dyn_w])
            )
        model = model.transform(ApplyMasks(self.prune_spec))
        model = model.transform(PropagateMasks(self.lossy))
        model = model.transform(RemoveMaskedChannels(self.lossy))
        return (model, False)
