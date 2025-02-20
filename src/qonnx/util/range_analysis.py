# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
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

import clize
import dataclasses as dc
import itertools
import numpy as np
import pprint
from onnx import ValueInfoProto
from warnings import warn

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_node
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from qonnx.transformation.general import ConvertDivToMul, ConvertSubToAdd
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.util.basic import get_by_name
from qonnx.util.cleanup import cleanup_model
from qonnx.util.onnx import node_to_model, valueinfo_to_tensor

# walk the graph to deduce range information about each tensor
# assumptions:
# - layout and shape inference already completed
# - range info is generated per-element, if un-broadcasting is enabled,
#   identical elements along axes will be de-duplicated back to a shape broadcastable
#   to the elementwise shape


# try to recover original broadcasted array by applying np.unique along all
# the axes, and keeping the ones that reduce that dimension to 1
def unbroadcast_tensor(array):
    if array is None:
        return None
    ret_cand = array
    for dim_ind in range(array.ndim):
        new_cand = np.unique(ret_cand, axis=dim_ind)
        if new_cand.shape[dim_ind] == 1:
            ret_cand = new_cand
    return ret_cand


# range (tuple of tensors) version of unbroadcast_tensor
def unbroadcast_range(range):
    if range is None:
        return None
    unb_0 = unbroadcast_tensor(range[0])
    unb_1 = unbroadcast_tensor(range[1])
    return (unb_0, unb_1)


# apply unbroadcasting to all RangeInfo in dict
def unbroadcast_range_dict(range_dict: dict):
    ret_dict = {}
    for key, val in range_dict.items():
        ret_dict[key] = val.unbroadcast()
    return ret_dict


def is_dyn_input(x, model):
    # return True if a given tensor has no initializer (=dynamic), False otherwise
    return model.get_initializer(x) is None and x != ""


def broadcast_range(tensor_range: tuple, tensor_vi_or_shape):
    # ensure the range has the apropriate (per-element shape)
    # i.e. range = (range_min, range_max) where range_min and
    # range_max have the same shape as the original tensor
    if tensor_range is None:
        return None
    if isinstance(tensor_vi_or_shape, ValueInfoProto):
        proto_tensor = valueinfo_to_tensor(tensor_vi_or_shape)
        tensor_shape = proto_tensor.shape
    else:
        tensor_shape = tensor_vi_or_shape
        proto_tensor = np.zeros(tensor_shape, np.float32)
    # fix shape using numpy broadcasting
    range_min = np.broadcast_to(tensor_range[0], proto_tensor.shape).astype(proto_tensor.dtype)
    range_max = np.broadcast_to(tensor_range[1], proto_tensor.shape).astype(proto_tensor.dtype)
    return (range_min, range_max)


# RangeInfo dataclass: we will use instances of this to represent the range information for tensors
@dc.dataclass
class RangeInfo:
    # full shape of the tensor
    shape: tuple = None
    # the range encountered in practice when observing tensors during inference
    range: tuple = None
    # (optional) the underlying integer range for tensor, if applicable
    # if this is set, so are scale and bias, to satisfy:
    # range = scale * int_range + bias
    int_range: tuple = None
    # (optional) the scaling factor applied to the integer range, if applicable
    scale: np.ndarray = None
    # (optional) the bias applied after the scaling, if applicable
    bias: np.ndarray = None
    # whether this particular range is always fixed (due to its tensor having an initializer)
    is_initializer: bool = False
    # history of scale/bias tensors suitable for later removal during streamlining
    history_scale: list = dc.field(default_factory=lambda: [])
    history_bias: list = dc.field(default_factory=lambda: [])

    def is_point_interval(self):
        # whether this is a point interval (min=max for all elements)
        return (self.range[0] == self.range[1]).all()

    def has_integer_info(self) -> bool:
        # whether the RangeInfo has its int_range, scale and bias populated
        integer_props = [self.int_range, self.scale, self.bias]
        return all([x is not None for x in integer_props])

    def unbroadcast(self):
        return RangeInfo(
            shape=self.shape,
            range=unbroadcast_range(self.range),
            int_range=unbroadcast_range(self.int_range),
            scale=unbroadcast_tensor(self.scale),
            bias=unbroadcast_tensor(self.bias),
            is_initializer=self.is_initializer,
        )

    def broadcast(self):
        return RangeInfo(
            shape=self.shape,
            range=broadcast_range(self.range, self.shape),
            int_range=broadcast_range(self.int_range, self.shape),
            scale=np.broadcast_to(self.scale, self.shape),
            bias=np.broadcast_to(self.bias, self.shape),
            is_initializer=self.is_initializer,
        )


# range computation for monotonic functions:
# suppose we have a layer z = f(x,y) taking in two inputs x and y, outputting z
# suppose that these have ranges x = (xmin, xmax), y = (ymin, ymax), z = (zmin, zmax)
# say we have access to the input ranges, and want to find the output range
# a monotonic function will have the property that the inputs that trigger zmin and zmax
# can be found at the "corners" of the input space. so we evaluate the function at all
# possible corners of the input space:
# c0 = f(xmin, ymin)
# c1 = f(xmax, ymin)
# c2 = f(xmin, ymax)
# c3 = f(xmax, ymax)
# now, we can find our output range by taking the min/max of these corners
# zmin = min(c0, c1, c2, c3)
# zmax = max(c0, c1, c2, c3)
def calc_monotonic_range(node, model, range_dict):
    opset_version = model.model.opset_import[0].version
    oname = node.output[0]
    dyn_inps = [x for x in node.input if is_dyn_input(x, model)]
    n_dyn_inp = len(dyn_inps)
    # create context for single-node execution
    ctx = {x: model.get_initializer(x) for x in node.input}
    for oname in node.output:
        ctx[oname] = valueinfo_to_tensor(model.get_tensor_valueinfo(oname))
    if n_dyn_inp == 0:
        # special case: all inputs were constants (e.g. quantized for trained weights)
        # so there is no proto vectors to operate over really - just need a single eval
        execute_node(node, ctx, model.graph, opset_version=opset_version)
        # grab new output and keep the entire thing as the range
        for oname in node.output:
            range_dict[oname].range = (ctx[oname], ctx[oname])
            range_dict[oname].is_initializer = True
        return
    # going beyond this point we are sure we have at least one dynamic input
    # generate min-max prototype vectors for each dynamic input
    proto_vectors = []
    for inp in dyn_inps:
        irange = range_dict[inp].range
        inp_vi = model.get_tensor_valueinfo(inp)
        # this handler is execution-based so make sure we have per-element ranges
        proto_vectors.append(broadcast_range(irange, inp_vi))
    # process all combinations of prototype vectors for dynamic inputs
    running_min = [None for i in range(len(node.output))]
    running_max = [None for i in range(len(node.output))]
    for inps in itertools.product(*proto_vectors):
        for i in range(n_dyn_inp):
            ctx[dyn_inps[i]] = inps[i]
        execute_node(node, ctx, model.graph, opset_version=opset_version)
        for oind, oname in enumerate(node.output):
            # grab new output and update running min/max
            out = ctx[oname]
            running_min[oind] = np.minimum(out, running_min[oind]) if running_min[oind] is not None else out
            running_max[oind] = np.maximum(out, running_max[oind]) if running_max[oind] is not None else out
    for oind, oname in enumerate(node.output):
        range_dict[oname].range = (running_min[oind], running_max[oind])


# fast interval matrix enclosure based on:
# Accelerating interval matrix multiplication by mixed precision arithmetic
# Ozaki et al.
# Algorithms 1 and 2, which turn are based on:
# Rump, Siegfried M. "INTLAB—interval laboratory." Developments in reliable computing.
# except no directed rounding (because numpy/Python has none)
def range_to_midpoint_radius(matrix_range):
    (matrix_min, matrix_max) = matrix_range
    midpoint = matrix_min + 0.5 * (matrix_max - matrix_min)
    radius = midpoint - matrix_min
    return (midpoint, radius)


def calc_matmul_range(range_A, range_B):
    (midpoint_A, radius_A) = range_to_midpoint_radius(range_A)
    (midpoint_B, radius_B) = range_to_midpoint_radius(range_B)
    radius = np.matmul(radius_A, np.abs(midpoint_B) + radius_B) + np.matmul(np.abs(midpoint_A), radius_B)
    out_base = np.matmul(midpoint_A, midpoint_B)
    out_max = out_base + radius
    out_min = out_base - radius
    return (out_min, out_max)


def calc_matmul_node_range(node, model, range_dict):
    # matmul range is execution-based so broadcast to elementwise shapes
    # otherwise we'll get shape mismatch problems for np.matmul
    range_A = range_dict[node.input[0]].broadcast().range
    range_B = range_dict[node.input[1]].broadcast().range
    range_dict[node.output[0]].range = calc_matmul_range(range_A, range_B)


# use inferred output datatype to calculate output ranges
def calc_range_outdtype(node, model, range_dict):
    oname = node.output[0]
    odt = model.get_tensor_datatype(oname)
    assert odt is not None, "Cannot infer %s range, dtype annotation is missing" % oname
    range_dict[oname].range = (odt.min(), odt.max())


# Softmax always produces outputs in [0,1]
def calc_softmax_range(node, model, range_dict):
    oname = node.output[0]
    assert node.op_type == "Softmax"
    range_dict[oname].range = (0, 1)


# LogSoftmax always produces outputs in [-inf,0], which is the log of the range
# of the Softmax
def calc_logsoftmax_range(node, model, range_dict):
    oname = node.output[0]
    assert node.op_type == "LogSoftmax"
    # Note: Replaces -inf by the smallest representable float 32 value
    range_dict[oname].range = (DataType["FLOAT32"].min(), 0)


# return whether a given tensor is a shape operand
def is_shape_operand(tensor_name, model):
    cons = model.find_consumer(tensor_name)
    if cons is not None:
        if cons.op_type == "Reshape" and list(cons.input).index(tensor_name) == 1:
            return True
    return False


# use initializers to mark point ranges i.e. tensor with initializer X has range (X, X)
def calc_range_all_initializers(model, range_dict):
    all_tensor_names = model.get_all_tensor_names()
    for tensor_name in all_tensor_names:
        tensor_init = model.get_initializer(tensor_name)
        if tensor_init is not None:
            range_dict[tensor_name] = RangeInfo(
                shape=tensor_init.shape, range=(tensor_init, tensor_init), is_initializer=True
            )
            # use % 1 == 0 to identify initializers with integer values (except shape
            # operands which would give rise to false scaled-int propagation)
            if ((tensor_init % 1) == 0).all() and not is_shape_operand(tensor_name, model):
                range_dict[tensor_name].int_range = (tensor_init, tensor_init)
                range_dict[tensor_name].scale = np.asarray([1.0], dtype=np.float32)
                range_dict[tensor_name].bias = np.asarray([0.0], dtype=np.float32)


# for several types of nodes, we dynamically convert ("lower") the node to something else that we can
# process before making a recursive call to range analysis and put those results back in
def calc_range_with_lowering(prep_transforms, lowering_transforms, node, model, range_dict):
    # run prep transforms to ensure lowering on single node will work correctly
    prep_model = model
    for trafo in prep_transforms:
        prep_model = prep_model.transform(trafo)
    # create a single-node model from this node
    node_model = ModelWrapper(node_to_model(node, prep_model))
    # run lowering pipeline on the single-node model
    for trafo in lowering_transforms:
        node_model = node_model.transform(trafo)
    # copy RangeInfo pertaining to node_model's top-level inputs to a new dict
    node_range_dict = {}
    for node_inp in node_model.graph.input:
        node_range_dict[node_inp.name] = range_dict[node_inp.name]
    # run range analysis on the lowered single-node model
    ret_range_dict, _ = range_analysis(node_model, irange=node_range_dict, report_mode=REPORT_MODE_RANGE)
    # copy results back into original range_dict
    for node_out in node.output:
        range_dict[node_out] = ret_range_dict[node_out]


def calc_conv_range(node, model, range_dict):
    prep_transforms = [FoldConstants(exclude_op_types=[])]
    lowering_transforms = [LowerConvsToMatMul()]
    calc_range_with_lowering(prep_transforms, lowering_transforms, node, model, range_dict)


def calc_bn_range(node, model, range_dict):
    prep_transforms = []
    lowering_transforms = [BatchNormToAffine()]
    calc_range_with_lowering(prep_transforms, lowering_transforms, node, model, range_dict)


def calc_sub_range(node, model, range_dict):
    prep_transforms = []
    lowering_transforms = [ConvertSubToAdd()]
    calc_range_with_lowering(prep_transforms, lowering_transforms, node, model, range_dict)


def calc_div_range(node, model, range_dict):
    prep_transforms = []
    lowering_transforms = [ConvertDivToMul()]
    calc_range_with_lowering(prep_transforms, lowering_transforms, node, model, range_dict)


def calc_gemm_range(node, model, range_dict):
    prep_transforms = []
    lowering_transforms = [GemmToMatMul()]
    calc_range_with_lowering(prep_transforms, lowering_transforms, node, model, range_dict)


# integer quantized NNs often carry around "scaled integers": instead of pure
# int operations, we typically have integers scaled by some scaling factor (which
# may come at different granularities: tensor-wise, channel-wise...) and possibly
# biased by some offset ("zero point" among other things)
# the scaled-int range analysis tries to distinguish the underlying integer parts
# of scaled-int tensors from any non-integer scaling factors and biases.
# this analysis is meant to run as a second step after regular range analysis is
# executed, and we already have derived elementwise (min, max) values for each
# intermediate tensors.


# return a list of Bool values, indicating whether each input to a node
# has integer range info associated with it (True) or not (False)
def check_int_inputs(node, range_dict):
    inp_int_info = [range_dict[x].has_integer_info() for x in node.input]
    return inp_int_info


# for nodes that directly "pick" their output elements from input elements,
# e.g permutation with or without repetition, we try to infer output scale and bias
# by executing that node with its input scale/bias as the input. if scale/bias on an
# input is not available, we check if that input is a constant (point interval) and use
# that value instead.
def calc_scalebias_from_execution(node, model, range_dict):
    opset_version = model.model.opset_import[0].version
    if len(node.output) > 1:
        warn("Cannot infer scale for multi-output node")
        return
    ctx = {}
    ctx[node.output[0]] = np.zeros(range_dict[node.output[0]].shape, dtype=np.float32)
    for inp in node.input:
        if not (range_dict[inp].scale is None):
            ctx[inp] = np.broadcast_to(range_dict[inp].scale, range_dict[inp].shape)
        elif range_dict[inp].is_point_interval():
            ctx[inp] = np.broadcast_to(range_dict[inp].range[0], range_dict[inp].shape)
        else:
            warn(f"Cannot infer scale for f{node.name}")
            return
    execute_node(node, ctx, model.graph, opset_version=opset_version)
    range_dict[node.output[0]].scale = ctx[node.output[0]]
    for inp in node.input:
        if not (range_dict[inp].bias is None):
            ctx[inp] = np.broadcast_to(range_dict[inp].bias, range_dict[inp].shape)
        elif range_dict[inp].is_point_interval():
            ctx[inp] = np.broadcast_to(range_dict[inp].range[0], range_dict[inp].shape)
        else:
            warn(f"Cannot infer bias for f{node.name}")
            return
    execute_node(node, ctx, model.graph, opset_version=opset_version)
    range_dict[node.output[0]].bias = ctx[node.output[0]]


# for a node that already has its output scale & bias computed, compute
# the integer range based on the full range
def calc_intrange_from_scalebias(node, model, range_dict):
    for oname in node.output:
        orange = range_dict[oname]
        if (orange.scale is None) or (orange.bias is None):
            warn("%s.%s has no output scale/bias, skipping" % (node.name, oname))
            continue
        # min/max may swap places due to negative scales
        min_cand = (orange.range[0] - orange.bias) / orange.scale
        max_cand = (orange.range[1] - orange.bias) / orange.scale
        orange_int_min = np.minimum(min_cand, max_cand)
        orange_int_min = np.round(orange_int_min)
        orange_int_max = np.maximum(min_cand, max_cand)
        orange_int_max = np.round(orange_int_max)
        range_dict[oname].int_range = (orange_int_min, orange_int_max)


# propagate integer range and scale/bias info for ReLU
def calc_intrange_relu(node, model, range_dict):
    inp_int_info = check_int_inputs(node, range_dict)
    if not any(inp_int_info):
        # must have at least one input with integer info, otherwise no point
        warn(node.name + " has no integer info on inputs, cannot propagate")
        return
    # irange_inf = range_dict[node.input[0]]
    # we'll use the ReLU output range to infer the integer parts
    # * output range can only come from the ReLU identity part (input > 0)
    # * scale and bias are always left unchanged, unless stuck channel
    # range_max = S*int_range_max + B
    # range_min = S*int_range_min + B
    # S and B are identical between input and output
    # range_dict[node.output[0]].scale = irange_inf.scale
    # range_dict[node.output[0]].bias = irange_inf.bias
    # calc_intrange_from_scalebias(node, model, range_dict)
    # scale/bias history is reset
    range_dict[node.output[0]].history_scale = []
    range_dict[node.output[0]].history_bias = []


# propagate integer range and scale/bias info for Quant
def calc_intrange_quant(node, model, range_dict):
    # get quantizer parameters
    q_scale = model.get_initializer(node.input[1])
    q_zeropt = model.get_initializer(node.input[2])
    q_bitwidth = model.get_initializer(node.input[3])
    scale_ok = not (q_scale is None)
    zeropt_ok = not (q_zeropt is None)
    bitwidth_ok = not (q_bitwidth is None)
    if not (scale_ok and zeropt_ok and bitwidth_ok):
        warn("%s has non-constant quantizer inputs, skipping" % node.name)
        return
    # we need to do a little style conversion for the scale/bias:
    # intrange calculations here represent quant tensors as Mx+N (x: int tensor, M: scale, N: bias)
    # whereas Quant nodes represent them as S(x-Z) (x: int tensor, S: scale, Z: zeropoint)
    # it follows that M = S and N = -SZ
    # TODO broadcast these to element shape?
    range_dict[node.output[0]].scale = q_scale
    range_dict[node.output[0]].bias = -(q_scale * q_zeropt)
    calc_intrange_from_scalebias(node, model, range_dict)
    # Quant nodes override the history for scale and bias, and sets them to
    # what was locally specified if they were not 1 and 0
    if (q_scale == 1).all():
        range_dict[node.output[0]].history_scale = []
    else:
        range_dict[node.output[0]].history_scale = [node.input[1]]
    if (q_zeropt == 0).all():
        range_dict[node.output[0]].history_bias = []
    else:
        range_dict[node.output[0]].history_bias = [node.input[2]]


# propagate integer range and scale/bias info for Trunc
def calc_intrange_trunc(node, model, range_dict):
    # get quantizer parameters
    t_scale = model.get_initializer(node.input[1])
    t_zeropt = model.get_initializer(node.input[2])
    t_bitwidth_in = model.get_initializer(node.input[3])
    t_bitwidth_out = model.get_initializer(node.input[4])
    scale_ok = not (t_scale is None)
    zeropt_ok = not (t_zeropt is None)
    in_bitwidth_ok = not (t_bitwidth_in is None)
    out_bitwidth_ok = not (t_bitwidth_out is None)
    if not (scale_ok and zeropt_ok and in_bitwidth_ok and out_bitwidth_ok):
        warn("%s has non-constant quantization parameters, skipping" % node.name)
        return
    # we need to do a little style conversion for the scale/bias:
    # intrange calculations here represent quant tensors as Mx+N (x: int tensor, M: scale, N: bias)
    # whereas Trunc nodes represent them as S(x-Z) (x: int tensor, S: scale, Z: zeropoint)
    # it follows that M = S and N = -SZ
    # TODO broadcast these to element shape?
    range_dict[node.output[0]].scale = t_scale
    range_dict[node.output[0]].bias = -(t_scale * t_zeropt)
    calc_intrange_from_scalebias(node, model, range_dict)
    # Trunc nodes override the history for scale and bias, and sets them to
    # what was locally specified if they were not 1 and 0
    if (t_scale == 1).all():
        range_dict[node.output[0]].history_scale = []
    else:
        range_dict[node.output[0]].history_scale = [node.input[1]]
    if (t_zeropt == 0).all():
        range_dict[node.output[0]].history_bias = []
    else:
        range_dict[node.output[0]].history_bias = [node.input[2]]


# propagates scale/bias info as-is without any changes
# but recalculate the int range info based on actual shapes
def calc_intrange_identity(node, model, range_dict):
    n_dyn_inps = [(model.get_initializer(x) is None) for x in node.input].count(True)
    assert n_dyn_inps == 1, "Identity int range prop needs a single dynamic input"
    irange_inf = range_dict[node.input[0]]
    for o in node.output:
        # TODO this will break for nodes that can change the output shape
        # when the scale/bias are not scalars but tensors, they will also change
        # shape after e.g. Transpose/Reshape/...
        range_dict[o].scale = irange_inf.scale
        range_dict[o].bias = irange_inf.bias
        # derive integer ranges from the full range using scale & bias info
        orange_inf = range_dict[o]
        int_min = (orange_inf.range[0] - orange_inf.bias) / orange_inf.scale
        int_max = (orange_inf.range[1] - orange_inf.bias) / orange_inf.scale
        int_min = np.round(int_min)
        int_max = np.round(int_max)
        orange_inf.int_range = (int_min, int_max)
        range_dict[o] = orange_inf
        # scale/bias history is not altered
        range_dict[o].history_scale = irange_inf.history_scale
        range_dict[o].history_bias = irange_inf.history_bias


def interval_prod(interval_a, interval_b):
    a_min, a_max = interval_a
    b_min, b_max = interval_b
    c0, c1, c2, c3 = a_min * b_min, a_min * b_max, a_max * b_min, a_max * b_max
    c_min = np.minimum(c3, np.minimum(c2, np.minimum(c1, c0)))
    c_max = np.maximum(c3, np.maximum(c2, np.maximum(c1, c0)))
    return (c_min, c_max)


# return a list of Bool values, indicating whether each input to a node
# has a point interval associated with it (True) or not (False)
def check_point_interval_inputs(node, range_dict):
    inp_pointinterval_info = [range_dict[x].is_point_interval() for x in node.input]
    return inp_pointinterval_info


def calc_intrange_add(node, model, range_dict):
    # our ability to do scaled-int range propagation depends on whether (some)
    # inputs have an integer component already
    inp_int_info = check_int_inputs(node, range_dict)
    if not any(inp_int_info):
        # must have at least one input with integer info, otherwise no point
        warn(node.name + " has no integer info on inputs, cannot propagate")
        return

    # we can do scaled-int range propagation for addition in two cases:
    # 1) when one of the intervals is a point interval (always treated as extra bias)
    # 2) when both intervals have matching scales
    ptint_info = check_point_interval_inputs(node, range_dict)

    if any(ptint_info):
        if all(ptint_info):
            # special case - all inputs are constants. would normally be const-folded
            # assume input 0 is the main input, input 1 is the bias
            ptint_inpname = node.input[1]
            irange_ptint = range_dict[ptint_inpname]
            nonptint_inpname = node.input[0]
            irange_nonptint = range_dict[nonptint_inpname]
        else:
            # point interval will go into scale and bias
            ptint_inpname = node.input[ptint_info.index(True)]
            irange_ptint = range_dict[ptint_inpname]
            nonptint_inpname = node.input[ptint_info.index(False)]
            irange_nonptint = range_dict[nonptint_inpname]
        # the non-point interval must have int info
        if irange_nonptint.int_range is None:
            warn(node.name + " unsupported combination of point and int intervals, cannot propagate")
            return
        # f + s*[i_min, i_max] + b so absorb the point interval into bias
        # = s*[i_min, i_max] + (f + b)
        range_dict[node.output[0]].int_range = irange_nonptint.int_range
        range_dict[node.output[0]].scale = irange_nonptint.scale
        range_dict[node.output[0]].bias = irange_nonptint.bias + irange_ptint.range[0]
        # scale history inherited from both sides, bias history updated with nonint param
        # (even though the scale is the same on both branches we need to track the tensors that
        # contributed to it on both sides)
        # TODO should we add the entire tensor generating the point interval to the bias history here?
        # TODO correspondingly - disconnect subgraphs entirely if their output is set to an initializer?
        range_dict[node.output[0]].history_scale = irange_nonptint.history_scale + irange_ptint.history_scale
        range_dict[node.output[0]].history_bias = irange_nonptint.history_bias + irange_ptint.history_bias + [ptint_inpname]
    else:
        # when all inputs are int, we can combine the scale factors into a single one
        # if there is an integer relationship between the scale factors
        irange_0 = range_dict[node.input[0]]
        irange_1 = range_dict[node.input[1]]
        # TODO: extend to integer relationship - for now only handle equal
        # if not, we'll fallback to mix of integer and non-integer operands
        if (irange_0.scale == irange_1.scale).all():
            # s*[i0_min, i0_max] + b0 + s*[i1_min, i1_max] + b1
            # = s*[i0_min+i1_min, i0_max+i1_max] + b0 + b1
            range_dict[node.output[0]].scale = irange_0.scale
            range_dict[node.output[0]].bias = irange_0.bias + irange_1.bias
            range_dict[node.output[0]].int_range = (
                irange_0.int_range[0] + irange_1.int_range[0],
                irange_0.int_range[1] + irange_1.int_range[1],
            )
            # scale history and bias history inherits from both sides
            # (even though the scale is the same on both branches we need to track the tensors that
            # contributed to it on both sides)
            range_dict[node.output[0]].history_scale = irange_0.history_scale + irange_1.history_scale
            range_dict[node.output[0]].history_bias = irange_0.history_bias + irange_1.history_bias
        else:
            warn(node.name + " incompatible scales for int intervals, cannot propagate")
            return


def calc_intrange_mul(node, model, range_dict):  #
    # our ability to do scaled-int range propagation depends on whether (some)
    # inputs have an integer component already
    inp_int_info = check_int_inputs(node, range_dict)
    if not any(inp_int_info):
        # must have at least one input with integer info, otherwise no point
        warn(node.name + " has no integer info on inputs, cannot propagate")
        return

    # we can do scaled-int range propagation for multiplication in two cases:
    # 1) when one of the intervals is a point interval (incorporated into scale and bias)
    # 2) when both int intervals have zero-valued bias
    ptint_info = check_point_interval_inputs(node, range_dict)

    if any(ptint_info):
        if all(ptint_info):
            # special case - all inputs are constants. would normally be const-folded
            # assume input 0 is the main input, input 1 is the scale
            ptint_inpname = node.input[1]
            irange_ptint = range_dict[ptint_inpname]
            nonptint_inpname = node.input[0]
            irange_nonptint = range_dict[nonptint_inpname]
        else:
            # point interval will go into scale and bias
            ptint_inpname = node.input[ptint_info.index(True)]
            irange_ptint = range_dict[ptint_inpname]
            nonptint_inpname = node.input[ptint_info.index(False)]
            irange_nonptint = range_dict[nonptint_inpname]
        # the non-point interval must have int info
        if irange_nonptint.int_range is None:
            warn(node.name + " unsupported combination of point and int intervals, cannot propagate")
            return
        # f * (s*[i_min, i_max] + b)
        # = f*s*[i_min, i_max] + (f * b)
        range_dict[node.output[0]].int_range = irange_nonptint.int_range
        range_dict[node.output[0]].scale = irange_nonptint.scale * irange_ptint.range[0]
        range_dict[node.output[0]].bias = irange_nonptint.bias * irange_ptint.range[0]
        # scale history updated with the nonint param, bias history remains
        # (even though the bias changes, don't want to track the same tensor for both bias and scale histories)
        range_dict[node.output[0]].history_scale = (
            irange_nonptint.history_scale + irange_ptint.history_scale + [ptint_inpname]
        )
        range_dict[node.output[0]].history_bias = irange_nonptint.history_bias + irange_ptint.history_bias
    elif all(inp_int_info):
        # when all inputs are int and biases are zero, we can multiply the
        # scale factors to get the output scale factor, and similarly multiply
        # the input int intervals to get the output int interval
        # s0*[i0_min, i0_max] * s1*[i1_min, i1_max]
        # = s0*s1*interval_prod([i0_min, i0_max], [i1_min_, i1_max])
        irange_0 = range_dict[node.input[0]]
        irange_1 = range_dict[node.input[1]]
        if (irange_0.bias == 0).all() and (irange_1.bias == 0).all():
            range_dict[node.output[0]].scale = irange_0.scale * irange_1.scale
            range_dict[node.output[0]].bias = np.asarray(0, dtype=np.float32)
            range_dict[node.output[0]].int_range = interval_prod(irange_0.int_range, irange_1.int_range)
            # bias history (should be empty) and scale history inherits from both sides
            range_dict[node.output[0]].history_scale = irange_0.history_scale + irange_1.history_scale
            range_dict[node.output[0]].history_bias = irange_0.history_bias + irange_1.history_bias
        else:
            warn(node.name + " nonzero biases for Mul, cannot propagate")
            return
    else:
        warn(node.name + " unsupported pattern, cannot propagate")
        return


def check_matmul_for_intrange_prop(node, range_dict):
    irange_0_inf = range_dict[node.input[0]]
    irange_1_inf = range_dict[node.input[1]]
    if not irange_0_inf.has_integer_info():
        warn(f"Input 0 of {node.name} has undefined bias, scale or int_range, can't do scaled-int propagation")
        return False
    if not irange_1_inf.has_integer_info():
        warn(f"Input 1 of {node.name} has undefined bias, scale or int_range, can't do scaled-int propagation")
        return False
    # for the output dot product to have a non-dynamic scale & bias, we need to put
    # some constraints on the scale & bias for the inputs
    # for the output dot product to have a non-dynamic scale & bias, we need to put
    # some constraints on the scale & bias for the inputs
    # for biases, we need the following conditions:
    # - one of the inputs must be constant-valued (point interval)
    # - the bias of the constant-valued input must be zero
    i0_bias_ok = irange_0_inf.is_point_interval() and (irange_0_inf.bias == 0).all()
    i1_bias_ok = irange_1_inf.is_point_interval() and (irange_1_inf.bias == 0).all()
    if not (i0_bias_ok or i1_bias_ok):
        warn(f"Non-const inputs of {node.name} have non-zero bias, can't do scaled-int propagation")
        return False
    # ensure scale information is un-broadcasted so we can check shapes etc properly
    i0_scale = unbroadcast_tensor(irange_0_inf.scale)
    i1_scale = unbroadcast_tensor(irange_0_inf.scale)
    # for a MatMul of shape (MxK) x (KxN) with scaling: we cannot have scales along the
    # dot product dimension, but per-tensor or per-dot-product are fine
    # i.e. either the scale is a scalar, or it has a non-1-shaped dimension for either
    # M (for input 0) or N (input 1) dimensions
    if i0_scale.size != 1:
        acceptable_scale_i0 = [1] * len(irange_0_inf.shape)
        acceptable_scale_i0[-2] = irange_0_inf.shape[-2]
        if list(irange_0_inf.scale.shape) != acceptable_scale_i0:
            warn(
                f"""Input 0 of {node.name} has scale {str(irange_0_inf.scale.shape)},
                but we need at most {str(acceptable_scale_i0)} so can't do scaled-int propagation"""
            )
            return False
    if i1_scale.size != 1:
        acceptable_scale_i1 = [1] * len(irange_1_inf.shape)
        acceptable_scale_i1[-1] = irange_1_inf.shape[-1]
        if list(irange_1_inf.scale.shape) != acceptable_scale_i1:
            warn(
                f"""Input 1 of {node.name} has scale {str(irange_1_inf.scale.shape)},
                but we need at most {str(acceptable_scale_i1)} so can't do scaled-int propagation"""
            )
            return False
    return True


def calc_intrange_matmul(node, model, range_dict):
    inp_int_info = check_int_inputs(node, range_dict)
    if not all(inp_int_info):
        warn(node.name + " does not have all-integer inputs, cannot propagate")
        return
    if not check_matmul_for_intrange_prop(node, range_dict):
        return

    # compute updated scale and bias
    i0scale = unbroadcast_tensor(range_dict[node.input[0]].scale)
    i1scale = unbroadcast_tensor(range_dict[node.input[1]].scale)
    scale = i0scale * i1scale
    # new bias = (bias_0 * scale_1 * input_1) OR (scale_0 * input_0 * bias_1)
    irange_0_inf = range_dict[node.input[0]]
    irange_1_inf = range_dict[node.input[1]]
    i0_bias_ok = irange_0_inf.is_point_interval() and (irange_0_inf.bias == 0).all()
    i1_bias_ok = irange_1_inf.is_point_interval() and (irange_1_inf.bias == 0).all()
    if i0_bias_ok and not i1_bias_ok:
        # new_bias is ((scale_0 * input_0) @ bias_1)
        # where @ is MatMul
        bias = (irange_0_inf.scale * irange_0_inf.int_range[0]) @ np.broadcast_to(irange_1_inf.bias, irange_1_inf.shape)
    elif i1_bias_ok and not i0_bias_ok:
        # new bias is (bias_0 @ (scale_1 * input_1))
        # where @ is MatMul
        bias = np.broadcast_to(irange_0_inf.bias, irange_0_inf.shape) @ (irange_1_inf.scale * irange_1_inf.int_range[0])
    else:
        assert False, f"Unhandled bias condition in {node.name}"
    range_dict[node.output[0]].scale = scale
    range_dict[node.output[0]].bias = bias
    calc_intrange_from_scalebias(node, model, range_dict)
    # inherit scale/bias history from both sides
    range_dict[node.output[0]].history_scale = (
        range_dict[node.input[0]].history_scale + range_dict[node.input[1]].history_scale
    )
    range_dict[node.output[0]].history_bias = range_dict[node.input[0]].history_bias + range_dict[node.input[1]].history_bias


def check_conv_for_intrange_prop(node, range_dict):
    groups = get_by_name(node.attribute, "group")
    if groups is None:
        # default to dense convs
        groups = 1
    else:
        groups = groups.i
    irange_0_inf = range_dict[node.input[0]]
    irange_1_inf = range_dict[node.input[1]]
    if len(node.input) == 3:
        warn(f"{node.name} has bias, not implemented for scaled-int propagation")
        return False
    # inputs have shape N,C,xxx (batch N, channels C, spatial dims xxx)
    ifm = irange_0_inf.shape[1]
    # weights have shape OFM,IFM,xxx (output chans OFM, input chans IFM, kernel spatial dims xxx)
    ofm = irange_1_inf.shape[0]
    # note that we treat "general" grouped convs (1 < groups < ofm) as dense
    is_depthwise = groups == ofm
    if not irange_0_inf.has_integer_info():
        warn(f"Input 0 of {node.name} has undefined bias, scale or int_range, can't do scaled-int propagation")
        return False
    if not irange_1_inf.has_integer_info():
        warn(f"Input 1 of {node.name} has undefined bias, scale or int_range, can't do scaled-int propagation")
        return False
    # for the output dot product to have a non-dynamic scale & bias, we need to put
    # some constraints on the scale & bias for the inputs
    # for biases, we need the following conditions:
    # - one of the inputs must be constant-valued (point interval)
    # - the bias of the constant-valued input must be zero
    i0_bias_ok = irange_0_inf.is_point_interval() and (irange_0_inf.bias == 0).all()
    i1_bias_ok = irange_1_inf.is_point_interval() and (irange_1_inf.bias == 0).all()
    if not (i0_bias_ok or i1_bias_ok):
        warn(f"Non-const inputs of {node.name} have non-zero bias, can't do scaled-int propagation")
        return False
    # ensure scale information is un-broadcasted so we can check shapes etc properly
    i0_scale = unbroadcast_tensor(irange_0_inf.scale)
    i1_scale = unbroadcast_tensor(irange_0_inf.scale)
    if is_depthwise:
        # acceptable scale factor granularities for depthwise convolutions:
        # - inputs (i0) can have per-tensor or per-channel quantization (different channels do not get mixed in same output)
        # - weights (i1) can have per-tensor or per-channel quantization
        # note that ifm = ofm in this case
        acceptable_scale_i0 = [1] * len(irange_0_inf.shape)
        acceptable_scale_i0[1] = ifm
        acceptable_scale_i1 = [1] * len(irange_1_inf.shape)
        acceptable_scale_i1[0] = ifm
        scale_i0_ok = (list(irange_0_inf.scale.shape) == acceptable_scale_i0) or (i0_scale.size == 1)
        scale_i1_ok = (list(irange_1_inf.scale.shape) == acceptable_scale_i1) or (i1_scale.size == 1)
        return scale_i0_ok and scale_i1_ok
    else:
        # acceptable scale factor granularities for dense convolutions:
        # - inputs (i0) can have per-tensor quantization (since different channels get mixed into same output)
        # - weights (i1) can have per-tensor or per-output-channel quantization
        acceptable_scale_i1 = [1] * len(irange_1_inf.shape)
        acceptable_scale_i1[0] = ofm
        scale_i0_ok = i0_scale.size == 1
        scale_i1_ok = (list(irange_1_inf.scale.shape) == acceptable_scale_i1) or (i1_scale.size == 1)
        return scale_i0_ok and scale_i1_ok


def calc_intrange_conv(node, model, range_dict):
    assert len(node.input) == 2, "Must call ExtractConvBias before calc_intrange_conv_nobias"
    inp_int_info = check_int_inputs(node, range_dict)
    if not all(inp_int_info):
        warn(node.name + " does not have all-integer inputs, cannot propagate")
        return
    if not check_conv_for_intrange_prop(node, range_dict):
        return

    # compute updated scale and bias
    iscale_fixed = unbroadcast_tensor(range_dict[node.input[0]].scale)
    wscale_fixed = unbroadcast_tensor(range_dict[node.input[1]].scale)
    if wscale_fixed.ndim > 1:
        target_example = range_dict[node.output[0]].shape
        # ofm,ifm,xx -> 1,ofm,xx
        desired_scale_shape = [1 for x in range(len(target_example))]
        desired_scale_shape[1] = target_example[1]
        wscale_fixed = wscale_fixed.reshape(desired_scale_shape)
    scale = iscale_fixed * wscale_fixed

    # new bias = (bias_0 * scale_1 * input_1) OR (scale_0 * input_0 * bias_1)
    irange_0_inf = range_dict[node.input[0]]
    irange_1_inf = range_dict[node.input[1]]
    i0_bias_ok = irange_0_inf.is_point_interval() and (irange_0_inf.bias == 0).all()
    i1_bias_ok = irange_1_inf.is_point_interval() and (irange_1_inf.bias == 0).all()

    if i0_bias_ok and not i1_bias_ok:
        # new_bias is ((scale_0 * input_0) @ bias_1)
        # where @ is convolution
        node_ctx = {
            node.input[0]: irange_0_inf.scale * irange_0_inf.int_range[0],
            node.input[1]: np.broadcast_to(irange_1_inf.bias, irange_1_inf.shape),
            node.output[0]: np.zeros(range_dict[node.output[0]].shape, dtype=np.float32),
        }
        execute_node(node, node_ctx, model.graph)
        bias = node_ctx[node.output[0]]
    elif i1_bias_ok and not i0_bias_ok:
        # new bias is (bias_0 @ (scale_1 * input_1))
        # where @ is convolution
        node_ctx = {
            node.input[0]: np.broadcast_to(irange_0_inf.bias, irange_0_inf.shape),
            node.input[1]: irange_1_inf.scale * irange_1_inf.int_range[0],
            node.output[0]: np.zeros(range_dict[node.output[0]].shape, dtype=np.float32),
        }
        execute_node(node, node_ctx, model.graph)
        bias = node_ctx[node.output[0]]
    else:
        assert False, f"Unhandled bias condition in {node.name}"
    range_dict[node.output[0]].scale = scale
    range_dict[node.output[0]].bias = unbroadcast_tensor(bias)
    calc_intrange_from_scalebias(node, model, range_dict)
    # inherit scale/bias history from both sides
    range_dict[node.output[0]].history_scale = (
        range_dict[node.input[0]].history_scale + range_dict[node.input[1]].history_scale
    )
    range_dict[node.output[0]].history_bias = range_dict[node.input[0]].history_bias + range_dict[node.input[1]].history_bias


# for nodes such as Im2Col, Reshape, Transpose the scale/bias/per-element output ranges
# are sourced from the scale/bias/per-element input ranges but the pattern can be complicated
def calc_intrange_eltwise_monotonic(node, model, range_dict):
    # TODO smarter decision-making here? e.g. do we need to check for MaxPool axes?
    calc_intrange_eltwise_monotonic_scalebiasfirst(node, model, range_dict)


def calc_intrange_eltwise_monotonic_scalebiasfirst(node, model, range_dict):
    # strategy: execute node only with scale and only with bias to infer what the output scale and bias
    # values are, then infer the integer ranges from that and the full range afterwards
    calc_scalebias_from_execution(node, model, range_dict)
    calc_intrange_from_scalebias(node, model, range_dict)
    # inherit scale/bias history from all inputs (most will be empty)
    aggr_history_scale = []
    aggr_history_bias = []
    for node_in in node.input:
        aggr_history_scale += range_dict[node_in].history_scale
        aggr_history_bias += range_dict[node_in].history_bias
    range_dict[node.output[0]].history_scale = aggr_history_scale
    range_dict[node.output[0]].history_bias = aggr_history_bias


def calc_intrange_eltwise_monotonic_intrangefirst(node, model, range_dict):
    # strategy: use regular range analysis (which will execute the node on the corners of the input
    # range) using the integer range as the input, which gives us the output integer range. then figure
    # out the scale/bias based on the output integer range.
    orange_inf = {out: range_dict[out] for out in node.output}
    int_range_dict = {}
    for node_out in node.output:
        oshape = model.get_tensor_shape(node_out)
        int_range_dict[node_out] = RangeInfo(shape=oshape)
    # use integer components of input ranges for new range computation
    for node_in in node.input:
        if not (range_dict[node_in].int_range is None):
            int_range_dict[node_in] = RangeInfo(
                shape=range_dict[node_in].shape,
                range=range_dict[node_in].int_range,
                is_initializer=range_dict[node_in].is_initializer,
            )
        else:
            # the shape-related input for Reshape, Transpose etc may give rise to this case
            int_range_dict[node_in] = range_dict[node_in]
    range_calc_fxn = optype_to_range_calc[node.op_type]
    range_calc_fxn(node, model, int_range_dict)
    for i, out in enumerate(node.output):
        int_orange_inf = int_range_dict[out]
        range_dict[out].int_range = int_orange_inf.range
        # now deduce the output scale factor and bias from all available info
        # range_max = S*int_range_max + B
        # range_min = S*int_range_min + B
        # so S = (range_max - range_min) / (int_range_max - int_range_min)
        # and afterwards, B = range_max - S*int_range_max
        # TODO scale and bias may contain NaN's when channels are stuck
        # how best to deal with this? leave as is? set to 1/0?
        # try to recover in some other way? (perturb the actual range before calling range_calc_fxn)
        scale = (orange_inf[out].range[1] - orange_inf[out].range[0]) / (int_orange_inf.range[1] - int_orange_inf.range[0])
        if not np.isfinite(scale).all():
            warn(f"{node.name} has stuck values, forcing scale to 1.0 for those")
            scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)
        bias = orange_inf[out].range[1] - scale * int_orange_inf.range[1]
        range_dict[out].scale = scale
        range_dict[out].bias = bias
    # inherit scale/bias history from all inputs (most will be empty)
    # TODO: Still needs to be extended for multiple outputs...
    aggr_history_scale = []
    aggr_history_bias = []
    for node_in in node.input:
        aggr_history_scale += range_dict[node_in].history_scale
        aggr_history_bias += range_dict[node_in].history_bias
    range_dict[node.output[0]].history_scale = aggr_history_scale
    range_dict[node.output[0]].history_bias = aggr_history_bias


# for several types of nodes, we dynamically convert ("lower") the node to something else that we can
# process before making a recursive call to scaled-int range analysis and put those results back in
def calc_intrange_with_lowering(prep_transforms, lowering_transforms, node, model, range_dict):
    # run prep transforms to ensure lowering on single node will work correctly
    prep_model = model
    for trafo in prep_transforms:
        prep_model = prep_model.transform(trafo)
    # create a single-node model from this node
    node_model = ModelWrapper(node_to_model(node, prep_model))
    # run lowering pipeline on the single-node model
    for trafo in lowering_transforms:
        node_model = node_model.transform(trafo)
    # copy RangeInfo pertaining to node_model's top-level inputs to a new dict
    node_range_dict = {}
    for node_inp in node_model.graph.input:
        node_range_dict[node_inp.name] = range_dict[node_inp.name]
    # run range analysis on the lowered single-node model
    ret_range_dict, _ = range_analysis(node_model, irange=node_range_dict, report_mode=REPORT_MODE_RANGE, scaled_int=True)
    # copy results back into original range_dict
    for node_out in node.output:
        range_dict[node_out] = ret_range_dict[node_out]


def calc_intrange_bn(node, model, range_dict):
    prep_transforms = []
    lowering_transforms = [BatchNormToAffine()]
    calc_intrange_with_lowering(prep_transforms, lowering_transforms, node, model, range_dict)


def calc_intrange_sub(node, model, range_dict):
    prep_transforms = []
    lowering_transforms = [ConvertSubToAdd()]
    calc_intrange_with_lowering(prep_transforms, lowering_transforms, node, model, range_dict)


def calc_intrange_div(node, model, range_dict):
    prep_transforms = []
    lowering_transforms = [ConvertDivToMul()]
    calc_intrange_with_lowering(prep_transforms, lowering_transforms, node, model, range_dict)


def calc_intrange_gemm(node, model, range_dict):
    prep_transforms = []
    lowering_transforms = [GemmToMatMul()]
    calc_intrange_with_lowering(prep_transforms, lowering_transforms, node, model, range_dict)


# handler functions for regular range analysis
optype_to_range_calc = {
    "Transpose": calc_monotonic_range,
    "MatMul": calc_matmul_node_range,
    "QuantMaxNorm": calc_range_outdtype,
    "Flatten": calc_monotonic_range,
    "Reshape": calc_monotonic_range,
    "Quant": calc_monotonic_range,
    "BipolarQuant": calc_monotonic_range,
    "Mul": calc_monotonic_range,
    "Sub": calc_monotonic_range,
    "Div": calc_monotonic_range,
    "Add": calc_monotonic_range,
    "BatchNormalization": calc_monotonic_range,
    "Pad": calc_monotonic_range,
    "AveragePool": calc_monotonic_range,
    "Trunc": calc_monotonic_range,
    "MaxPool": calc_monotonic_range,
    "Resize": calc_monotonic_range,
    "Upsample": calc_monotonic_range,
    "GlobalAveragePool": calc_monotonic_range,
    "QuantizeLinear": calc_monotonic_range,
    "DequantizeLinear": calc_monotonic_range,
    "Concat": calc_monotonic_range,
    "Split": calc_monotonic_range,
    "Im2Col": calc_monotonic_range,
    # Monotonic activation functions: This list is not completer yet, there are
    # some not supported/produced by export, so they are not verified and thus
    # not added here.
    "Identity": calc_monotonic_range,
    "Relu": calc_monotonic_range,
    "LeakyRelu": calc_monotonic_range,
    "Clip": calc_monotonic_range,
    "Selu": calc_monotonic_range,
    "Celu": calc_monotonic_range,
    "Elu": calc_monotonic_range,
    "Sigmoid": calc_monotonic_range,
    "HardSigmoid": calc_monotonic_range,
    "Tanh": calc_monotonic_range,
    "Softplus": calc_monotonic_range,
    "Exp": calc_monotonic_range,
    "Log": calc_monotonic_range,
    "Sqrt": calc_monotonic_range,
    "Erf": calc_monotonic_range,
    "Floor": calc_monotonic_range,
    "Ceil": calc_monotonic_range,
    "Round": calc_monotonic_range,
    "Sign": calc_monotonic_range,
    # Softmax has a defined output range of [0,1] while LogSoftmax yields the
    # log of this range
    "Softmax": calc_softmax_range,
    "LogSoftmax": calc_logsoftmax_range,
    # Squeeze and Unsqueeze are special cases of Reshape, which ist monotonic
    "Squeeze": calc_monotonic_range,
    "Unsqueeze": calc_monotonic_range,
    # Treat MultiThreshold as monotonic. This might be necessary for iterated
    # rounds of activation function to MultiThreshold conversion to absorb
    # chains of monotonic activation functions into MultiThreshold
    # TODO: Check whether this is actually ok...
    "MultiThreshold": calc_monotonic_range,
    "Conv": calc_conv_range,
    "Gemm": calc_gemm_range,
}

# handler functions for scaled-integer range analysis
optype_to_intrange_calc = {
    "MatMul": calc_intrange_matmul,
    "Conv": calc_intrange_conv,
    "Mul": calc_intrange_mul,
    "Relu": calc_intrange_relu,
    "Quant": calc_intrange_quant,
    "Pad": calc_intrange_eltwise_monotonic,
    "MaxPool": calc_intrange_eltwise_monotonic,
    "Im2Col": calc_intrange_eltwise_monotonic,
    "Concat": calc_intrange_eltwise_monotonic,
    # TODO: Workaround for some weird RA behavior producing NANs, zero scales or
    #  ranges from -0 to +0. So far only observed in rather complex topology
    #  involving residual connections, attention and novel activation functions
    #  and it is unclear how to reproduce this in isolation...
    "Add": calc_intrange_add,
    "Reshape": calc_intrange_eltwise_monotonic,
    "Transpose": calc_intrange_eltwise_monotonic,
    "Split": calc_intrange_eltwise_monotonic,
    # Treat MultiThreshold as monotonic. This might be necessary for iterated
    # rounds of activation function to MultiThreshold conversion to absorb
    # chains of monotonic activation functions into MultiThreshold
    # TODO: Check whether this is actually ok...
    "MultiThreshold": calc_intrange_eltwise_monotonic,
    "Sub": calc_intrange_sub,
    "Div": calc_intrange_div,
    "Gemm": calc_intrange_gemm,
    "BatchNormalization": calc_intrange_bn,
    "AveragePool": calc_intrange_eltwise_monotonic,
    "Trunc": calc_intrange_trunc,
}


# walk the graph node by node and propagate scaled-int range info
# assumes that regular range analysis was already carried out
def calc_intrange(model, range_dict, do_unbroadcast, stop_at_nodename):
    for node in model.graph.node:
        op_ok = node.op_type in optype_to_intrange_calc.keys()
        if op_ok:
            range_calc_fxn = optype_to_intrange_calc[node.op_type]
            range_calc_fxn(node, model, range_dict)
            for node_out in node.output:
                if do_unbroadcast:
                    range_dict[node_out] = range_dict[node_out].unbroadcast()
                else:
                    # ensure all produced ranges are per-element
                    out_vi = model.get_tensor_valueinfo(node_out)
                    range_dict[node_out].int_range = broadcast_range(range_dict[node_out].int_range, out_vi)
        else:
            warn("Skipping %s : op_ok? (%s) %s" % (node.name, node.op_type, str(op_ok)))
        if stop_at_nodename != "" and node.name == stop_at_nodename:
            break


REPORT_MODE_RANGE = "range"
REPORT_MODE_STUCKCHANNEL = "stuck_channel"
REPORT_MODE_ZEROSTUCKCHANNEL = "zerostuck_channel"

report_modes = {REPORT_MODE_RANGE, REPORT_MODE_STUCKCHANNEL, REPORT_MODE_ZEROSTUCKCHANNEL}

report_mode_options = clize.parameters.mapped(
    [
        (REPORT_MODE_RANGE, [REPORT_MODE_RANGE], "Report ranges"),
        (REPORT_MODE_STUCKCHANNEL, [REPORT_MODE_STUCKCHANNEL], "Report stuck channels"),
        (REPORT_MODE_ZEROSTUCKCHANNEL, [REPORT_MODE_ZEROSTUCKCHANNEL], "Report 0-stuck channels"),
    ]
)


def range_analysis(
    model_filename_or_wrapper,
    *,
    irange="",
    key_filter: str = "",
    save_modified_model: str = "",
    report_mode: report_mode_options = REPORT_MODE_STUCKCHANNEL,
    prettyprint=False,
    do_cleanup=False,
    strip_initializers_from_report=True,
    scaled_int=False,
    do_unbroadcast=False,
    stop_at_nodename="",
):
    assert report_mode in report_modes, "Unrecognized report_mode, must be " + str(report_modes)
    if isinstance(model_filename_or_wrapper, ModelWrapper):
        model = model_filename_or_wrapper
    else:
        model = ModelWrapper(model_filename_or_wrapper)
    if isinstance(irange, str):
        if irange == "":
            range_min = None
            range_max = None
        else:
            irange = eval(irange)
            range_min, range_max = irange
            if not isinstance(range_min, np.ndarray):
                range_min = np.asarray(range_min, dtype=np.float32)
            if not isinstance(range_max, np.ndarray):
                range_max = np.asarray(range_max, dtype=np.float32)
    elif isinstance(irange, tuple):
        range_min, range_max = irange
        if not isinstance(range_min, np.ndarray):
            range_min = np.asarray(range_min, dtype=np.float32)
        if not isinstance(range_max, np.ndarray):
            range_max = np.asarray(range_max, dtype=np.float32)
    elif isinstance(irange, RangeInfo):
        pass
    elif isinstance(irange, dict):
        pass
    else:
        assert False, "Unknown irange type"
    if do_cleanup:
        model = cleanup_model(model, preserve_qnt_ops=True)
    model = model.transform(InferDataTypes(allow_scaledint_dtypes=True))
    if save_modified_model != "":
        model.save(save_modified_model)
    range_dict = {}
    stuck_chans = {}

    if isinstance(irange, dict):
        # directly use provided range dict
        range_dict = irange
    else:
        # start by calculating/annotating range info for input tensors
        for inp in model.graph.input:
            iname = inp.name
            if isinstance(irange, RangeInfo):
                range_dict[iname] = irange
                # capture any non-1/non-0 input scale/bias as part of history
                if not (irange.scale is None) and (irange.scale != 1).all():
                    range_dict[iname].history_scale = [iname]
                if not (irange.bias is None) and (irange.bias != 0).all():
                    range_dict[iname].history_bias = [iname]
            else:
                if range_min is None or range_max is None:
                    # use idt annotation
                    idt = model.get_tensor_datatype(iname)
                    assert idt is not None, "Could not infer irange, please specify"
                    range_min = idt.min()
                    range_max = idt.max()
                ishape = model.get_tensor_shape(iname)
                range_dict[iname] = RangeInfo(shape=ishape, range=(range_min, range_max))

    # add range info for all tensors with initializers
    calc_range_all_initializers(model, range_dict)

    # cleanup input/initializer ranges before start
    if do_unbroadcast:
        range_dict = unbroadcast_range_dict(range_dict)

    # now walk the graph node by node and propagate range info
    for node in model.graph.node:
        dyn_inputs = [x for x in node.input if is_dyn_input(x, model)]
        inprange_ok = all([x in range_dict.keys() for x in dyn_inputs])
        op_ok = node.op_type in optype_to_range_calc.keys()
        if inprange_ok and op_ok:
            # create entries in range_dict with RangeInfo type for all outputs
            # since range analysis functions will be assigning to the .range member of
            # this RangeInfo directly later on
            for node_out in node.output:
                range_dict[node_out] = RangeInfo(shape=model.get_tensor_shape(node_out))
            range_calc_fxn = optype_to_range_calc[node.op_type]
            range_calc_fxn(node, model, range_dict)

            for node_out in node.output:
                if do_unbroadcast:
                    range_dict[node_out] = range_dict[node_out].unbroadcast()
                else:
                    # ensure all produced ranges are per-element
                    out_vi = model.get_tensor_valueinfo(node_out)
                    range_dict[node_out].range = broadcast_range(range_dict[node_out].range, out_vi)
            # TODO bring back stuck channel analysis after simplification is re-introduced
        else:
            warn("Skipping %s : inp_range? %s op_ok? (%s) %s" % (node.name, str(inprange_ok), node.op_type, str(op_ok)))
        if stop_at_nodename != "" and node.name == stop_at_nodename:
            break

    # if scaled-int range prop is enabled, call as postproc
    if scaled_int:
        calc_intrange(model, range_dict, do_unbroadcast, stop_at_nodename)

    # range dict is now complete, apply filters and formatting
    if report_mode in [REPORT_MODE_ZEROSTUCKCHANNEL, REPORT_MODE_STUCKCHANNEL]:
        ret = stuck_chans
    else:
        ret = range_dict
        if strip_initializers_from_report:
            # exclude all initializer ranges for reporting
            ret = {k: v for (k, v) in ret.items() if not v.is_initializer}

    # only keep tensors (keys) where filter appears in the name
    if key_filter != "":
        ret = {k: v for (k, v) in ret.items() if key_filter in k}
    # only keep tensors (keys) where filter appears in the name
    if key_filter != "":
        ret = {k: v for (k, v) in ret.items() if key_filter in k}

    if report_mode == REPORT_MODE_RANGE:
        # TODO convert ranges in report to regular Python lists for nicer printing
        pass
    elif report_mode == REPORT_MODE_ZEROSTUCKCHANNEL:
        # only leave channels that are stuck at zero
        # value info removed since implicitly 0
        new_ret = {}
        for tname, schans in ret.items():
            schans_only_zero = set([x[0] for x in schans if x[1] == 0])
            if len(schans_only_zero) > 0:
                new_ret[tname] = schans_only_zero
        ret = new_ret
    if prettyprint:
        ret = pprint.pformat(ret, sort_dicts=False)
    # Return the range information and the transformed model as we might have
    # added, removed or renamed some tensors above, and thus we need the new
    # model to match tensor names from range information.
    return ret, model


def main():
    clize.run(range_analysis)


if __name__ == "__main__":
    main()
