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

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_node
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from qonnx.transformation.general import ConvertSubToAdd
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.util.cleanup import cleanup_model
from qonnx.util.onnx import valueinfo_to_tensor

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


def is_dyn_input(x, model):
    # return True if a given tensor has no initializer (=dynamic), False otherwise
    return model.get_initializer(x) is None and x != ""


def promote_range_shape(tensor_range: tuple, tensor_vi_or_shape):
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
    if isinstance(tensor_range[0], np.ndarray) and tensor_range[0].shape == tensor_shape:
        return tensor_range
    else:
        # fix shape using numpy broadcasting
        range_min = tensor_range[0] + np.zeros_like(proto_tensor)
        range_max = tensor_range[1] + np.zeros_like(proto_tensor)
        return (range_min, range_max)


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
        proto_vectors.append(promote_range_shape(irange, inp_vi))
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
    range_A = range_dict[node.input[0]].range
    range_B = range_dict[node.input[1]].range
    range_dict[node.output[0]].range = calc_matmul_range(range_A, range_B)


# use inferred output datatype to calculate output ranges
def calc_range_outdtype(node, model, range_dict):
    oname = node.output[0]
    odt = model.get_tensor_datatype(oname)
    assert odt is not None, "Cannot infer %s range, dtype annotation is missing" % oname
    range_dict[oname].range = (odt.min(), odt.max())


# use initializers to mark point ranges i.e. tensor with initializer X has range (X, X)
def calc_range_all_initializers(model, range_dict):
    all_tensor_names = model.get_all_tensor_names()
    for tensor_name in all_tensor_names:
        tensor_init = model.get_initializer(tensor_name)
        if tensor_init is not None:
            range_dict[tensor_name] = RangeInfo(
                shape=tensor_init.shape, range=(tensor_init, tensor_init), is_initializer=True
            )
            # use % 1 == 0 to identify initializers with integer values
            if ((tensor_init % 1) == 0).all():
                range_dict[tensor_name].int_range = (tensor_init, tensor_init)
                range_dict[tensor_name].scale = np.asarray([1.0], dtype=np.float32)
                range_dict[tensor_name].bias = np.asarray([0.0], dtype=np.float32)


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
    irange_inf = range_dict[node.input[0]]
    # we'll use the ReLU output range to infer the integer parts
    # * output range can only come from the ReLU identity part (input > 0)
    # * scale and bias are always left unchanged, unless stuck channel
    # range_max = S*int_range_max + B
    # range_min = S*int_range_min + B
    # S and B are identical between input and output
    range_dict[node.output[0]].scale = irange_inf.scale
    range_dict[node.output[0]].bias = irange_inf.bias
    calc_intrange_from_scalebias(node, model, range_dict)


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


def is_point_interval(range_inf):
    return (range_inf.range[0] == range_inf.range[1]).all()


def interval_prod(interval_a, interval_b):
    a_min, a_max = interval_a
    b_min, b_max = interval_b
    c0, c1, c2, c3 = a_min * b_min, a_min * b_max, a_max * b_min, a_max * b_max
    c_min = np.minimum(c3, np.minimum(c2, np.minimum(c1, c0)))
    c_max = np.maximum(c3, np.maximum(c2, np.maximum(c1, c0)))
    return (c_min, c_max)


def calc_intrange_add(node, model, range_dict):
    # our ability to do scaled-int range propagation depends on whether all
    # inputs have an integer component already
    inp_int_info = check_int_inputs(node, range_dict)
    if not any(inp_int_info):
        # must have at least one input with integer info, otherwise no point
        warn(node.name + " has no integer info on inputs, cannot propagate")
        return
    elif all(inp_int_info):
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
            return
    # mix of integer and non-integer operands
    # [f_min, f_max] + s*[i_min, i_max] + b
    # = s*[i_min, i_max] + [f_min+b, f_max+b]
    # we can only handle this when the non-integer operand is a point interval
    # f_min = f_max = f which simplifies the expression to:
    #  = s*[i_min, i_max] + f + b
    if all(inp_int_info):
        # TODO treat whichever one is point interval as the nonint
        irange_int = range_dict[node.input[0]]
        irange_nonint = range_dict[node.input[1]]
    else:
        irange_nonint = range_dict[node.input[inp_int_info.index(False)]]
        if not is_point_interval(irange_nonint):
            warn(node.name + " has non-int input which is not point interval, cannot propagate")
            return
        irange_int = range_dict[node.input[inp_int_info.index(True)]]
    range_dict[node.output[0]].int_range = irange_int.int_range
    range_dict[node.output[0]].scale = irange_int.scale
    range_dict[node.output[0]].bias = irange_int.bias + irange_nonint.range[0]


def calc_intrange_mul(node, model, range_dict):
    # our ability to do scaled-int range propagation depends on whether all
    # inputs have an integer component already
    inp_int_info = check_int_inputs(node, range_dict)
    if not any(inp_int_info):
        # must have at least one input with integer info, otherwise no point
        warn(node.name + " has no integer info on inputs, cannot propagate")
        return
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
            range_dict[node.output[0]].bias = 0
            range_dict[node.output[0]].int_range = interval_prod(irange_0.int_range, irange_1.int_range)
            return
        else:
            warn("Found multiplication of nonzero-bias ints, cannot propagate")
            # TODO could potentially treat this as a mix of int&nonint but
            # need to identify which input to treat as nonint
            return
    # mix of integer and non-integer operands
    # [f_min, f_max] * (s*[i_min, i_max] + b)
    # = s*[i_min, i_max]*[f_min, f_max] + b*[f_min, f_max]
    # we can only handle this when the non-integer operand is a point interval
    # f_min = f_max = f which simplifies the expression to:
    #  = f*s*[i_min, i_max] + b*f
    irange_nonint = range_dict[node.input[inp_int_info.index(False)]]
    if not is_point_interval(irange_nonint):
        warn(node.name + " has non-int input which is not point interval, cannot propagate")
        return
    irange_int = range_dict[node.input[inp_int_info.index(True)]]
    range_dict[node.output[0]].int_range = irange_int.int_range
    range_dict[node.output[0]].scale = irange_int.scale * irange_nonint.range[0]
    range_dict[node.output[0]].bias = irange_int.bias * irange_nonint.range[0]


def calc_intrange_matmul(node, model, range_dict):
    inp_int_info = check_int_inputs(node, range_dict)
    if not all(inp_int_info):
        warn(node.name + " does not have all-integer inputs, cannot propagate")
        return
    for node_in in node.input:
        irange_inf = range_dict[node_in]
        # be extra conservative for now: no negative scales, no biases
        assert (irange_inf.scale >= 0).all(), "Need nonnegative scale for inputs"
        assert (irange_inf.bias == 0).all(), "Need zero bias for weights"
    # TODO need to do an extra check - scales must be at most channelwise for both sides,
    # can't be elementwise. otherwise we cannot do scaled-int range analysis
    orange_inf = range_dict[node.output[0]]
    int_range_dict = {}
    for node_out in node.output:
        int_range_dict[node_out] = RangeInfo()
    # use integer components of input ranges for new range computation
    for node_in in node.input:
        int_range_dict[node_in] = RangeInfo(
            shape=range_dict[node_in].shape,
            range=range_dict[node_in].int_range,
            is_initializer=range_dict[node_in].is_initializer,
        )
    range_calc_fxn = optype_to_range_calc[node.op_type]
    range_calc_fxn(node, model, int_range_dict)
    int_orange_inf = int_range_dict[node.output[0]]
    # now deduce the output scale factor and bias from all available info
    # range_max = S*int_range_max + B
    # range_min = S*int_range_min + B
    # so S = (range_max - range_min) / (int_range_max - int_range_min)
    # and afterwards, B = range_max - S*int_range_max
    # TODO scale and bias may contain NaN's when channels are stuck
    # how best to deal with this? leave as is? set to 1/0?
    # try to recover in some other way? (perturb the actual range before calling range_calc_fxn)
    scale = (orange_inf.range[1] - orange_inf.range[0]) / (int_orange_inf.range[1] - int_orange_inf.range[0])
    bias = orange_inf.range[1] - scale * int_orange_inf.range[1]
    range_dict[node.output[0]].scale = scale
    range_dict[node.output[0]].bias = bias
    range_dict[node.output[0]].int_range = int_orange_inf.range


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
    "Relu": calc_monotonic_range,
    "Pad": calc_monotonic_range,
    "AveragePool": calc_monotonic_range,
    "Trunc": calc_monotonic_range,
    "MaxPool": calc_monotonic_range,
    "Resize": calc_monotonic_range,
    "Upsample": calc_monotonic_range,
    "GlobalAveragePool": calc_monotonic_range,
    "QuantizeLinear": calc_monotonic_range,
    "DequantizeLinear": calc_monotonic_range,
    "Clip": calc_monotonic_range,
    "Sigmoid": calc_monotonic_range,
    "Concat": calc_monotonic_range,
    "Split": calc_monotonic_range,
    "Im2Col": calc_monotonic_range,
}

# handler functions for scaled-integer range analysis
optype_to_intrange_calc = {
    "MatMul": calc_intrange_matmul,
    "Add": calc_intrange_add,
    "Mul": calc_intrange_mul,
    "Relu": calc_intrange_relu,
    "Quant": calc_intrange_quant,
    "Pad": calc_intrange_identity,
    "MaxPool": calc_intrange_identity,
    "Reshape": calc_intrange_identity,
}


# walk the graph node by node and propagate scaled-int range info
# assumes that regular range analysis was already carried out
def calc_intrange(model, range_dict, do_unbroadcast):
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
                    range_dict[node_out].int_range = promote_range_shape(range_dict[node_out].int_range, out_vi)
        else:
            warn("Skipping %s : op_ok? (%s) %s" % (node.name, node.op_type, str(op_ok)))


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
    lower_ops=False,
    prettyprint=False,
    do_cleanup=False,
    strip_initializers_from_report=True,
    scaled_int=False,
    do_unbroadcast=False
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
            if isinstance(range_min, list):
                range_min = np.asarray(range_min, dtype=np.float32)
            if isinstance(range_max, list):
                range_max = np.asarray(range_max, dtype=np.float32)
    elif isinstance(irange, tuple):
        range_min, range_max = irange
    elif isinstance(irange, RangeInfo):
        pass
    else:
        assert False, "Unknown irange type"
    if do_cleanup:
        model = cleanup_model(model, preserve_qnt_ops=False)
    if lower_ops:
        model = model.transform(LowerConvsToMatMul())
        model = model.transform(GemmToMatMul())
        model = model.transform(BatchNormToAffine())
        model = model.transform(ConvertSubToAdd())
        model = cleanup_model(model)
    # call constant folding & shape inference, this preserves weight quantizers
    # (but do not do extra full cleanup, in order to preserve node/tensor naming)
    # TODO is this redundant? remove?
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(InferDataTypes())
    if save_modified_model != "":
        model.save(save_modified_model)
    range_dict = {}
    stuck_chans = {}

    # start by calculating/annotating range info for input tensors
    for inp in model.graph.input:
        iname = inp.name
        if isinstance(irange, RangeInfo):
            range_dict[iname] = irange
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
                    range_dict[node_out].range = promote_range_shape(range_dict[node_out].range, out_vi)
            # TODO bring back stuck channel analysis after simplification is re-introduced
        else:
            warn("Skipping %s : inp_range? %s op_ok? (%s) %s" % (node.name, str(inprange_ok), node.op_type, str(op_ok)))

    # if scaled-int range prop is enabled, call as postproc
    if scaled_int:
        calc_intrange(model, range_dict, do_unbroadcast)

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
    return ret


def main():
    clize.run(range_analysis)


if __name__ == "__main__":
    main()
