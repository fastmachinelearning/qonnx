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

import clize
import itertools
import numpy as np
import pprint
from warnings import warn

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_node
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name
from qonnx.util.cleanup import cleanup_model
from qonnx.util.onnx import valueinfo_to_tensor

# walk the graph to deduce range information about each tensor
# assumptions:
# - layout and shape inference already completed
# - any quantized weights are resolved into initializers
# - range info is generated per-channel (tuple of 1D arrays) or per-tensor (tuple of scalars)


def calculate_matvec_accumulator_extremum(matrix: np.ndarray, vec_min, vec_max):
    """Calculate the minimum and maximum possible result (accumulator) values for a dot product A*x,
    given matrix A of dims (MH, MW), and vector (MW) with range (vec_min, vec_max). vec_min and
    vec_max are either scalars, or 1D arrays of length MW.
    Returns (acc_min, acc_max) where acc_min and acc_max are 1D arrays of length MH."""
    max_vectors = np.where(matrix > 0, vec_max, vec_min)
    min_vectors = np.where(matrix > 0, vec_min, vec_max)
    max_values = (matrix * max_vectors).sum(axis=1)
    min_values = (matrix * min_vectors).sum(axis=1)
    return (min_values, max_values)


def calc_gemm_range(node, model, range_dict):
    alpha = get_by_name(node.attribute, "alpha").f
    beta = get_by_name(node.attribute, "beta").f
    transA = get_by_name(node.attribute, "transA")
    if transA is not None:
        transA = transA.i
    else:
        transA = 0
    transB = get_by_name(node.attribute, "transB")
    if transB is not None:
        transB = transB.i
    else:
        transB = 1
    assert (not transA) and transB
    iname = node.input[0]
    wname = node.input[1]
    bname = None
    if len(node.input) > 2:
        bname = node.input[2]
    oname = node.output[0]

    irange = range_dict[iname]
    imin, imax = irange
    weights = model.get_initializer(wname)
    assert weights is not None, "Uninitialized Gemm weights"
    if type(imin) is np.ndarray:
        assert len(imin) == weights.shape[1], "Dot product length mismatch, np broadcast may be wrong"
    pmin, pmax = calculate_matvec_accumulator_extremum(weights, imin, imax)
    # apply Gemm scale factors to matrix multiply output
    pmin *= alpha
    pmax *= alpha
    # if there is a bias, apply it to the range
    if bname is not None:
        bias = model.get_initializer(bname)
        assert bias is not None, "Uninitialized Gemm bias"
        pmin += beta * bias
        pmax += beta * bias
    ret = (pmin, pmax)
    range_dict[oname] = ret


def calc_matmul_range(node, model, range_dict):
    iname = node.input[0]
    wname = node.input[1]
    oname = node.output[0]
    irange = range_dict[iname]
    imin, imax = irange
    weights = model.get_initializer(wname)
    assert weights is not None, "Uninitialized MatMul weights"
    # util function expects (mh, mw) so transpose
    weights = weights.transpose()
    if type(imin) is np.ndarray:
        assert len(imin) == weights.shape[1], "Dot product length mismatch, np broadcast may be wrong"
    ret = calculate_matvec_accumulator_extremum(weights, imin, imax)
    range_dict[oname] = ret


def calc_conv_range(node, model, range_dict):
    iname = node.input[0]
    wname = node.input[1]
    assert len(node.input) == 2, "Found unsupported Conv with bias"
    oname = node.output[0]
    irange = range_dict[iname]
    imin, imax = irange
    weights = model.get_initializer(wname)
    assert weights is not None, "Uninitialized Conv weights"
    # do weight reshaping to treat Conv similar to MatMul
    # (mh, mw) = (ofm, (ifm x k0 x k1 x ...))
    conv_ofm = weights.shape[0]
    conv_ifm = weights.shape[1]
    weights = weights.reshape(conv_ofm, -1)
    k_total = weights.shape[1] // conv_ifm
    groups = get_by_name(node.attribute, "group")
    if groups is None:
        # default to dense convs
        groups = 1
    else:
        groups = groups.i
    # TODO smarter check, other kinds of grouped convs out there..
    is_depthwise = groups > 1
    # need to construct specialzed input range vectors for Conv
    if is_depthwise:
        conv_ifm = conv_ofm
    if type(imin) is np.ndarray:
        imin_rep = np.repeat(imin, k_total)
        imax_rep = np.repeat(imax, k_total)
    else:
        imin_rep = imin
        imax_rep = imax
    dw_ret_min = []
    dw_ret_max = []
    for i in range(conv_ofm):
        w_slice = weights[i, :].reshape(1, -1)
        if is_depthwise and type(imin_rep) is np.ndarray:
            dw_ret = calculate_matvec_accumulator_extremum(
                w_slice, imin_rep[i * k_total : (i + 1) * k_total], imax_rep[i * k_total : (i + 1) * k_total]
            )
        else:
            dw_ret = calculate_matvec_accumulator_extremum(w_slice, imin_rep, imax_rep)
        dw_ret_min.append(dw_ret[0].item())
        dw_ret_max.append(dw_ret[1].item())
    ret = (np.asarray(dw_ret_min), np.asarray(dw_ret_max))
    range_dict[oname] = ret


def calc_convtranspose_range(node, model, range_dict):
    iname = node.input[0]
    wname = node.input[1]
    assert len(node.input) == 2, "Found unsupported ConvTranspose with bias"
    oname = node.output[0]
    irange = range_dict[iname]
    imin, imax = irange
    weights = model.get_initializer(wname)
    assert weights is not None, "Uninitialized ConvTranspose weights"
    groups = get_by_name(node.attribute, "group")
    if groups is None:
        # default to dense convs
        groups = 1
    else:
        groups = groups.i
    assert groups == 1, "Only dense (non-grouped) ConvTranspose is supported"
    # do weight reshaping to treat Conv similar to MatMul
    # (mh, mw) = (ofm, (ifm x k0 x k1 x ...))
    conv_ofm = weights.shape[1]
    conv_ifm = weights.shape[0]
    weights = weights.transpose(1, 0, 2, 3).reshape(conv_ofm, -1)
    k_total = weights.shape[1] // conv_ifm
    if type(imin) is np.ndarray:
        imin_rep = np.repeat(imin, k_total)
        imax_rep = np.repeat(imax, k_total)
    else:
        imin_rep = imin
        imax_rep = imax
    dw_ret_min = []
    dw_ret_max = []
    for i in range(conv_ofm):
        w_slice = weights[i, :].reshape(1, -1)
        dw_ret = calculate_matvec_accumulator_extremum(w_slice, imin_rep, imax_rep)
        dw_ret_min.append(dw_ret[0].item())
        dw_ret_max.append(dw_ret[1].item())
    ret = (np.asarray(dw_ret_min), np.asarray(dw_ret_max))
    range_dict[oname] = ret


def get_minmax_prototype_tensors(irange, ishp, inp_vi, i_channel_axis=1):
    proto_min = valueinfo_to_tensor(inp_vi)
    proto_max = valueinfo_to_tensor(inp_vi)
    if type(irange[0]) in [float, int, np.float16, np.float32, np.float64, np.uint8, np.int8]:
        imin, imax = irange
        proto_min[...] = imin
        proto_max[...] = imax
    elif type(irange[0]) is np.ndarray:
        # irange is [(min_ch0, max_ch0), (min_ch1, max_ch1) ...]
        n_ch = ishp[i_channel_axis]
        proto_min = np.moveaxis(proto_min, i_channel_axis, 0)
        proto_max = np.moveaxis(proto_max, i_channel_axis, 0)
        for ch in range(n_ch):
            proto_min[ch, ...] = irange[0][ch]
            proto_max[ch, ...] = irange[1][ch]
        proto_min = np.moveaxis(proto_min, 0, i_channel_axis)
        proto_max = np.moveaxis(proto_max, 0, i_channel_axis)
    else:
        assert False, "Unknown range type"
    return (proto_min, proto_max)


def is_dyn_input(x, model):
    return model.get_initializer(x) is None and x != ""


def calc_monotonic_range(node, model, range_dict, i_channel_axis=1):
    opset_version = model.model.opset_import[0].version
    oname = node.output[0]
    dyn_inps = [x for x in node.input if is_dyn_input(x, model)]
    n_dyn_inp = len(dyn_inps)
    proto_vectors = []
    # generate min-max prototype vectors for each dynamic input
    for inp in dyn_inps:
        irange = range_dict[inp]
        ishp = model.get_tensor_shape(inp)
        inp_vi = model.get_tensor_valueinfo(inp)
        proto_vectors.append(get_minmax_prototype_tensors(irange, ishp, inp_vi, i_channel_axis))
    # process all combinations of prototype vectors for dynamic inputs
    running_min = [None for i in range(len(node.output))]
    running_max = [None for i in range(len(node.output))]
    # create context for single-node execution
    ctx = {x: model.get_initializer(x) for x in node.input}
    for oname in node.output:
        ctx[oname] = valueinfo_to_tensor(model.get_tensor_valueinfo(oname))
    # assume all outputs are homogenous wrt data layout (e.g. channel axis
    # always lives in the same position)
    axes_to_min = [i for i in range(ctx[oname].ndim)]
    axes_to_min.remove(i_channel_axis)
    axes_to_min = tuple(axes_to_min)
    for inps in itertools.product(*proto_vectors):
        for i in range(n_dyn_inp):
            ctx[dyn_inps[i]] = inps[i]
        execute_node(node, ctx, model.graph, opset_version=opset_version)
        for oind, oname in enumerate(node.output):
            # grab new output and update running min/max
            out = ctx[oname]
            chanwise_min = out.min(axis=axes_to_min).flatten()
            chanwise_max = out.max(axis=axes_to_min).flatten()
            running_min[oind] = (
                np.minimum(chanwise_min, running_min[oind]).flatten() if running_min[oind] is not None else chanwise_min
            )
            running_max[oind] = (
                np.maximum(chanwise_max, running_max[oind]).flatten() if running_max[oind] is not None else chanwise_max
            )
    for oind, oname in enumerate(node.output):
        range_dict[oname] = (running_min[oind], running_max[oind])


def calc_range_outdtype(node, model, range_dict):
    oname = node.output[0]
    odt = model.get_tensor_datatype(oname)
    assert odt is not None, "Cannot infer %s range, dtype annotation is missing" % oname
    range_dict[oname] = (odt.min(), odt.max())


optype_to_range_calc = {
    "Transpose": calc_monotonic_range,
    "MatMul": calc_matmul_range,
    "Conv": calc_conv_range,
    "ConvTranspose": calc_convtranspose_range,
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
    "Trunc": calc_range_outdtype,
    "MaxPool": calc_monotonic_range,
    "Resize": calc_monotonic_range,
    "Upsample": calc_monotonic_range,
    "GlobalAveragePool": calc_monotonic_range,
    "Gemm": calc_gemm_range,
    "QuantizeLinear": calc_monotonic_range,
    "DequantizeLinear": calc_monotonic_range,
    "Clip": calc_monotonic_range,
    "Sigmoid": calc_monotonic_range,
    "Concat": calc_monotonic_range,
    "Split": calc_monotonic_range,
}


def simplify_range(range):
    """Where possible, simplify a range that is expressed as channelwise ranges
    back to a scalar range if all channels' ranges were equal."""
    rmin = range[0]
    rmax = range[1]
    if type(rmin) is np.ndarray and type(rmax) is np.ndarray:
        rmin_eq = all(rmin == rmin[0])
        rmax_eq = all(rmax == rmax[0])
        if rmin_eq and rmax_eq:
            return (rmin[0], rmax[0])
        else:
            return range
    else:
        return range


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
    report_mode: report_mode_options = REPORT_MODE_STUCKCHANNEL,
    prettyprint=False,
    do_cleanup=False
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
    else:
        assert False, "Unknown irange type"
    if do_cleanup:
        model = cleanup_model(model)
    # call constant folding with no exclusions to get weight initializers
    # (but not full cleanup, in order to preserve node/tensor naming)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants(exclude_op_types=[]))
    model = model.transform(InferDataTypes())
    range_dict = {}
    stuck_chans = {}

    # start by calculating/annotating range info for input tensors
    for inp in model.graph.input:
        iname = inp.name
        if range_min is None or range_max is None:
            # use idt annotation
            idt = model.get_tensor_datatype(iname)
            assert idt is not None, "Could not infer irange, please specify"
            range_min = idt.min()
            range_max = idt.max()
        range_dict[iname] = (range_min, range_max)

    for node in model.graph.node:
        dyn_inputs = [x for x in node.input if is_dyn_input(x, model)]
        inprange_ok = all([x in range_dict.keys() for x in dyn_inputs])
        op_ok = node.op_type in optype_to_range_calc.keys()
        if inprange_ok and op_ok:
            range_calc_fxn = optype_to_range_calc[node.op_type]
            range_calc_fxn(node, model, range_dict)
            out_range = range_dict[node.output[0]]
            tensor_stuck_chans = np.nonzero(out_range[0] == out_range[1])[0]
            if len(tensor_stuck_chans) > 0:
                list_stuck_chans = list(tensor_stuck_chans)
                list_stuck_values = list(out_range[0][tensor_stuck_chans])
                stuck_chans[node.output[0]] = list(zip(list_stuck_chans, list_stuck_values))
            range_dict[node.output[0]] = simplify_range(out_range)
        else:
            warn("Skipping %s : inp_range? %s op_ok? (%s) %s" % (node.name, str(inprange_ok), node.op_type, str(op_ok)))

    # range dict is now complete, apply filters and formatting
    if report_mode in [REPORT_MODE_ZEROSTUCKCHANNEL, REPORT_MODE_STUCKCHANNEL]:
        ret = stuck_chans
    else:
        ret = range_dict
    # only keep tensors (keys) where filter appears in the name
    if key_filter != "":
        ret = {k: v for (k, v) in ret.items() if key_filter in k}

    if report_mode == REPORT_MODE_RANGE:
        # convert ranges in report to regular Python lists
        for tname, trange in ret.items():
            if type(trange[0]) is np.ndarray:
                ret[tname] = (list(trange[0]), list(trange[1]))
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
