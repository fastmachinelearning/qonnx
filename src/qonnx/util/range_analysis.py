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
import numpy as np
from warnings import warn

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_node
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import get_by_name
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


def propagate_range(node, model, range_dict):
    iname = node.input[0]
    node_irange = range_dict[iname]
    for oname in node.output:
        range_dict[oname] = node_irange


def calc_matmul_range(node, model, range_dict):
    iname = node.input[0]
    wname = node.input[1]
    oname = node.output[0]
    irange = range_dict[iname]
    imin, imax = irange
    weights = model.get_initializer(wname)
    assert weights is not None, "Uninitialized MatMul weights"
    is_depthwise = False
    if node.op_type == "Conv":
        # do weight reshaping to treat Conv as MatMul
        # (mh, mw) = (ofm, (ifm x k0 x k1 x ...))
        conv_ofm = weights.shape[0]
        weights = weights.reshape(conv_ofm, -1)
        groups = get_by_name(node.attribute, "group").i
        # TODO smarter check, other kinds of grouped convs out there..
        is_depthwise = groups > 1
    if node.op_type == "MatMul":
        # util function expects (mh, mw) so transpose
        weights = weights.transpose()
    if is_depthwise:
        # need to construct specialzed input range vectors
        dw_ret_min = []
        dw_ret_max = []
        for i in range(conv_ofm):
            mw = weights.shape[1]
            w_slice = weights[i, :].reshape(1, mw)
            if type(imin) is np.ndarray:
                imin_rep = np.repeat(imin[i], mw)
                imax_rep = np.repeat(imax[i], mw)
            else:
                imin_rep = imin
                imax_rep = imax
            dw_ret = calculate_matvec_accumulator_extremum(w_slice, imin_rep, imax_rep)
            dw_ret_min.append(dw_ret[0].item())
            dw_ret_max.append(dw_ret[1].item())
        ret = (np.asarray(dw_ret_min), np.asarray(dw_ret_max))
    else:
        if type(imin) is np.ndarray:
            assert len(imin) == weights.shape[1], "Dot product length mismatch, np broadcast may be wrong"
        ret = calculate_matvec_accumulator_extremum(weights, imin, imax)
    range_dict[oname] = ret


def calc_monotonic_range(node, model, range_dict, i_channel_axis=1):
    iname = node.input[0]
    oname = node.output[0]
    inp_vi = model.get_tensor_valueinfo(iname)
    const_inps = [model.get_initializer(x) for x in node.input]
    assert const_inps[0] is None
    oname = node.output[0]
    # create prototype min and max vectors
    irange = range_dict[iname]
    ishp = model.get_tensor_shape(iname)
    proto_min = valueinfo_to_tensor(inp_vi)
    proto_max = valueinfo_to_tensor(inp_vi)
    if type(irange[0]) in [float, int, np.float32]:
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
    ctx = {x: model.get_initializer(x) for x in node.input}
    ctx[iname] = proto_min
    ctx[oname] = valueinfo_to_tensor(model.get_tensor_valueinfo(oname))
    execute_node(node, ctx, model.graph)
    o_0 = ctx[oname]
    ctx[iname] = proto_max
    execute_node(node, ctx, model.graph)
    o_1 = ctx[oname]
    axes_to_min = [i for i in range(o_0.ndim)]
    axes_to_min.remove(i_channel_axis)
    axes_to_min = tuple(axes_to_min)
    o0_min = o_0.min(axis=axes_to_min)
    o0_max = o_0.max(axis=axes_to_min)
    o1_min = o_1.min(axis=axes_to_min)
    o1_max = o_1.max(axis=axes_to_min)
    o_min = np.minimum(o0_min, o1_min).flatten()
    o_max = np.maximum(o0_max, o1_max).flatten()
    range_dict[oname] = (o_min, o_max)


def calc_eltwiseadd_range(node, model, range_dict):
    i0range = range_dict[node.input[0]]
    i1range = range_dict[node.input[1]]
    cands = []
    if type(i0range) is tuple and type(i1range) is tuple:
        for i0 in range(2):
            for i1 in range(2):
                cands.append(i0range[i0] + i1range[i1])
        newmin = min(cands)
        newmax = max(cands)
        ret = (newmin, newmax)
    elif type(i0range) is list and type(i1range) is list:
        assert len(i0range) == len(i1range), "Range list shape mismatch"
        ret = []
        for i in range(len(i1range)):
            for i0 in range(2):
                for i1 in range(2):
                    cands.append(i0range[i0] + i1range[i1])
            newmin = min(cands)
            newmax = max(cands)
            ret.append((newmin, newmax))
    range_dict[node.output[0]] = ret


def calc_range_outdtype(node, model, range_dict):
    oname = node.output[0]
    odt = model.get_tensor_datatype(oname)
    assert odt is not None, "Cannot infer %s range, dtype annotation is missing" % oname
    range_dict[oname] = (odt.min(), odt.max())


optype_to_range_calc = {
    "Transpose": propagate_range,
    "Im2Col": propagate_range,
    "MatMul": calc_matmul_range,
    "Conv": calc_matmul_range,
    "QuantMaxNorm": calc_range_outdtype,
    "Flatten": propagate_range,
    "Reshape": propagate_range,
    "Quant": calc_monotonic_range,
    "Mul": calc_monotonic_range,
    "Sub": calc_monotonic_range,
    "Div": calc_monotonic_range,
    "Add": calc_monotonic_range,
    "BatchNormalization": calc_monotonic_range,
    "Relu": calc_monotonic_range,
    "Pad": propagate_range,
    "AveragePool": calc_monotonic_range,
    "Trunc": calc_range_outdtype,
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


report_modes = {"range", "only_stuck_channel", "only_zerostuck_channel"}

REPORT_MODE_RANGE = "range"
REPORT_MODE_STUCKCHANNEL = "stuck_channel"
REPORT_MODE_ZEROSTUCKCHANNEL = "zerostuck_channel"

report_mode_options = clize.parameters.mapped(
    [
        (REPORT_MODE_RANGE, [REPORT_MODE_RANGE], "Report ranges"),
        (REPORT_MODE_STUCKCHANNEL, [REPORT_MODE_STUCKCHANNEL], "Report stuck channels"),
        (REPORT_MODE_ZEROSTUCKCHANNEL, [REPORT_MODE_ZEROSTUCKCHANNEL], "Report 0-stuck channels"),
    ]
)


def range_analysis(
    onnx_path: str,
    *,
    irange: str = "",
    key_filter: str = "",
    report_mode: report_mode_options = REPORT_MODE_ZEROSTUCKCHANNEL
):
    model = ModelWrapper(onnx_path)
    model = model.transform(InferDataTypes())
    range_dict = {}
    stuck_chans = {}

    # start by calculating range info for input tensors
    for inp in model.graph.input:
        iname = inp.name
        if irange == "":
            idt = model.get_tensor_datatype(iname)
            range_min = idt.min()
            range_max = idt.max()
        else:
            irange = irange.split(",")
            range_min, range_max = float(irange[0]), float(irange[1])
        range_dict[iname] = (range_min, range_max)

    for node in model.graph.node:
        inprange_exists = node.input[0] in range_dict.keys()
        op_ok = node.op_type in optype_to_range_calc.keys()
        if inprange_exists and op_ok:
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
            warn("Skipping %s : inp_range? %s op_ok? (%s) %s" % (node.name, str(inprange_exists), node.op_type, str(op_ok)))

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
    return ret


def main():
    clize.run(range_analysis)


if __name__ == "__main__":
    main()
