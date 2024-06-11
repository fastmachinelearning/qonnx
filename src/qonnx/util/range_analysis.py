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

import dataclasses as dc
import itertools
import numpy as np

from qonnx.core.onnx_exec import execute_node
from qonnx.util.onnx import valueinfo_to_tensor

# walk the graph to deduce range information about each tensor
# assumptions:
# - layout and shape inference already completed
# - range info is generated per-element (broadcasted to this shape even if identical entries)


# RangeInfo dataclass: we will use instances of this to represent the range information for tensors
@dc.dataclass
class RangeInfo:
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


def is_dyn_input(x, model):
    # return True if a given tensor has no initializer (=dynamic), False otherwise
    return model.get_initializer(x) is None and x != ""


def promote_range_shape(tensor_range: tuple, tensor_vi):
    # ensure the range has the apropriate (per-element shape)
    # i.e. range = (range_min, range_max) where range_min and
    # range_max have the same shape as the original tensor
    proto_tensor = valueinfo_to_tensor(tensor_vi)
    tensor_shape = proto_tensor.shape
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
