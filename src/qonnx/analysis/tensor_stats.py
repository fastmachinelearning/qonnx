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

import clize
import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import tqdm

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.range_analysis import (
    REPORT_MODE_RANGE,
    REPORT_MODE_STUCKCHANNEL,
    REPORT_MODE_ZEROSTUCKCHANNEL,
    report_mode_options,
    report_modes,
)


def update_tensor_stats(tensor, axes, ret_dict={}):
    shp = tensor.shape
    if ret_dict == {}:
        ret_dict["shape"] = shp
    else:
        assert ret_dict["shape"] == shp
    for axis in axes:
        tensor_new = np.moveaxis(tensor, axis, 0).reshape(shp[axis], -1)
        ret_axis = {
            "min": np.min(tensor_new, axis=1),
            "max": np.max(tensor_new, axis=1),
        }
        axis_name = "axis%d" % axis
        if axis_name in ret_dict:
            ret_dict[axis_name] = ret_axis
            ret_axis["min"] = np.minimum(ret_axis["min"], ret_dict[axis_name]["min"])
            ret_axis["max"] = np.maximum(ret_axis["max"], ret_dict[axis_name]["max"])
        ret_dict[axis_name] = ret_axis
    return ret_dict


def tensor_stats(
    modelwrapper_or_filename,
    act_dump_dir: str,
    output_stats_dir: str,
    *,
    axes="1",
    plot=True,
    report_mode: report_mode_options = REPORT_MODE_RANGE
):
    assert report_mode in report_modes, "Unknown report_mode"
    if not isinstance(modelwrapper_or_filename, ModelWrapper):
        model = ModelWrapper(modelwrapper_or_filename)
    else:
        model = modelwrapper_or_filename
    if not isinstance(axes, list):
        axes = [int(x.strip()) for x in axes.split(",")]
    if not os.path.isdir(output_stats_dir):
        os.makedirs(output_stats_dir)

    all_tensor_dump_files = []
    all_tensor_dump_files = [f for f in os.listdir(act_dump_dir) if os.path.isfile(os.path.join(act_dump_dir, f))]
    all_tensor_dump_files = [f for f in all_tensor_dump_files if f.endswith(".npy")]
    tensorwise_stats = {}
    stuck_chans = {}
    for outp in tqdm(model.graph.output, "Tensors"):
        tname = outp.name
        t_files = [f for f in all_tensor_dump_files if f.startswith(tname)]
        tensorwise_stats[tname] = {}
        for f in tqdm(t_files, "Batches"):
            t_file = np.load(os.path.join(act_dump_dir, f))
            tensorwise_stats[tname] = update_tensor_stats(t_file, axes=axes, ret_dict=tensorwise_stats[tname])
        tstats = tensorwise_stats[tname]
        for axis in axes:
            axis_name = "axis%d" % axis
            axis_data = tstats[axis_name]
            axis_min = axis_data["min"]
            axis_max = axis_data["max"]
            tensor_stuck_chans = np.nonzero(axis_min == axis_max)[0]
            if report_mode in [REPORT_MODE_STUCKCHANNEL, REPORT_MODE_ZEROSTUCKCHANNEL]:
                if len(tensor_stuck_chans) > 0:
                    list_stuck_chans = list(tensor_stuck_chans)
                    list_stuck_values = list(axis_min[tensor_stuck_chans])
                    stuck_chans[tname] = list(zip(list_stuck_chans, list_stuck_values))
        if plot:
            for axis in axes:
                axis_name = "axis%d" % axis
                data = tensorwise_stats[tname][axis_name]
                axis_min = data["min"]
                axis_max = data["max"]
                axis_range = axis_max - axis_min
                chans = [i for i in range(len(axis_min))]
                plt.clf()
                plt.figure(constrained_layout=True, figsize=(5, len(axis_min) / 3))
                bars = plt.barh(chans, axis_range, left=axis_min)
                bar_labels = [str((axis_min[i], axis_max[i])) for i in range(len(axis_min))]
                plt.bar_label(bars, bar_labels)
                plt.yticks([x for x in range(len(axis_range))])
                plt.xlabel("Channel number")
                plt.ylabel("Channel range")
                plt.title("Observed range for %s_%s" % (tname, axis_name))

                plt.savefig(output_stats_dir + "/%s_%s.png" % (tname, axis_name))

    if report_mode == REPORT_MODE_RANGE:
        ret = tensorwise_stats
    elif report_mode == REPORT_MODE_STUCKCHANNEL:
        ret = stuck_chans
    elif report_mode == REPORT_MODE_ZEROSTUCKCHANNEL:
        # only leave channels that are stuck at zero
        # value info removed since implicitly 0
        ret = stuck_chans
        new_ret = {}
        for tname, schans in ret.items():
            schans_only_zero = set([x[0] for x in schans if x[1] == 0])
            if len(schans_only_zero) > 0:
                new_ret[tname] = schans_only_zero
        ret = new_ret
    return ret


def main():
    clize.run(tensor_stats)


if __name__ == "__main__":
    main()
