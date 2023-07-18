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

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.pruning import PruneChannels
from qonnx.util.cleanup import cleanup_model
from qonnx.util.range_analysis import range_analysis


def prune_stuck_channels(input_filename_or_modelwrapper, *, irange="0,1", output_filename=""):
    if not isinstance(input_filename_or_modelwrapper, ModelWrapper):
        model = ModelWrapper(input_filename_or_modelwrapper)
    else:
        model = input_filename_or_modelwrapper
    model = cleanup_model(model, preserve_qnt_ops=True)
    model = model.transform(InferDataTypes())
    model = cleanup_model(model, preserve_qnt_ops=False)
    stuck_chans = range_analysis(model, irange=irange, key_filter="Quant")
    pruned_model = model.transform(PruneChannels(stuck_chans))
    if output_filename != "":
        pruned_model.save(output_filename)
    return pruned_model


def main():
    clize.run(prune_stuck_channels)


if __name__ == "__main__":
    main()
