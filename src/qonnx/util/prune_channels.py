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
from qonnx.transformation.pruning import PruneChannels


def prune_channels(input_filename, prunespec_filename, *, lossy=True, output_filename=""):
    """
    Prune channels from specified tensors and their dependencies from a model.
    The model must have already been cleaned up by qonnx-cleanup, including the
    --extract-conv-bias=True --preserve-qnt-ops=False options.

    :param input_filename: Filename for the input ONNX model
    :param prunespec_filename: Filename for the pruning specification, formatted as a Python dict
        formatted as {tensor_name : {axis : {channels}}}. See test_pruning.py for examples.
    :param lossy: Whether to perform lossy pruning, see the PruneChannels transformation for description.
    :param output_filename: If specified, write the resulting pruned model to this filename. Otherwise,
        the input_filename will be used with a _pruned suffix.
    """
    model = ModelWrapper(input_filename)
    with open(prunespec_filename) as f:
        prunespec_dict = dict(eval(f.read()))
    pruned_model = model.transform(PruneChannels(prunespec_dict, lossy))
    if output_filename == "":
        output_filename = input_filename.replace(".onnx", "_pruned.onnx")
    pruned_model.save(output_filename)


def main():
    clize.run(prune_channels)


if __name__ == "__main__":
    main()
