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
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean


def to_channels_last(in_file, *, make_input_channels_last=False, out_file=None):
    """Execute a set of graph transformations to convert an ONNX file to the channels last data format.
    The input file have been previously cleaned by the cleanup transformation or commandline tool.

    :param in_file: Filename for the input ONNX model
    :param make_input_channels_last: Sets if the input of the model should also be converted to the channels
        last data layout (True) or if a transpose node should be left at the beginning of the graph (False).
        Defaults to False.
    :param out_file: If set, filename for the output ONNX model. Set to in_file with _chan_last
        suffix otherwise.
    """

    # Execute transformation
    model = ModelWrapper(in_file)
    model = model.transform(ConvertToChannelsLastAndClean(make_input_channels_last=make_input_channels_last))
    if out_file is None:
        out_file = in_file.replace(".onnx", "_channels_last.onnx")
    model.save(out_file)


def main():
    clize.run(to_channels_last)


if __name__ == "__main__":
    main()
