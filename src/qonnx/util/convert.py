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
from qonnx.transformation.qcdq_to_qonnx import QCDQToQuant
from qonnx.transformation.qonnx_to_qcdq import QuantToQCDQ

CONVERT_MODE_QCDQ = "qcdq"
CONVERT_MODE_QUANT = "quant"

convert_modes = {CONVERT_MODE_QCDQ, CONVERT_MODE_QUANT}

convert_mode_options = clize.parameters.mapped(
    [
        (CONVERT_MODE_QCDQ, [CONVERT_MODE_QCDQ], "Convert from Quant to QCDQ"),
        (CONVERT_MODE_QUANT, [CONVERT_MODE_QUANT], "Convert from QCDQ to Quant"),
    ]
)


def convert(input_model_file, *, output_style: convert_mode_options, output_file: str = None):
    """Convert an ONNX file from one style of quantization to another, where possible.
    Please see the documentation on the QuantToQCDQ and QCDQToQuant
    transformations to learn more about the particular limitations.

    :param input_model_file: Filename for the input ONNX model.
    :param output_style: Quantization style for the output.
    :param output_file: If specified, write the output ONNX model to this filename.
        Otherwise, will default to the input file with an _output_style suffix.
    """
    model = ModelWrapper(input_model_file)
    if output_style == CONVERT_MODE_QCDQ:
        model = model.transform(QuantToQCDQ())
    elif output_style == CONVERT_MODE_QUANT:
        model = model.transform(QCDQToQuant())
    else:
        print("Unknown output_style for conversion: %s" % output_style)
        exit(-1)
    if output_file is None:
        output_file = input_model_file.replace(".onnx", "_%s.onnx" % output_style)
    model.save(output_file)


def main():
    clize.run(convert)


if __name__ == "__main__":
    main()
