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
from qonnx.transformation.change_batchsize import ChangeBatchSize
from qonnx.transformation.extract_conv_bias import ExtractBiasFromConv
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
)
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit


def cleanup_model(model, preserve_qnt_ops=True, override_inpsize=None, extract_conv_bias=False):
    """Execute the transformations for the cleanup function on a model level.
    This allows the reuse of the cleanup transformations, without needing to read/write the model from/to disk.

    :param model: A raw QONNX model from as example Brevitas.
    :return model_clean: The cleaned model
    """

    # temporary fix for QONNX op domains
    qonnx_domain_ops = ["Quant", "Trunc", "BipolarQuant"]
    for q_op_type in qonnx_domain_ops:
        qnt_nodes = model.get_nodes_by_op_type(q_op_type)
        for qnt_node in qnt_nodes:
            qnt_node.domain = "qonnx.custom_op.general"
    if preserve_qnt_ops:
        preserve_qnt_optypes = ["Quant", "BipolarQuant", "QuantizeLinear", "DequantizeLinear"]
    else:
        preserve_qnt_optypes = []

    if override_inpsize is not None:
        if type(override_inpsize) is str:
            override_inpsize = eval(override_inpsize)
        if type(override_inpsize) is int:
            override_batchsize = override_inpsize
            model = model.transform(ChangeBatchSize(override_batchsize))
        elif type(override_inpsize) is tuple:
            override_batchsize = override_inpsize[0]
            model = model.transform(ChangeBatchSize(override_batchsize))
            iname = model.graph.input[0].name
            model.set_tensor_shape(iname, override_inpsize)

    cleanup_transformations = [
        InferShapes(),
        GiveUniqueParameterTensors(),
        FoldConstants(exclude_op_types=preserve_qnt_optypes),
        FoldTransposeIntoQuantInit(),
        RemoveUnusedTensors(),
        RemoveStaticGraphInputs(),
        GiveUniqueNodeNames(),
        GiveReadableTensorNames(),
    ]
    for t in cleanup_transformations:
        model = model.transform(t)

    if extract_conv_bias:
        model = model.transform(ExtractBiasFromConv())
        model = model.transform(InferShapes())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())

    return model


def cleanup(in_file, *, out_file=None, preserve_qnt_ops=True, override_inpsize: str = None, extract_conv_bias=False):
    """Execute a set of graph transformations to clean-up the given ONNX file.

    :param in_file: Filename for the input ONNX model
    :param preserve_qnt_ops: Preserve weight quantization operators
    :param out_file: If set, filename for the output ONNX model. Set to in_file with _clean
        suffix otherwise.
    :param override_inpsize: If specified, override the input size (e.g. "(1,3,224,224)" to set all or
        just 1 to set batchsize to 1) for the ONNX graph
    :param extract_conv_bias: If specified, separate Conv bias into its own Add node
    """

    model = ModelWrapper(in_file)
    model = cleanup_model(
        model, preserve_qnt_ops=preserve_qnt_ops, override_inpsize=override_inpsize, extract_conv_bias=extract_conv_bias
    )
    if out_file is None:
        out_file = in_file.replace(".onnx", "_clean.onnx")
    model.save(out_file)


def main():
    clize.run(cleanup)


if __name__ == "__main__":
    main()
