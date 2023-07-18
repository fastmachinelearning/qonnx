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
import onnxruntime as rt
from tqdm import tqdm

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.change_batchsize import ChangeBatchSize
from qonnx.transformation.expose_intermediate import ExposeIntermediateTensorsPatternList
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.onnx import valueinfo_to_tensor

OUTPUT_MODE_IND = "tensor_index"
OUTPUT_MODE_NAME = "tensor_name"

output_modes = {OUTPUT_MODE_IND, OUTPUT_MODE_NAME}

output_mode_options = clize.parameters.mapped(
    [
        (OUTPUT_MODE_IND, [OUTPUT_MODE_IND], "Output files named by index"),
        (OUTPUT_MODE_NAME, [OUTPUT_MODE_NAME], "Output files named by tensor"),
    ]
)


def exec_qonnx(
    qonnx_model_file,
    *in_npy,
    override_batchsize: int = None,
    override_opset: int = None,
    expose_intermediates: str = None,
    output_prefix: str = "out_",
    output_mode: output_mode_options = OUTPUT_MODE_NAME,
    argmax_verify_npy: str = None,
    save_modified_model: str = None,
    input_pix2float=False,
    input_zerocenter=False,
    maxiters: int = None,
    output_nosave=False
):
    """Execute a given QONNX model by initializing its inputs from .npy files, and write outputs
    as .npy files.
    The input model have been previously cleaned by the cleanup transformation or commandline tool.

    :param qonnx_model_file: Filename for the input ONNX model
    :param in_npy: List of .npy files to supply as inputs. If not specified, inputs will be set to zero.
    :param override_batchsize: If specified, override the batch size for the ONNX graph
    :param override_opset: If specified, override the imported ONNX opset to this version.
    :param expose_intermediates: Comma-separated list of tensor name patterns.
        Matched patterns will expose intermediate outputs as top-level outputs.
    :param output_prefix: Prefix for the generated output files.
    :param output_mode: Naming mode for generated output files.
    :param argmax_verify_npy: If specified, take argmax of output and compare to this file for top-1 accuracy measurement
    :param save_modified_model: If specified, save the modified model
        (after batchsize changes or exposed intermediate tensors) with this filename
    :param input_pix2float: If specified, do uint8 [0,255] -> fp32 [0,1] mapping for input
    :param input_zerocenter: If specified together with pix2float, do uint8 [0,255] -> fp32 [-1,+1] mapping for input
    :param maxiters: If specified, limit maximum number of iterations (batches) to be processed
    :param output_nosave: If specified, do not save output tensors to files
    """
    assert output_mode in output_modes, "Unrecognized output mode"

    # Execute transformation
    model = ModelWrapper(qonnx_model_file)
    if override_opset is not None:
        model.model.opset_import[0].version = override_opset

    if override_batchsize is not None:
        model = model.transform(ChangeBatchSize(override_batchsize))
        model = model.transform(InferShapes())
        bsize = override_batchsize
    else:
        bsize = model.get_tensor_shape(model.graph.input[0].name)[0]

    if expose_intermediates is not None:
        pattern_list = expose_intermediates.split(",")
        pattern_list = [x.strip() for x in pattern_list]
        model = model.transform(FoldConstants(exclude_op_types=[]))
        model = model.transform(ExposeIntermediateTensorsPatternList(pattern_list, dynamic_only=True))

    if save_modified_model is not None:
        model.save(save_modified_model)

    n_custom_nodes = len(model.get_finn_nodes())
    if n_custom_nodes == 0:
        print("No custom qonnx nodes found, running in onnxruntime")

    ok = 0
    nok = 0
    iter = 0
    dset_size = 0
    n_dset_iters = 0
    inp_data = []
    labels = None
    if len(in_npy) > 0:
        # load provided npy files and arrange in batches
        inp_data = [np.load(x) for x in in_npy]
        inp_data_reshaped = []
        for inp in inp_data:
            dset_size = inp.shape[0]
            assert dset_size % bsize == 0, "Batch size %d must divide dataset size %d" % (bsize, dset_size)
            n_dset_iters = dset_size // bsize
            inp = inp.reshape(n_dset_iters, bsize, *inp.shape[1:])
            inp_data_reshaped.append(inp)
        inp_data = inp_data_reshaped
        if argmax_verify_npy is not None:
            labels = np.load(argmax_verify_npy)
            assert labels.shape[0] == dset_size, "Label size must match dataset size"
            labels = labels.reshape(n_dset_iters, bsize, *labels.shape[1:])
    else:
        # create 0-valued tensors of appropriate shape
        n_dset_iters = 1
        inp_data = [valueinfo_to_tensor(model.get_tensor_valueinfo(i.name)) for i in model.graph.input]
        inp_data = [np.expand_dims(x, axis=0) for x in inp_data]

    if maxiters is not None:
        n_dset_iters = min(n_dset_iters, maxiters)

    pbar = tqdm(range(n_dset_iters))

    for iter in pbar:
        iter_suffix = "_batch%d" % iter
        idict = {}
        if not argmax_verify_npy:
            pbar.set_description("Batch [%d/%d]: running" % (iter + 1, n_dset_iters))
        # supply inputs and execute
        for inp_ind, inp in enumerate(model.graph.input):
            if input_pix2float:
                idict[inp.name] = (inp_data[inp_ind][iter] / 255.0).astype(np.float32)
                if input_zerocenter:
                    idict[inp.name] = (2 * idict[inp.name] - 1.0).astype(np.float32)
            else:
                idict[inp.name] = inp_data[inp_ind][iter]
        if n_custom_nodes > 0:
            # run node-by-node in qonnx
            odict = execute_onnx(model, idict)
        else:
            # run using onnxruntime
            sess = rt.InferenceSession(model.model.SerializeToString())
            output_list = sess.run(None, idict)
            odict = {outp.name: output_list[oind] for oind, outp in enumerate(model.graph.output)}
        if not output_nosave:
            for out_ind, outp in enumerate(model.graph.output):
                # save generated outputs
                if output_mode == OUTPUT_MODE_IND:
                    oname = "%d" % out_ind
                elif output_mode == OUTPUT_MODE_NAME:
                    oname = outp.name
                np.save(output_prefix + oname + iter_suffix + ".npy", odict[outp.name])
        if argmax_verify_npy:
            # measure accuracy for output
            ret = odict[model.graph.output[0].name]
            ret = np.argmax(ret, axis=-1)
            ok_batch = np.count_nonzero(ret == labels[iter])
            nok_batch = bsize - ok_batch
            ok += ok_batch
            nok += nok_batch
            accuracy_batch = ok_batch / bsize
            accuracy_overall = ok / (ok + nok)
            pbar.set_description(
                "Batch [%d/%d]: ok %d nok %d accuracy %f (overall ok %d nok %d accuracy %f)"
                % (iter + 1, n_dset_iters, ok_batch, nok_batch, accuracy_batch, ok, nok, accuracy_overall)
            )


def main():
    clize.run(exec_qonnx)


if __name__ == "__main__":
    main()
