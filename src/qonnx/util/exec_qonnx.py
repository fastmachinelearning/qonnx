import clize
import numpy as np
import onnx

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx


def exec_qonnx(qonnx_model_file, *in_npy, override_opset: int = None, output_prefix: str = "out_"):
    """Execute a given QONNX model by initializing its inputs from .npy files, and write outputs
    as .npy files.
    The input model have been previously cleaned by the cleanup transformation or commandline tool.

    :param qonnx_model_file: Filename for the input ONNX model
    :param in_npy: List of .npy files to supply as inputs. If not specified, inputs will be set to zero.
    :param override_opset: If specified, override the imported ONNX opset to this version.
    :param output_prefix: Prefix for the generated output files.
    """

    # Execute transformation
    model = ModelWrapper(qonnx_model_file)
    if override_opset is not None:
        model.model.opset_import[0].version = override_opset

    idict = {}
    inp_ind = 0
    for inp in model.graph.input:
        if inp_ind < len(in_npy):
            idict[inp.name] = np.load(in_npy[inp_ind])
        else:
            i_tensor_shape = model.get_tensor_shape(inp.name)
            i_dtype_npy = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[inp.type.tensor_type.elem_type]
            idict[inp.name] = np.zeros(i_tensor_shape, dtype=i_dtype_npy)
        inp_ind += 1
    odict = execute_onnx(model, idict)
    outp_ind = 0
    for outp in model.graph.output:
        np.save(output_prefix + "%d.npy" % outp_ind, odict[outp.name])
        outp_ind += 1


def main():
    clize.run(exec_qonnx)


if __name__ == "__main__":
    main()
