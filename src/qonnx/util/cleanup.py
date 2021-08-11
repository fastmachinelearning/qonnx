import clize
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
)
from finn.transformation.infer_shapes import InferShapes


def cleanup(in_file, *, out_file=None):
    """Execute a set of graph transformations to clean-up the given ONNX file.

    :param in_file: Filename for the input ONNX model
    :param out_file: If set, filename for the output ONNX model. Set to in_file with _clean
        suffix otherwise.
    """

    model = ModelWrapper(in_file)
    cleanup_transformations = [
        InferShapes(),
        GiveUniqueParameterTensors(),
        FoldConstants(),
        RemoveUnusedTensors(),
        RemoveStaticGraphInputs(),
        GiveUniqueNodeNames(),
        GiveReadableTensorNames(),
    ]
    for t in cleanup_transformations:
        model = model.transform(t)
    if out_file is None:
        out_file = in_file.replace(".onnx", "_clean.onnx")
    model.save(out_file)


def main():
    clize.run(cleanup)


if __name__ == "__main__":
    main()
