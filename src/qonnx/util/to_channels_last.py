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
