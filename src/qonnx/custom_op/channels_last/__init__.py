from qonnx.custom_op.channels_last.batch_normalization import BatchNormalization
from qonnx.custom_op.channels_last.conv import Conv
from qonnx.custom_op.channels_last.max_pool import MaxPool

# channels-last ops are defined by the underlying ONNX standard op
# thus, we can define them for any version of the original op
# so we emulate a custom op dictionary that mimics the support for any
# {ChannelsLastOp}_vX instead of hardcoding what versions are supported


class ChannelsLastCustomOpDict(dict):
    def __init__(self):
        self._custom_ops = {"Conv": Conv, "MaxPool": MaxPool, "BatchNormalization": BatchNormalization}

    def __getitem__(self, key):
        base_key = key.split("_v")[0]  # Extract base key (e.g., Conv from Conv_v13)
        if base_key in self._custom_ops:
            return self._custom_ops[base_key]
        raise KeyError(f"Channels-last CustomOp '{key}' not found.")

    def keys(self):
        return self._custom_ops.keys()


custom_op = ChannelsLastCustomOpDict()
