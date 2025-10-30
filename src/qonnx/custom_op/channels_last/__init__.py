# Importing registers CustomOps in qonnx.custom_op.channels_last domain
from qonnx.custom_op.channels_last.batch_normalization import BatchNormalization
from qonnx.custom_op.channels_last.conv import Conv
from qonnx.custom_op.channels_last.max_pool import MaxPool

# Legacy dictionary for backward compatibility
custom_op = {
    "Conv": Conv,
    "MaxPool": MaxPool,
    "BatchNormalization": BatchNormalization,
}