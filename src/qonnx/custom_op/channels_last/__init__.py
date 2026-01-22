# Importing registers CustomOps in qonnx.custom_op.channels_last domain
from qonnx.custom_op.channels_last.batch_normalization import (
    BatchNormalization_v1,
    BatchNormalization_v9,
    BatchNormalization_v14,
)
from qonnx.custom_op.channels_last.conv import Conv_v1
from qonnx.custom_op.channels_last.max_pool import MaxPool_v1, MaxPool_v10

__all__ = [
    "Conv_v1",
    "MaxPool_v1",
    "MaxPool_v10",
    "BatchNormalization_v1",
    "BatchNormalization_v9",
    "BatchNormalization_v14",
]
