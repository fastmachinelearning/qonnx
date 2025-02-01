from qonnx.custom_op.channels_last.avg_pool import AveragePool
from qonnx.custom_op.channels_last.batch_normalization import BatchNormalization
from qonnx.custom_op.channels_last.conv import Conv
from qonnx.custom_op.channels_last.max_pool import MaxPool

custom_op = dict()

custom_op["Conv"] = Conv
custom_op["MaxPool"] = MaxPool
custom_op["BatchNormalization"] = BatchNormalization
custom_op["AveragePool"] = AveragePool
