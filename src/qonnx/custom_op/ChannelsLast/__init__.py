from qonnx.custom_op.ChannelsLast.wrapped_ops import BatchNormalization, Conv, MaxPool

custom_op = dict()

custom_op["Conv"] = Conv
custom_op["MaxPool"] = MaxPool
custom_op["BatchNormalization"] = BatchNormalization
