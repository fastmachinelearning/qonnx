from qonnx.custom_op.channels_last.batch_normalization import BatchNormalization
from qonnx.custom_op.channels_last.conv import Conv
from qonnx.custom_op.channels_last.max_pool import MaxPool
from qonnx.custom_op.channels_last.concat import Concat
from qonnx.custom_op.channels_last.resize import Resize


custom_op = dict()

custom_op["Conv"] = Conv
custom_op["MaxPool"] = MaxPool
custom_op["BatchNormalization"] = BatchNormalization
custom_op["Concat"] = Concat
custom_op["Resize"] = Resize

