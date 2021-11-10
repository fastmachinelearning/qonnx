import onnxruntime

from qonnx import converters


def reseed(newseed):
    onnxruntime.set_seed(newseed)
