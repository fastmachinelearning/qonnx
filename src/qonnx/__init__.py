import onnxruntime

from qonnx import converters  # noqa: F401


def reseed(newseed):
    onnxruntime.set_seed(newseed)
