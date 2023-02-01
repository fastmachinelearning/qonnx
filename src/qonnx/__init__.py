import onnxruntime


def reseed(newseed):
    onnxruntime.set_seed(newseed)
