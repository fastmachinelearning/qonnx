import numpy
import onnxruntime
import tensorflow


def reseed(newseed):
    print(f"pytest-randomly: reseed with {newseed}")
    onnxruntime.set_seed(newseed)
    tensorflow.random.set_seed(newseed)
    numpy.random.seed(seed=newseed)
