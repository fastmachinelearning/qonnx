def reseed(newseed):
    import numpy
    import onnxruntime

    print(f"pytest-randomly: reseed with {newseed}")
    onnxruntime.set_seed(newseed)
    numpy.random.seed(seed=(newseed % 2**32))
