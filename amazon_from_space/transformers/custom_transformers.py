import numpy
from PIL import Image
from fuel.transformers import ExpectsAxisLabels, Transformer, SourcewiseTransformer
import math
import random

class Standardize(Transformer):
    """RStandardize an image

    Parameters
    ----------
    data_stream : instance of :class:`AbstractDataStream`
        The data stream to wrap.
    max_value : the max value to divide with
    """

    def __init__(self, data_stream, max_value, **kwargs):
        self.max_value = float(max_value)
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(Standardize, self).__init__(data_stream, **kwargs)

    def transform_batch(self, batch):
        output = ([],[])

        for features, labels in zip(batch[0], batch[1]):
            output[0].append(self._example_transform(features))
            output[1].append(labels)
        return output

    def transform_example(self, example):
        return self._example_transform(example)

    def _example_transform(self, example):
        return numpy.asarray(example/self.max_value, dtype=numpy.float32)

