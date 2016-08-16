import numpy
from PIL import Image
from fuel.transformers import ExpectsAxisLabels, Transformer, SourcewiseTransformer,AgnosticSourcewiseTransformer
from fuel import config
import random
import pickle as pkl

class CloneAndTransform(Transformer):
    """translate image between 0 and 1
    """
    def __init__(self, data_stream, **kwargs):
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(CloneAndTransform, self).__init__(data_stream, **kwargs)

    def transform_batch(self, batch):
        output = ([],[],[])
        for features_cat, features_num, labels in zip(batch[0], batch[1], batch[2]):
            output[0].append(features_cat)
            output[1].append(features_num)
            output[2].append(self.transform_example(features_cat, features_num))
        return output

    def transform_example(self, example_cat, example_num):
        return self._example_transform(example_cat, example_num)

    def _example_transform(self, example_cat, example_num):
        # n_features = example_cat.shape[0] + example_num.shape[0]
        # output     = numpy.zeros(n_features)
        # output[:example_cat.shape[0]] = example_cat
        # output[example_cat.shape[0]:] = example_num
        return numpy.concatenate((example_cat, example_num))


