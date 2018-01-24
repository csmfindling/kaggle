import numpy
from PIL import Image
from fuel.transformers import ExpectsAxisLabels, Transformer, SourcewiseTransformer
import math
import random


class RandomDownscale(Transformer):
    """Randomly downscale a video with minimum dimension given as parameter

    Parameters
    ----------
    data_stream : instance of :class:`AbstractDataStream`
        The data stream to wrap.
    min_dimension_size : int
        The desired length of the smallest dimension.
    resample : str, optional
        Resampling filter for PIL to use to upsample any images requiring
        it. Options include 'nearest' (default), 'bilinear', and 'bicubic'.
        See the PIL documentation for more detailed information.

    Notes
    -----
    This transformer only works with square images (width == height)

    """
    def __init__(self, data_stream, min_dimension_size, resample='nearest',
                 **kwargs):
        self.min_dimension_size = min_dimension_size
        try:
            self.resample = getattr(Image, resample.upper())
        except AttributeError:
            raise ValueError("unknown resampling filter '{}'".format(resample))
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(RandomDownscale, self).__init__(data_stream, **kwargs)

    def transform_batch(self, batch):
        output = ([],[],[],[])

        for case, multiplier, images, targets in zip(batch[0], batch[1], batch[2], batch[3]):
            output[0].append(case)
            output[3].append(targets)
            rescaled_imgs, new_multiplier = self._example_transform(images) 
            output[2].append(rescaled_imgs)
            output[1].append(multiplier*new_multiplier)

        return output

    def transform_example(self, example):
        return self._example_transform(example)

    def _example_transform(self, example):
        if example.ndim > 3 or example.ndim < 2:
            raise NotImplementedError
        original_height, original_width = example.shape[-2:]

        new_size = random.randint(self.min_dimension_size, original_width)
        multiplier = float(new_size)/width

        dt = example.dtype
        target = numpy.zeros((example.shape[0], new_size, new_size))

        for i in range(example.shape[0]):

            im = Image.fromarray(example[i,:,:].astype('int16'))
            im = numpy.array(im.resize((new_size, new_size), resample=self.resample)).astype(dt)

            target[i,:,:] = im
        return target, multiplier


class RandomRotate(SourcewiseTransformer, ExpectsAxisLabels):
    """Randomly rotate a video with max angle as parameter

    Parameters
    ----------
    data_stream : instance of :class:`AbstractDataStream`
        The data stream to wrap.
    min_dimension_size : int
        The desired length of the smallest dimension.
    resample : str, optional
        Resampling filter for PIL to use to upsample any images requiring
        it. Options include 'nearest' (default), 'bilinear', and 'bicubic'.
        See the PIL documentation for more detailed information.

    Notes
    -----
    This transformer only works with square images (width == height)

    """
    def __init__(self, data_stream, maximum_rotation=math.pi, resample='bilinear',
                 **kwargs):
        try:
            self.resample = getattr(Image, resample.upper())
        except AttributeError:
            raise ValueError("unknown resampling filter '{}'".format(resample))
        self.maximum_rotation = numpy.rad2deg(maximum_rotation)

        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(RandomRotate, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, batch, source_name):
        return [self._example_transform(im, source_name) for im in batch]

    def transform_source_example(self, example, source_name):
        return self._example_transform(example, source_name)

    def _example_transform(self, example, source_name):
        if example.ndim > 3 or example.ndim < 2:
            raise NotImplementedError
        original_height, original_width = example.shape[-2:]

        rotation_angle = random.uniform(-self.maximum_rotation, self.maximum_rotation) + random.randint(0,3)*90

        dt = example.dtype
        target = numpy.zeros((example.shape[0], original_height, original_width))

        for i in range(example.shape[0]):

            im = Image.fromarray(example[i,:,:].astype('int16'))
            im = numpy.array(im.rotate(rotation_angle,
                                        resample=self.resample)).astype(dt)


            target[i,:,:] = im
        return target
