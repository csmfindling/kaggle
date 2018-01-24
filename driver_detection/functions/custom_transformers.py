import numpy
from PIL import Image
from fuel.transformers import ExpectsAxisLabels, Transformer, SourcewiseTransformer,AgnosticSourcewiseTransformer
from fuel import config
import random
import pickle as pkl

class RandomDownscale(Transformer):
    """Randomly downscale an image with minimum dimension given as parameter
    """
    def __init__(self, data_stream, min_dimension_size, resample='bilinear',
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
        raise NotImplementedError
        output = ([],[])
        for image, label in zip(batch[0], batch[1]):
            rescaled_img = self._example_transform(image) 
            output[0].append(rescaled_img)
            output[1].append(label)
        return output

    def transform_example(self, example):
        return self._example_transform(example)

    def _example_transform(self, example):
        if example.ndim > 3 or example.ndim < 2:
            raise NotImplementedError
        height, width, channels = example.shape
        dt                      = example.dtype
        assert(width == height)

        new_size = random.randint(self.min_dimension_size, height)
        im = Image.fromarray(example.astype('uint8'))
        im = numpy.array(im.resize((new_size, new_size), resample=self.resample)).astype(dt)

        return im

class RandomFixedSizeCrop(Transformer):
    """Randomly crops an image with minimum dimension given as parameter
    """
    def __init__(self, data_stream, window_shape, **kwargs):
        self.window_shape = window_shape
        self.rng = kwargs.pop('rng', None)
        self.warned_axis_labels = False
        if self.rng is None:
            self.rng = numpy.random.RandomState(config.default_seed)
        kwargs.setdefault('produces_examples', False)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(RandomFixedSizeCrop, self).__init__(data_stream, **kwargs)

    def transform_batch(self, batch):
        raise NotImplementedError
        output = ([],[])
        for image, label in zip(batch[0], batch[1]):
            cropped_img = self._example_transform(image) 
            output[0].append(cropped_img)
            output[1].append(label)
        return output

    def transform_example(self, example):
        return self._example_transform(example)

    def _example_transform(self, example):
        if example.ndim > 4 or example.ndim < 3:
            raise NotImplementedError
        windowed_height, windowed_width = self.window_shape
        height, width, channels         = example.shape
        im                              = numpy.zeros((windowed_height, windowed_width, channels))
        
        off_h = self.rng.random_integers(0, height - windowed_height)
        off_w = self.rng.random_integers(0, width - windowed_width)

        for i in range(channels):
            im[:,:,i] = example[off_h:off_h + windowed_height, off_w:off_w + windowed_width, i]
        return im


class FixedSizeCrop(Transformer):
    """Crops an image with minimum dimension given as parameter
    """
    def __init__(self, data_stream, window_shape, **kwargs):
        self.window_shape = window_shape
        self.rng = kwargs.pop('rng', None)
        self.warned_axis_labels = False
        if self.rng is None:
            self.rng = numpy.random.RandomState(config.default_seed)
        kwargs.setdefault('produces_examples', False)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(FixedSizeCrop, self).__init__(data_stream, **kwargs)

    def transform_batch(self, batch):
        raise NotImplementedError
        output = ([],[])
        for image, label in zip(batch[0], batch[1]):
            cropped_img = self._example_transform(image) 
            output[0].append(cropped_img)
            output[1].append(label)
        return output

    def transform_example(self, example):
        return self._example_transform(example)

    def _example_transform(self, example):
        if example.ndim > 4 or example.ndim < 3:
            raise NotImplementedError
        windowed_height, windowed_width = self.window_shape
        height, width, channels         = example.shape
        im                              = numpy.zeros((windowed_height, windowed_width, channels))
        
        off_h = (height - windowed_height)/2
        off_w = (width - windowed_width)/2

        for i in range(channels):
            im[:,:,i] = example[off_h:off_h + windowed_height, off_w:off_w + windowed_width, i]
        return im


class RandomRotate(Transformer):
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
    def __init__(self, data_stream, maximum_rotation, resample='bilinear',
                 **kwargs):
        self.maximum_rotation = maximum_rotation
        try:
            self.resample = getattr(Image, resample.upper())
        except AttributeError:
            raise ValueError("unknown resampling filter '{}'".format(resample))
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(RandomRotate, self).__init__(data_stream, **kwargs)

    def transform_batch(self, batch):
        output = ([],[],[])
        for driver_id, image, label in zip(batch[0], batch[1], batch[2]):
            normalised_img = self._example_transform(image) 
            output[0].append(driver_id)
            output[1].append(normalised_img)
            output[2].append(label)
        return output

    def transform_example(self, example):
        return self._example_transform(example)

    def _example_transform(self, example):
        if example.ndim > 3 or example.ndim < 2:
            raise NotImplementedError

        if example.ndim == 2:
            height, width = example.shape
            rotation_angle          = random.uniform(-self.maximum_rotation, self.maximum_rotation) #+ random.randint(0,3)*90
            dt                      = example.dtype

            im = Image.fromarray(example.astype('uint8'))
            im = numpy.array(im.rotate(rotation_angle,
                                            resample=self.resample)).astype(dt)
        elif example.ndim == 3:
            channels, height, width = example.shape
            rotation_angle          = random.uniform(-self.maximum_rotation, self.maximum_rotation) #+ random.randint(0,3)*90
            dt                      = example.dtype

            rolled = numpy.rollaxis(example, 0, 3).astype('uint8')

            im = Image.fromarray(rolled)
            im = numpy.array(im.rotate(rotation_angle,
                                            resample=self.resample)).astype(dt)
            im = numpy.rollaxis(im, 2, 0)


        return im

class Normalize(Transformer):
    """translate image between 0 and 1
    """
    def __init__(self, data_stream, resample='bilinear',
                 **kwargs):
        try:
            self.resample = getattr(Image, resample.upper())
        except AttributeError:
            raise ValueError("unknown resampling filter '{}'".format(resample))
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(Normalize, self).__init__(data_stream, **kwargs)

    def transform_batch(self, batch):
        output = ([],[],[])
        for driver_id, image, label in zip(batch[0], batch[1], batch[2]):
            normalised_img = self._example_transform(image) 
            output[0].append(driver_id)
            output[1].append(normalised_img)
            output[2].append(label)
        return output

    def transform_example(self, example):
        return self._example_transform(example)

    def _example_transform(self, example):
        if example.ndim > 3 or example.ndim < 2:
            raise NotImplementedError
        # height, width, channels = example.shape
        # im                      = numpy.zeros((height, width, channels))
        # for i in range(channels):
        #     std  = numpy.std(example[:,:,i])
        #     mean = numpy.mean(example[:,:,i])
        #     im[:,:,i] = (example[:,:,i] - numpy.mean(example[:,:,i]))/numpy.std(example[:,:,i])
        return example/255.


class Cast(Transformer):
    """Scales and shifts selected sources by scalar quantities.
    Incoming sources will be treated as numpy arrays (i.e. using
    `numpy.asarray`).
    Parameters
    ----------
    scale : float
        Scaling factor.
    shift : float
        Shifting factor.
    """
    def __init__(self, data_stream, dtype,
                 **kwargs):
        if dtype == 'floatX':
            dtype = config.floatX
        self.dtype = 'float32'
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(Cast, self).__init__(data_stream, **kwargs)

    def transform_batch(self, batch):
        output = ([],[],[])
        for driver_id, image, label in zip(batch[0], batch[1], batch[2]):
            casted_img = self._example_transform(image) 
            output[0].append(driver_id.astype(numpy.int32))
            output[1].append(casted_img)
            output[2].append(label.astype(numpy.int32))
        return output

    def transform_example(self, example):
        return self._example_transform(example)

    def _example_transform(self, example):
        if example.ndim > 3 or example.ndim < 2:
            raise NotImplementedError
        return numpy.asarray(example, dtype=self.dtype)


