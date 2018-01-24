import numpy
from PIL import Image
from fuel.transformers import ExpectsAxisLabels, Transformer, SourcewiseTransformer,AgnosticSourcewiseTransformer
from fuel import config
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
        output = ([],[],[],[])
        for case, images, targets, multiplier in zip(batch[0], batch[1], batch[2], batch[3]):
            output[0].append(case)
            rescaled_imgs, new_multiplier, new_size = self._example_transform(targets) 
            output[2].append(rescaled_imgs)
            rescaled_imgs, new_multiplier, new_size = self._example_transform(images, new_size) 
            output[1].append(rescaled_imgs)
            output[3].append(multiplier*new_multiplier)
        return output

    def transform_example(self, example):
        return self._example_transform(example)

    def _example_transform(self, example, new_size=-1):
        if example.ndim > 3 or example.ndim < 2:
            raise NotImplementedError
        original_height, original_width = example.shape[-2:]

        if new_size == -1:
            new_size = random.randint(self.min_dimension_size, original_width)
        multiplier = float(new_size)/original_width

        dt     = example.dtype
        target = numpy.zeros((example.shape[0], new_size, new_size))

        for i in range(example.shape[0]):
    
            im = Image.fromarray(example[i].astype('int16'))
            im = numpy.array(im.resize((new_size, new_size), resample=self.resample)).astype(dt)

            target[i,:,:] = im
        return target, multiplier, new_size

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
        output = ([],[],[],[])
        for case, images, targets, multiplier in zip(batch[0], batch[1], batch[2], batch[3]):
            output[0].append(case)
            rescaled_imgs, rotation_angle = self._example_transform(targets) 
            output[2].append(rescaled_imgs)
            rescaled_imgs, rotation_angle = self._example_transform(images,rotation_angle) 
            output[1].append(rescaled_imgs)
            output[3].append(multiplier)
        return output

    def transform_example(self, example):
        return self._example_transform(example)

    def _example_transform(self, example, rotation_angle=-1):
        if example.ndim > 3 or example.ndim < 2:
            raise NotImplementedError
        original_height, original_width = example.shape[-2:]

        if rotation_angle == -1:
            rotation_angle = random.uniform(-self.maximum_rotation, self.maximum_rotation) + random.randint(0,3)*90

        print rotation_angle

        dt = example.dtype
        target = numpy.zeros((example.shape[0], original_height, original_width))

        for i in range(example.shape[0]):

            im = Image.fromarray(example[i,:,:].astype('int16'))
            im = numpy.array(im.rotate(rotation_angle,
                                        resample=self.resample)).astype(dt)


            target[i,:,:] = im
        return target, rotation_angle

class RandomLimit(Transformer):
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
    def __init__(self, data_stream, maximum_limitation, resample='bilinear',
                 **kwargs):
        self.maximum_limitation = maximum_limitation
        try:
            self.resample = getattr(Image, resample.upper())
        except AttributeError:
            raise ValueError("unknown resampling filter '{}'".format(resample))
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(RandomLimit, self).__init__(data_stream, **kwargs)

    def transform_batch(self, batch):
        output = ([],[],[],[])
        for case, images, targets, multiplier in zip(batch[0], batch[1], batch[2], batch[3]):
            output[0].append(case)
            limited_imgs, index = self._example_transform(targets) 
            output[2].append(limited_imgs)
            limited_imgs, index = self._example_transform(images,index) 
            output[1].append(limited_imgs)
            output[3].append(multiplier)
        return output

    def transform_example(self, example):
        return self._example_transform(example)

    def _example_transform(self, example, first_index=-1):
        if example.ndim > 3 or example.ndim < 2:
            raise NotImplementedError
        original_height, original_width = example.shape[-2:]
        nb_of_frames                    = example.shape[0]
        assert(nb_of_frames >= self.maximum_limitation)
        if first_index == -1:
            first_index = numpy.random.randint(nb_of_frames - self.maximum_limitation + 1)
        #print first_index
        dt = example.dtype
        target = numpy.zeros((self.maximum_limitation, original_height, original_width))

        for i in range(self.maximum_limitation):
            target[i,:,:] = example[first_index + i, :, :]
        return target, first_index

class Normalize(Transformer):
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
        output = ([],[],[],[])
        for case, images, targets, multiplier in zip(batch[0], batch[1], batch[2], batch[3]):
            output[0].append(case)
            output[2].append(targets)
            normalised_imgs = self._example_transform(images) 
            output[1].append(normalised_imgs)
            output[3].append(multiplier)
        return output

    def transform_example(self, example):
        return self._example_transform(example)

    def _example_transform(self, example, first_index=-1):
        if example.ndim > 3 or example.ndim < 2:
            raise NotImplementedError
        return (example - numpy.mean(example))/numpy.std(example)


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
        self.dtype = dtype
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(Cast, self).__init__(data_stream, **kwargs)

    def transform_batch(self, batch):
        output = ([],[],[],[])
        for case, images, targets, multiplier in zip(batch[0], batch[1], batch[2], batch[3]):
            output[0].append(case)
            casted_trgs = self._example_transform(targets) 
            output[2].append(casted_trgs)
            casted_imgs = self._example_transform(images) 
            output[1].append(casted_imgs)
            output[3].append(multiplier)
        return output

    def transform_example(self, example):
        return self._example_transform(example)

    def _example_transform(self, example):
        if example.ndim > 3 or example.ndim < 2:
            raise NotImplementedError
        return numpy.asarray(example, dtype=self.dtype)


class RandomFixedSizeCrop(Transformer):
    """Randomly crop images to a fixed window size.
    Parameters
    ----------
    data_stream : :class:`AbstractDataStream`
        The data stream to wrap.
    window_shape : tuple
        The `(height, width)` tuple representing the size of the output
        window.
    Notes
    -----
    This transformer expects to act on stream sources which provide one of
     * Single images represented as 3-dimensional ndarrays, with layout
       `(channel, height, width)`.
     * Batches of images represented as lists of 3-dimensional ndarrays,
       possibly of different shapes (i.e. images of differing
       heights/widths).
     * Batches of images represented as 4-dimensional ndarrays, with
       layout `(batch, channel, height, width)`.
    The format of the stream will be un-altered, i.e. if lists are
    yielded by `data_stream` then lists will be yielded by this
    transformer.
    """
    def __init__(self, data_stream, window_shape, **kwargs):
        self.window_shape = window_shape
        self.rng = kwargs.pop('rng', None)
        self.warned_axis_labels = False
        if self.rng is None:
            self.rng = numpy.random.RandomState(config.default_seed)
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(RandomFixedSizeCrop, self).__init__(data_stream, **kwargs)

    def transform_batch(self, batch):
        output = ([],[],[],[])
        for case, images, targets, multiplier in zip(batch[0], batch[1], batch[2], batch[3]):
            output[0].append(case)
            rotated_trgs, off_h, off_w = self._example_transform(targets) 
            output[2].append(rotated_trgs)
            rotated_imgs, off_h, off_w = self._example_transform(images, off_h, off_w) 
            output[1].append(rotated_imgs)
            output[3].append(multiplier)
        return output

    def transform_example(self, example):
        return self._example_transform(example)

    def _example_transform(self, example, off_h = -1, off_w = -1):
        if example.ndim > 3 or example.ndim < 2:
            raise NotImplementedError
        windowed_height, windowed_width = self.window_shape
        if not isinstance(example, numpy.ndarray) or example.ndim != 3:
            raise ValueError("uninterpretable example format; expected "
                             "ndarray with ndim = 3")
        image_height, image_width = example.shape[1:]
        if image_height < windowed_height or image_width < windowed_width:
            raise ValueError("can't obtain ({}, {}) window from image "
                             "dimensions ({}, {})".format(
                                 windowed_height, windowed_width,
                                 image_height, image_width))
        if off_h == -1:
            if image_height - windowed_height > 0:
                off_h = self.rng.random_integers(0, image_height - windowed_height)
            else:
                off_h = 0
        if off_w == -1:
            if image_width - windowed_width > 0:
                off_w = self.rng.random_integers(0, image_width - windowed_width)
            else:
                off_w = 0
        return example[:, off_h:off_h + windowed_height,
                       off_w:off_w + windowed_width], off_h, off_w



################################################# OLD ########################################################

# class Cast(AgnosticSourcewiseTransformer):
#     """Casts selected sources as some dtype.
#     Incoming sources will be treated as numpy arrays (i.e. using
#     `numpy.asarray`).
#     Parameters
#     ----------
#     dtype : str
#         Data type to cast to. Can be any valid numpy dtype, or 'floatX',
#         in which case ``fuel.config.floatX`` is used.
#     """
#     def __init__(self, data_stream, dtype, **kwargs):
#         if dtype == 'floatX':
#             dtype = config.floatX
#         self.dtype = dtype
#         if data_stream.axis_labels:
#             kwargs.setdefault('axis_labels', data_stream.axis_labels.copy())
#         super(Cast, self).__init__(
#             data_stream, data_stream.produces_examples, **kwargs)

#     def transform_any_source(self, source_data, _):
#         l   = len(source_data)
#         res = [] 
#         for i in range(l):
#             res.append(numpy.asarray(source_data[i], dtype=self.dtype))
#         return res




