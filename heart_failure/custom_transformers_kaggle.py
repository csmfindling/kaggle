import numpy
from PIL import Image
from fuel.transformers import ExpectsAxisLabels, Transformer, SourcewiseTransformer,AgnosticSourcewiseTransformer
from fuel import config
import math
import random
from sklearn import linear_model
import pickle as pkl

class ApplyMask(Transformer):
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
    def __init__(self, data_stream, resample='nearest',
                 **kwargs):
        try:
            self.resample = getattr(Image, resample.upper())
        except AttributeError:
            raise ValueError("unknown resampling filter '{}'".format(resample))
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(ApplyMask, self).__init__(data_stream, **kwargs)

    def transform_batch(self, batch):
        output = ([],[],[],[],[],[])
        for case, position, multiplier, sax, images, targets in zip(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]):
            output[0].append(case)
            output[1].append(position)
            output[3].append(sax)
            masked_imgs = self._example_transform(images) 
            output[4].append(masked_imgs)
            output[2].append(multiplier)
            output[5].append(targets)
        return output

    def transform_example(self, example):
        return self._example_transform(example)

    def _example_transform(self, example):
        if example.ndim > 4 or example.ndim < 3:
            raise NotImplementedError
        depth, time, height, width = example.shape

        mask    = pkl.load(open("mask.pkl","rb"))
        mask    = mask*1
        im_mask = Image.fromarray(mask.astype('int16'))
        resized_mask  = im_mask.resize((width,height))
        array_mask    = numpy.array(resized_mask.getdata()).reshape(height, width)

        dt     = example.dtype

        for i in range(depth):
            for j in range(time):

                example[i,j]   = numpy.multiply(example[i,j],array_mask)

        return example.astype(dt)

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
        output = ([],[],[],[],[],[])
        for case, position, multiplier, sax, images, targets in zip(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]):
            output[0].append(case)
            output[1].append(position)
            output[3].append(sax)
            rescaled_imgs, new_multiplier = self._example_transform(images) 
            output[4].append(rescaled_imgs)
            output[2].append(multiplier*new_multiplier)
            output[5].append(targets)
        return output

    def transform_example(self, example):
        return self._example_transform(example)

    def _example_transform(self, example):
        if example.ndim > 4 or example.ndim < 3:
            raise NotImplementedError
        depth, time, height, width = example.shape

        new_size = random.randint(self.min_dimension_size, width)
        multiplier = (float(width)/float(new_size))**2

        dt     = example.dtype
        target = numpy.zeros((depth, time, new_size, new_size))

        for i in range(depth):
            for j in range(time):
    
                im = Image.fromarray(example[i,j].astype('int16'))
                im = numpy.array(im.resize((new_size, new_size), resample=self.resample)).astype(dt)

                target[i,j,:,:] = im
        return target, multiplier

class OrderFeatures(Transformer):
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
    def __init__(self, data_stream, resample='bilinear',
                 **kwargs):
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(OrderFeatures, self).__init__(data_stream, **kwargs)

    def transform_batch(self, batch):
        clf = linear_model.LinearRegression()
        output = ([],[],[],[],[],[])
        for case, position, multiplier, sax, images, targets in zip(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]):
            clf.fit(position[:,:2], position[:,2])
            new_coordinates = numpy.array([position[:,0], position[:,1], clf.predict(position[:,:2])]).T
            origine         = numpy.array([0,0,clf.predict([[0,0]])])
            index           = numpy.argsort(numpy.sign(numpy.sum(origine) - numpy.sum(new_coordinates, axis=1)) * numpy.sum((origine - new_coordinates)**2, axis=1))
            #index           = numpy.argsort(numpy.sum((origine - new_coordinates)**2, axis=1))
            output[0].append(case)
            output[1].append(position[index])
            output[3].append(sax[index])
            output[4].append(images[index])
            output[2].append(multiplier)
            output[5].append(targets)
        return output

    def transform_example(self, example):
        return self._example_transform(example)

    def _example_transform(self, example):
        if example.ndim > 4 or example.ndim < 3:
            raise NotImplementedError
        return 0

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
        output = ([],[],[],[],[],[])
        for case, position, multiplier, sax, images, targets in zip(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]):
            output[0].append(case)
            output[1].append(position)
            output[2].append(multiplier)
            output[3].append(sax)
            rotated_imgs = self._example_transform(images) 
            output[4].append(rotated_imgs)
            output[5].append(targets)
        return output

    def transform_example(self, example):
        return self._example_transform(example)

    def _example_transform(self, example):
        if example.ndim > 4 or example.ndim < 3:
            raise NotImplementedError

        depth, time, height, width = example.shape
        rotation_angle = random.uniform(-self.maximum_rotation, self.maximum_rotation) + random.randint(0,3)*90
        dt = example.dtype
        target = numpy.zeros((depth, time, height, width))

        for i in range(depth):
            for j in range(time):
                im = Image.fromarray(example[i,j,:,:].astype('int16'))
                im = numpy.array(im.rotate(rotation_angle,
                                        resample=self.resample)).astype(dt)

                target[i,j,:,:] = im
        return target

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
        output = ([],[],[],[],[],[])
        for case, position, multiplier, sax, images, targets in zip(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]):
            output[0].append(case)
            output[1].append(position)
            output[2].append(multiplier)
            output[3].append(sax)
            limited_imgs = self._example_transform(images) 
            output[4].append(limited_imgs)
            output[5].append(targets)
        return output

    def transform_example(self, example):
        return self._example_transform(example)

    def _example_transform(self, example):
        if example.ndim > 4 or example.ndim < 3:
            raise NotImplementedError
        nb_of_frames                    = example.shape[1]
        assert(nb_of_frames >= self.maximum_limitation)
        depth, time, height, width = example.shape
        target = numpy.zeros((depth, self.maximum_limitation, height, width))
        first_index = numpy.random.randint(nb_of_frames - self.maximum_limitation + 1)
        for i in range(depth):
            for j in range(self.maximum_limitation):        
                target[i,j,:,:] = example[i,first_index + j, :, :]
        return target

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
        output = ([],[],[],[],[],[])
        for case, position, multiplier, sax, images, targets in zip(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]):
            output[0].append(case)
            output[1].append(position)
            output[2].append(multiplier)
            output[3].append(sax)
            normalised_imgs = self._example_transform(images) 
            output[4].append(normalised_imgs)
            output[5].append(targets)
        return output

    def transform_example(self, example):
        return self._example_transform(example)

    def _example_transform(self, example):
        if example.ndim > 4 or example.ndim < 3:
            raise NotImplementedError
        depth, time, height, width = example.shape
        target = numpy.zeros((depth, time, height, width))
        for i in range(depth):
	    if numpy.std(example[i]) == 0:
		target[i] = numpy.zeros(target[i].shape)
	    else:
            	target[i] = (example[i] - numpy.mean(example[i]))/numpy.std(example[i])
        return target


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
        output = ([],[],[],[],[],[])
        for case, position, multiplier, sax, images, targets in zip(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]):
            output[0].append(case)
            output[1].append(position)
            output[2].append(multiplier)
            output[3].append(sax)
            casted_imgs = self._example_transform(images) 
            output[4].append(casted_imgs)
            output[5].append(targets)
        return output

    def transform_example(self, example):
        return self._example_transform(example)

    def _example_transform(self, example):
        if example.ndim > 4 or example.ndim < 3:
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
        kwargs.setdefault('produces_examples', False)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(RandomFixedSizeCrop, self).__init__(data_stream, **kwargs)

    def transform_batch(self, batch):
        output = ([],[],[],[],[],[])
        for case, position, multiplier, sax, images, targets in zip(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]):
            output[0].append(case)
            output[1].append(position)
            output[2].append(multiplier)
            output[3].append(sax)
            cropped_imgs = self._example_transform(images) 
            output[4].append(cropped_imgs)
            output[5].append(targets)
        return output

    def transform_example(self, example):
        return self._example_transform(example)

    def _example_transform(self, example):
        if example.ndim > 4 or example.ndim < 3:
            raise NotImplementedError
        windowed_height, windowed_width = self.window_shape
        if not isinstance(example, numpy.ndarray) or example.ndim != 4:
            raise ValueError("uninterpretable example format; expected "
                             "ndarray with ndim = 4")
        image_height, image_width = example.shape[2:]
        if image_height < windowed_height or image_width < windowed_width:
            raise ValueError("can't obtain ({}, {}) window from image "
                             "dimensions ({}, {})".format(
                                 windowed_height, windowed_width,
                                 image_height, image_width))
        depth, time, height, width = example.shape
        target = numpy.zeros((depth, time, windowed_height, windowed_width))
        
        off_h = self.rng.random_integers(0, image_height - windowed_height)
        off_w = self.rng.random_integers(0, image_width - windowed_width)

        for i in range(depth):
            for j in range(time):
                target[i,j] = example[i, j, off_h:off_h + windowed_height, off_w:off_w + windowed_width]
        return target


class ZeroPadding(Transformer):
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
    def __init__(self, data_stream, resample='bilinear',
                 **kwargs):
        self.max=22
        try:
            self.resample = getattr(Image, resample.upper())
        except AttributeError:
            raise ValueError("unknown resampling filter '{}'".format(resample))
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(ZeroPadding, self).__init__(data_stream, **kwargs)

    def transform_batch(self, batch):
        output = ([],[],[],[],[],[])
        max_number_sax = numpy.max([batch[1][i].shape[0] for i in range(len(batch[2]))])
        max_check      = numpy.max([batch[3][i].shape[0] for i in range(len(batch[2]))])
        max_check_1    = numpy.max([batch[4][i].shape[0] for i in range(len(batch[2]))])
        if max_number_sax != max_check or max_number_sax != max_check_1:
            raise ValueError("problem with case '{}'".format(batch[0][i]))
        output = ([],[],[],[],[],[])
        for case, position, multiplier, sax, images, targets in zip(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]):
            output[0].append(case)
            output[2].append(multiplier)
            zero_padded_sax = self._example_transform(sax, self.max)
            output[3].append(zero_padded_sax)
            zero_padded_imgs = self._example_transform(images, self.max) 
            output[4].append(zero_padded_imgs)
            output[5].append(targets)
            zero_padded_position = self._example_transform(position, self.max)
            output[1].append(zero_padded_position)
        return output

    def transform_example(self, example):
        return self._example_transform(example)

    def _example_transform(self, example, max_number_sax):
        result = example
        if example.ndim == 2:
            if example.shape[0] < max_number_sax:
                result = numpy.zeros([max_number_sax, 3])
                result[:example.shape[0]] = example
        elif example.ndim == 1:
            if example.shape[0] < max_number_sax:
                result = numpy.zeros(max_number_sax).astype('int16') - 1
                result[:example.shape[0]] = example
        elif example.ndim == 4:
            if example.shape[0] < max_number_sax:
                depth, time, height, width = example.shape
                result = numpy.zeros((max_number_sax, time, height, width))
                result[:depth] = example
        else:
            raise ValueError("wrong dimension number")
        return result
