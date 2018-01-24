# source : https://github.com/mila-udem/blocks-examples/blob/master/mnist_lenet/__init__.py

import theano
from theano import tensor
from blocks.bricks import BatchNormalization, Rectifier, Linear, Softmax, MLP, BatchNormalizedMLP, FeedforwardSequence, Rectifier
from blocks.bricks.conv import Convolutional, ConvolutionalSequence, Flattener, MaxPooling
from blocks.initialization import IsotropicGaussian, Uniform, Constant
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.select import Selector
import numpy


def convolutional_sequence(filter_size, num_filters, image_size, num_channels=1):
    
    layers = []
    # layers.append(BatchNormalization(name='batchnorm_pixels'))

    layers.append(Convolutional(filter_size=filter_size, num_filters=num_filters, use_bias=True, tied_biases=True, name='conv_1'))
    layers.append(BatchNormalization(name='batchnorm_1'))
    layers.append(Rectifier(name='non_linear_1'))
    
    layers.append(Convolutional(filter_size=filter_size, num_filters=num_filters, use_bias=True, tied_biases=False, name='conv_2'))
    layers.append(BatchNormalization(name='batchnorm_2'))
    layers.append(Rectifier(name='non_linear_2'))
    
    layers.append(MaxPooling(pooling_size=(2,2), name='maxpool_2'))
        
    layers.append(Convolutional(filter_size=filter_size, num_filters=num_filters*2, use_bias=True, tied_biases=True, name='conv_3'))
    layers.append(BatchNormalization(name='batchnorm_3'))
    layers.append(Rectifier(name='non_linear_3'))

    layers.append(Convolutional(filter_size=filter_size, num_filters=num_filters*2, use_bias=True, tied_biases=True, name='conv_4'))
    layers.append(BatchNormalization(name='batchnorm_4'))
    layers.append(Rectifier(name='non_linear_4'))
    
    layers.append(MaxPooling(pooling_size=(2,2), name='maxpool_4'))
    
    layers.append(Convolutional(filter_size=filter_size, num_filters=num_filters*4, use_bias=True, tied_biases=False, name='conv_5'))
    layers.append(BatchNormalization(name='batchnorm_5'))
    layers.append(Rectifier(name='non_linear_5'))

    layers.append(Convolutional(filter_size=filter_size, num_filters=num_filters*4, use_bias=True, tied_biases=True, name='conv_6'))
    layers.append(BatchNormalization(name='batchnorm_6'))
    layers.append(Rectifier(name='non_linear_6'))
    
    layers.append(MaxPooling(pooling_size=(2,2), name='maxpool_6'))

    layers.append(Convolutional(filter_size=filter_size, num_filters=num_filters*8, use_bias=True, tied_biases=True, name='conv_7'))
    layers.append(BatchNormalization(name='batchnorm_7'))
    layers.append(Rectifier(name='non_linear_7'))

    layers.append(Convolutional(filter_size=filter_size, num_filters=num_filters*8, use_bias=True, tied_biases=True, name='conv_8'))
    layers.append(BatchNormalization(name='batchnorm_8'))
    layers.append(Rectifier(name='non_linear_8'))
    
    layers.append(MaxPooling(pooling_size=(2,2), name='maxpool_8'))
    
    return ConvolutionalSequence(layers, num_channels=num_channels, image_size=image_size, biases_init=Constant(0.),  weights_init=IsotropicGaussian(0.01))


# images_train = tensor.dtensor4('images_train'); labels_train  = tensor.lmatrix('labels_train'); images_valid = tensor.dtensor4('images_valid'); labels_valid  = tensor.lmatrix('labels_valid')
def build_model(images, labels):
    
    # Construct a bottom convolutional sequence
    bottom_conv_sequence = convolutional_sequence((3,3), 16, (160, 160))
    bottom_conv_sequence._push_allocation_config()
    
    # Flatten layer
    flattener = Flattener()

    # Construct a top MLP
    conv_out_dim = numpy.prod(bottom_conv_sequence.get_dim('output'))
    #top_mlp = MLP([Rectifier(name='non_linear_9'), Softmax(name='non_linear_11')], [conv_out_dim, 1024, 10], weights_init=IsotropicGaussian(), biases_init=Constant(0))
    top_mlp = BatchNormalizedMLP([Rectifier(name='non_linear_9'), Softmax(name='non_linear_11')], [conv_out_dim, 1024, 10], weights_init=IsotropicGaussian(), biases_init=Constant(0))
    
    # Construct feedforward sequence
    ss_seq = FeedforwardSequence([bottom_conv_sequence.apply, flattener.apply, top_mlp.apply])
    ss_seq.push_initialization_config()
    ss_seq.initialize()
    
    prediction = ss_seq.apply(images)
    cost_noreg = CategoricalCrossEntropy().apply(labels.flatten(), prediction)

    # add regularization
    selector = Selector([top_mlp])
    Ws = selector.get_parameters('W')
    mlp_brick_name = 'batchnormalizedmlp'
    W0 = Ws['/%s/linear_0.W' % mlp_brick_name]
    W1 = Ws['/%s/linear_1.W' % mlp_brick_name]

    cost = cost_noreg + .01 * (W0 ** 2).mean() + .01 * (W1 ** 2).mean()


    return cost
                  
    
