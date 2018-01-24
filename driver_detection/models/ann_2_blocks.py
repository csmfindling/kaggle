# source : https://github.com/mila-udem/blocks-examples/blob/master/mnist_lenet/__init__.py

import theano
from theano import tensor
from blocks.bricks import BatchNormalization, Rectifier, Linear, Softmax, MLP, FeedforwardSequence
from blocks.bricks.conv import Convolutional, ConvolutionalSequence, Flattener, MaxPooling
from blocks.initialization import IsotropicGaussian, Uniform, Constant
from blocks.bricks.cost import CategoricalCrossEntropy
import numpy


def convolutional_sequence(filter_size, num_filters, image_size, num_channels=3):
    layers = []
    layers.append(Convolutional(filter_size=filter_size, num_filters=num_filters, num_channels=num_channels, use_bias=True, tied_biases=True, weights_init=IsotropicGaussian(0.01),image_size=image_size, name='conv_1'))
    layers.append(BatchNormalization(input_dim=layers[0].get_dim('output'), name='batchnorm_1'))
    layers.append(MaxPooling(pooling_size=(2,2), padding=(1,1), weights_init=IsotropicGaussian(0.01), name='maxpool_1'))
    layers.append(Convolutional(filter_size=filter_size, num_filters=num_filters*2, num_channels=num_channels, use_bias=True, tied_biases=True, weights_init=IsotropicGaussian(0.01),image_size=image_size, name='conv_2'))
    layers.append(BatchNormalization(input_dim=layers[3].get_dim('output'), name='batchnorm_2'))
    layers.append(MaxPooling(pooling_size=(2,2), padding=(1,1), weights_init=IsotropicGaussian(0.01), name='maxpool_2'))
    layers.append(Convolutional(filter_size=filter_size, num_filters=num_filters*4, num_channels=num_channels, use_bias=True, tied_biases=True, weights_init=IsotropicGaussian(0.01),image_size=image_size, name='conv_3'))
    layers.append(BatchNormalization(input_dim=layers[6].get_dim('output'), name='batchnorm_3'))
    layers.append(MaxPooling(pooling_size=(2,2), padding=(1,1), weights_init=IsotropicGaussian(0.01), name='maxpool_3'))
    layers.append(Convolutional(filter_size=filter_size, num_filters=num_filters*8, num_channels=num_channels, use_bias=True, tied_biases=True, weights_init=IsotropicGaussian(0.01),image_size=image_size, name='conv_4'))
    layers.append(BatchNormalization(input_dim=layers[9].get_dim('output'), name='batchnorm_4'))
    layers.append(MaxPooling(pooling_size=(2,2), padding=(1,1), weights_init=IsotropicGaussian(0.01), name='maxpool_4'))
    layers.append(Convolutional(filter_size=filter_size, num_filters=num_filters*16, num_channels=num_channels, use_bias=True, tied_biases=True, weights_init=IsotropicGaussian(0.01),image_size=image_size, name='conv_5'))
    layers.append(BatchNormalization(input_dim=layers[12].get_dim('output'), name='batchnorm_5'))
    layers.append(MaxPooling(pooling_size=(2,2), padding=(1,1), weights_init=IsotropicGaussian(0.01), name='maxpool_5'))
    return ConvolutionalSequence(layers, num_channels=num_channels, image_size=image_size, biases_init=Uniform(width=.1))


# images_train = tensor.dtensor4('images_train'); labels_train  = tensor.lmatrix('labels_train'); images_valid = tensor.dtensor4('images_valid'); labels_valid  = tensor.lmatrix('labels_valid')
def build_model(images, labels):
    
    # Construct a bottom convolutional sequence
    bottom_conv_sequence = convolutional_sequence((3,3), 16, (150, 150))
    bottom_conv_sequence._push_allocation_config()
    
    # Flatten layer
    flattener = Flattener()
    
    # Construct a top MLP
    conv_out_dim = numpy.prod(bottom_conv_sequence.get_dim('output'))
    top_mlp = MLP([Rectifier(), Softmax()], [conv_out_dim, 500, 10], weights_init=IsotropicGaussian(), biases_init=Constant(1))
    
    # Construct feedforward sequence
    ss_seq = FeedforwardSequence([bottom_conv_sequence.apply, flattener.apply, top_mlp.apply])
    ss_seq.push_initialization_config()
    ss_seq.initialize()
    
    prediction = ss_seq.apply(images)
    cost       = CategoricalCrossEntropy().apply(labels.flatten(), prediction)
    
    return cost
                  
    