import theano
from theano import tensor, config
from blocks.bricks import BatchNormalization, Rectifier, Linear, Softmax, MLP, BatchNormalizedMLP, FeedforwardSequence, Rectifier
from blocks.bricks.conv import Convolutional, ConvolutionalSequence, Flattener, MaxPooling
from blocks.initialization import IsotropicGaussian, Uniform, Constant
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.select import Selector
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.roles import PARAMETER, WEIGHT, BIAS

import numpy

from feature_extraction import VGG, SubstractBatch

from sklearn_theano.feature_extraction.caffe.balanced_vgg import create_theano_expressions


def build_model(images, labels):
    
    vgg = VGG(layer='conv3_4')
    vgg.push_initialization_config()
    vgg.initialize()

    sb = SubstractBatch()

    # Construct a bottom convolutional sequence
    layers = [
             Convolutional(filter_size=(3,3), num_filters=100, use_bias=True, tied_biases=True, name='final_conv0'),
             BatchNormalization(name='batchnorm_1'),
             Rectifier(name='final_conv0_act'),
             Convolutional(filter_size=(3,3), num_filters=100, use_bias=True, tied_biases=True, name='final_conv1'),
             BatchNormalization(name='batchnorm_2'),
             Rectifier(name='final_conv1_act'),
             MaxPooling(pooling_size=(2,2), name='maxpool_final')
             ]
    bottom_conv_sequence = ConvolutionalSequence(layers, num_channels=256, image_size=(40, 40), biases_init=Constant(0.), weights_init=IsotropicGaussian(0.01))
    bottom_conv_sequence._push_allocation_config()
    
    # Flatten layer
    flattener = Flattener()

    # Construct a top MLP
    conv_out_dim = numpy.prod(bottom_conv_sequence.get_dim('output'))
    print 'dim output conv:', bottom_conv_sequence.get_dim('output')
    # conv_out_dim = 20 * 40 * 40
    top_mlp = BatchNormalizedMLP([Rectifier(name='non_linear_9'), Softmax(name='non_linear_11')], [conv_out_dim, 1024, 10], weights_init=IsotropicGaussian(), biases_init=Constant(0))
    
    # Construct feedforward sequence
    ss_seq = FeedforwardSequence([vgg.apply, bottom_conv_sequence.apply, flattener.apply, top_mlp.apply])
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
    cost = cost_noreg + .0001 * (W0 ** 2).sum() + .001 * (W1 ** 2).sum()


    # define learned parameters
    selector = Selector([ss_seq])
    Ws = selector.get_parameters('W')
    bs = selector.get_parameters('b')
    BNSCs = selector.get_parameters('batch_norm_scale')
    BNSHs = selector.get_parameters('batch_norm_shift')

    parameters_top = []
    parameters_top += [v for k, v in Ws.items()]
    parameters_top += [v for k, v in bs.items()]
    parameters_top += [v for k, v in BNSCs.items()]
    parameters_top += [v for k, v in BNSHs.items()]

    selector = Selector([vgg])
    convs = selector.get_parameters()

    parameters_all = []
    parameters_all += parameters_top
    parameters_all += [v for k, v in convs.items()]

    return cost, [parameters_top, parameters_all]
                  
    
