import theano
from theano import tensor, config
from blocks.bricks import BatchNormalization, Rectifier, Linear, Softmax, MLP, BatchNormalizedMLP, FeedforwardSequence, Rectifier
from blocks.bricks.conv import Convolutional, ConvolutionalSequence, Flattener, MaxPooling
from blocks.initialization import IsotropicGaussian, Uniform, Constant
from blocks.select import Selector
from blocks.graph import ComputationGraph, apply_dropout
from blocks.filter import VariableFilter
from blocks.roles import OUTPUT
import numpy

from elementary_blocks_simple import VGG, top_direction_block, StructuredCost

images = tensor.ftensor4('images')
labels = tensor.ftensor4('labels')

def build_model(images, labels):
    
    vgg = VGG(layer='conv4_4')
    vgg.push_initialization_config()
    vgg.initialize()

    tdb = top_direction_block()
    tdb.push_initialization_config()
    tdb.initialize()

    # Construct feedforward sequence
    ss_seq = FeedforwardSequence([vgg.apply, tdb.apply])
    ss_seq.push_initialization_config()
    ss_seq.initialize()
    
    prediction = ss_seq.apply(images)
    cost       = StructuredCost().apply(labels, theano.tensor.clip(prediction, 1e-5, 1 - 1e-5))

    cg           = ComputationGraph(cost)
    cg_dropout   = apply_dropout(cg, [VariableFilter(roles=[OUTPUT])(cg.variables)[0]], .5)
    cost_dropout = cg_dropout.outputs[0]

    # define learned parameters
    selector = Selector([ss_seq])
    W         = selector.get_parameters()
    parameters = []
    parameters += [v for k, v in W.items()]

    return cost_dropout, parameters 


    
