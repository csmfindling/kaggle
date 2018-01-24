from __future__ import print_function

from lasagne.regularization import regularize_layer_params, l2
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer, batch_norm, get_output, ConcatLayer, LSTMLayer, get_all_params, DimshuffleLayer, FlattenLayer
from lasagne.nonlinearities import leaky_rectify, softmax, sigmoid, linear, rectify
from lasagne.objectives import squared_error, binary_crossentropy
from lasagne.updates import nesterov_momentum
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
import theano
import theano.tensor as T


def get_model(input_var, target_var, multiply_var):

    # input layer with unspecified batch size
    layer     = InputLayer(shape=(None, 12, 64, 64), input_var=input_var) #InputLayer(shape=(None, 1, 30, 64, 64), input_var=input_var)
    layer     = DimshuffleLayer(layer, (0, 'x', 1, 2, 3))

    # Z-score?

    # Convolution then batchNormalisation then activation layer, then zero padding layer followed by a dropout layer
    layer         = batch_norm(Conv3DDNNLayer(incoming=layer, num_filters=16, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=rectify))
    layer         = batch_norm(Conv3DDNNLayer(incoming=layer, num_filters=16, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=rectify))
    layer         = batch_norm(Conv3DDNNLayer(incoming=layer, num_filters=16, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=rectify))
    layer         = Conv3DDNNLayer(incoming=layer, num_filters=1, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=sigmoid)
    layer_prediction  = layer

    # Loss
    prediction           = get_output(layer_prediction)
    loss                 = binary_crossentropy(prediction[:,0,:,:,:], target_var).mean()

    #Updates : Stochastic Gradient Descent (SGD) with Nesterov momentum
    params               = get_all_params(layer_prediction, trainable=True)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network, disabling dropout layers.
    test_prediction      = get_output(layer_prediction, deterministic=True)
    test_loss            = binary_crossentropy(test_prediction[:,0,:,:,:], target_var).mean()

    return test_prediction, prediction, loss, params
