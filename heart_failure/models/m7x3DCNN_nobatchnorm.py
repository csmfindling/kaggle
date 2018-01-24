from __future__ import print_function

from lasagne.regularization import regularize_layer_params, l2
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer,  get_output, ConcatLayer, LSTMLayer, get_all_params, DimshuffleLayer, FlattenLayer
from lasagne.nonlinearities import rectify, softmax, sigmoid, linear
from lasagne.objectives import squared_error
from lasagne.updates import nesterov_momentum
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
import theano
import theano.tensor as T


def center_normalize(x):
    """
    Custom activation for online sample-wise center and std. normalization
    """
    return (x - K.mean(x)) / K.std(x)


def get_model(input_var, target_var, multiply_var):

    # input layer with unspecified batch size
    layer_input     = InputLayer(shape=(None, 30, 64, 64), input_var=input_var) #InputLayer(shape=(None, 1, 30, 64, 64), input_var=input_var)
    layer_0         = DimshuffleLayer(layer_input, (0, 'x', 1, 2, 3))

    # Z-score?

    # Convolution then batchNormalisation then activation layer, then zero padding layer followed by a dropout layer
    layer_1         = Conv3DDNNLayer(incoming=layer_0, num_filters=16, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=rectify)
    layer_2         = Conv3DDNNLayer(incoming=layer_1, num_filters=16, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=rectify)
    layer_3         = MaxPool3DDNNLayer(layer_2, pool_size=(2, 2, 2), stride=(2, 2, 2), pad=(1, 1, 1))
    layer_4         = DropoutLayer(layer_3, p=0.25)

    # Convolution then batchNormalisation then activation layer, then zero padding layer followed by a dropout layer
    layer_5         = Conv3DDNNLayer(incoming=layer_4, num_filters=32, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=rectify)
    layer_6         = Conv3DDNNLayer(incoming=layer_5, num_filters=32, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=rectify)
    layer_7         = MaxPool3DDNNLayer(layer_6, pool_size=(2, 2, 2), stride=(2, 2, 2), pad=(1, 1, 1))
    layer_8         = DropoutLayer(layer_7, p=0.25)
    
    # Convolution then batchNormalisation then activation layer, then zero padding layer followed by a dropout layer
    layer_5         = Conv3DDNNLayer(incoming=layer_8, num_filters=64, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=rectify)
    layer_6         = Conv3DDNNLayer(incoming=layer_5, num_filters=64, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=rectify)
    layer_7         = Conv3DDNNLayer(incoming=layer_6, num_filters=64, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=rectify)
    layer_8         = MaxPool3DDNNLayer(layer_7, pool_size=(2, 2, 2), stride=(2, 2, 2), pad=(1, 1, 1))
    layer_9         = DropoutLayer(layer_8, p=0.25)

    layer_flatten   = FlattenLayer(layer_9)

    # Output Layer
    layer_hidden         = DenseLayer(layer_flatten, 500, nonlinearity=rectify)
    layer_prediction     = DenseLayer(layer_hidden, 2, nonlinearity=linear)

    # Loss
    prediction           = get_output(layer_prediction) / multiply_var**2
    loss                 = T.abs_(prediction - target_var).mean()

    #Updates : Stochastic Gradient Descent (SGD) with Nesterov momentum
    params               = get_all_params(layer_prediction, trainable=True)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network, disabling dropout layers.
    test_prediction      = get_output(layer_prediction, deterministic=True) / multiply_var**2
    test_loss            = squared_error(test_prediction, target_var)
    test_loss            = test_loss.mean()

    # crps estimate
    crps                 = T.abs_(test_prediction - target_var).mean()/600

    return test_prediction, crps, loss, params
