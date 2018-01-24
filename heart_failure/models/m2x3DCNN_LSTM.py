from __future__ import print_function

from lasagne.regularization import regularize_layer_params, l2
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer, batch_norm, get_output, ConcatLayer, LSTMLayer, get_all_params, DimshuffleLayer
from lasagne.nonlinearities import leaky_rectify, softmax, sigmoid, linear
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
    layer_input     = InputLayer(shape=(None, 30, 80, 80), input_var=input_var) #InputLayer(shape=(None, 1, 30, 64, 64), input_var=input_var)
    layer_0         = DimshuffleLayer(layer_input, (0, 'x', 1, 2, 3))

    # Z-score?

    # Convolution then batchNormalisation then activation layer, then zero padding layer followed by a dropout layer
    layer_1         = batch_norm(Conv3DDNNLayer(incoming=layer_0, num_filters=64, filter_size=(3,3,3), stride=(1,3,3), pad='same', nonlinearity=leaky_rectify))
    layer_2         = MaxPool3DDNNLayer(layer_1, pool_size=(1, 2, 2), stride=(1, 2, 2), pad=(0, 1, 1))
    layer_3         = DropoutLayer(layer_2, p=0.25)

    # Convolution then batchNormalisation then activation layer, then zero padding layer followed by a dropout layer
    layer_4         = batch_norm(Conv3DDNNLayer(incoming=layer_3, num_filters=128, filter_size=(3,3,3), stride=(1,3,3), pad='same', nonlinearity=leaky_rectify))
    layer_5         = MaxPool3DDNNLayer(layer_4, pool_size=(1, 2, 2), stride=(1, 2, 2), pad=(0, 1, 1))
    layer_6         = DropoutLayer(layer_5, p=0.25)

    # Recurrent layer
    layer_7         = DimshuffleLayer(layer_6, (0,2,1,3,4))
    layer_8         = LSTMLayer(layer_7, num_units=612, only_return_final=True)
    layer_9         = DropoutLayer(layer_8, p=0.25)

    # Output Layer
    layer_hidden         = DenseLayer(layer_9, 500, nonlinearity=sigmoid)
    layer_prediction     = DenseLayer(layer_hidden, 2, nonlinearity=linear)

    # Loss
    prediction           = get_output(layer_prediction) / multiply_var
    loss                 = squared_error(prediction, target_var)
    loss                 = loss.mean()

    #Updates : Stochastic Gradient Descent (SGD) with Nesterov momentum
    params               = get_all_params(layer_prediction, trainable=True)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network, disabling dropout layers.
    test_prediction      = get_output(layer_prediction, deterministic=True) / multiply_var
    test_loss            = squared_error(test_prediction, target_var)
    test_loss            = test_loss.mean()

    # crps estimate
    crps                 = T.abs_(test_prediction - target_var).mean()/600

    return test_prediction, crps, loss, params
