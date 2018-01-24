from __future__ import print_function

from lasagne.regularization import regularize_layer_params, l2
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, DropoutLayer, FlattenLayer, DenseLayer, batch_norm, get_output, get_all_params, ConcatLayer
from lasagne.nonlinearities import leaky_rectify, sigmoid, linear
from lasagne.objectives import squared_error
from lasagne.updates import nesterov_momentum
import theano
import theano.tensor as T

def center_normalize(x):
    """
    Custom activation for online sample-wise center and std. normalization
    """
    return (x - K.mean(x)) / K.std(x)


def get_model(input_var, target_var, multiply_var):

    # input layer with unspecified batch size
    layer_both_0         = InputLayer(shape=(None, 30, 64, 64), input_var=input_var)

    # Z-score?

    # Convolution then batchNormalisation then activation layer, twice, then zero padding layer followed by a dropout layer
    layer_both_1         = batch_norm(Conv2DLayer(layer_both_0, 64, (3, 3), pad='same', nonlinearity=leaky_rectify))
    layer_both_2         = batch_norm(Conv2DLayer(layer_both_1, 64, (3, 3), pad='same', nonlinearity=leaky_rectify))
    layer_both_3         = MaxPool2DLayer(layer_both_2, pool_size=(2, 2), stride=(2, 2), pad=(1, 1))
    layer_both_4         = DropoutLayer(layer_both_3, p=0.25)

    # Convolution then batchNormalisation then activation layer, twice, then zero padding layer followed by a dropout layer
    layer_both_5         = batch_norm(Conv2DLayer(layer_both_4, 128, (3, 3), pad='same', nonlinearity=leaky_rectify))
    layer_both_6         = batch_norm(Conv2DLayer(layer_both_5, 128, (3, 3), pad='same', nonlinearity=leaky_rectify))
    layer_both_7         = MaxPool2DLayer(layer_both_6, pool_size=(2, 2), stride=(2, 2), pad=(1, 1))
    layer_both_8         = DropoutLayer(layer_both_7, p=0.25)

    # Convolution then batchNormalisation then activation layer, twice, then zero padding layer followed by a dropout layer
    layer_both_9         = batch_norm(Conv2DLayer(layer_both_8, 256, (3, 3), pad='same', nonlinearity=leaky_rectify))
    layer_both_10        = batch_norm(Conv2DLayer(layer_both_9, 256, (3, 3), pad='same', nonlinearity=leaky_rectify))
    layer_both_11        = batch_norm(Conv2DLayer(layer_both_10, 256, (3, 3), pad='same', nonlinearity=leaky_rectify))
    layer_both_12        = MaxPool2DLayer(layer_both_11, pool_size=(2, 2), stride=(2, 2), pad=(1, 1))
    layer_both_13        = DropoutLayer(layer_both_12, p=0.25)

    # Flatten
    layer_flatten        = FlattenLayer(layer_both_13)

    # Prediction
    layer_hidden         = DenseLayer(layer_flatten, 500, nonlinearity=sigmoid)
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










