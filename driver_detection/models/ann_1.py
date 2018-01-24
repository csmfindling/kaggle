from __future__ import print_function

import theano
from lasagne.regularization import regularize_layer_params, l2
from lasagne.layers import InputLayer, DenseLayer, batch_norm, get_output, get_all_params, Conv2DLayer, FlattenLayer, ElemwiseSumLayer, DimshuffleLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.objectives import categorical_crossentropy, squared_error

def build_model(images, labels):
    
    layer_input = InputLayer(shape=(None, 120, 120, 3), input_var=images)
    layer       = DimshuffleLayer(layer_input, (0, 3, 1, 2))

    # Convolution then batchNormalisation then activation layer, then zero padding layer followed by a dropout layer
    layer         = batch_norm(Conv2DLayer(incoming=layer, num_filters=16, filter_size=(3,3), stride=(1,1), pad='same', nonlinearity=rectify))
    shortcut      = layer
    layer         = batch_norm(Conv2DLayer(incoming=layer, num_filters=16, filter_size=(3,3), stride=(1,1), pad='same', nonlinearity=rectify))
    layer         = batch_norm(Conv2DLayer(incoming=layer, num_filters=16, filter_size=(3,3), stride=(1,1), pad='same', nonlinearity=rectify))
    layer     = ElemwiseSumLayer([layer, shortcut])
    shortcut      = layer
    layer         = batch_norm(Conv2DLayer(incoming=layer, num_filters=16, filter_size=(3,3), stride=(1,1), pad='same', nonlinearity=rectify))
    layer         = batch_norm(Conv2DLayer(incoming=layer, num_filters=16, filter_size=(3,3), stride=(1,1), pad='same', nonlinearity=rectify))
    layer     = ElemwiseSumLayer([layer, shortcut])
    shortcut      = layer
    layer         = batch_norm(Conv2DLayer(incoming=layer, num_filters=16, filter_size=(3,3), stride=(1,1), pad='same', nonlinearity=rectify))
    layer         = batch_norm(Conv2DLayer(incoming=layer, num_filters=16, filter_size=(3,3), stride=(1,1), pad='same', nonlinearity=rectify))
    layer     = ElemwiseSumLayer([layer, shortcut])
    shortcut      = layer
    layer         = batch_norm(Conv2DLayer(incoming=layer, num_filters=16, filter_size=(3,3), stride=(1,1), pad='same', nonlinearity=rectify))
    layer         = batch_norm(Conv2DLayer(incoming=layer, num_filters=16, filter_size=(3,3), stride=(1,1), pad='same', nonlinearity=rectify))
    layer     = ElemwiseSumLayer([layer, shortcut])
    shortcut      = layer
    layer         = batch_norm(Conv2DLayer(incoming=layer, num_filters=16, filter_size=(3,3), stride=(1,1), pad='same', nonlinearity=rectify))
    layer         = batch_norm(Conv2DLayer(incoming=layer, num_filters=16, filter_size=(3,3), stride=(1,1), pad='same', nonlinearity=rectify))
    layer     = ElemwiseSumLayer([layer, shortcut])
    shortcut      = layer
    layer         = batch_norm(Conv2DLayer(incoming=layer, num_filters=16, filter_size=(3,3), stride=(1,1), pad='same', nonlinearity=rectify))
    layer         = batch_norm(Conv2DLayer(incoming=layer, num_filters=16, filter_size=(3,3), stride=(1,1), pad='same', nonlinearity=rectify))
    layer     = ElemwiseSumLayer([layer, shortcut])
    layer         = batch_norm(Conv2DLayer(incoming=layer, num_filters=16, filter_size=(3,3), stride=(1,1), pad='same', nonlinearity=rectify))
    
    layer        = FlattenLayer(layer)
    layer        = DenseLayer(layer, num_units=10, nonlinearity=softmax)

    layer_prediction  = layer

    prediction           = get_output(layer_prediction)

    loss                 = categorical_crossentropy(prediction, labels).mean()

    params               = get_all_params(layer_prediction, trainable=True)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network, disabling dropout layers.
    test_prediction      = get_output(layer_prediction, deterministic=True)
    test_loss            = squared_error(test_prediction, labels).mean()

    return test_prediction, test_loss, prediction, loss, params


