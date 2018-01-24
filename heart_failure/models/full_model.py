from __future__ import print_function

from lasagne.regularization import regularize_layer_params, l2
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer, batch_norm, get_output, ConcatLayer, LSTMLayer, get_all_params, DimshuffleLayer, FlattenLayer, ElemwiseSumLayer, ExpressionLayer
from lasagne.nonlinearities import leaky_rectify, softmax, sigmoid, linear, rectify
from lasagne.objectives import squared_error, binary_crossentropy
from lasagne.updates import nesterov_momentum
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
import theano
import theano.tensor as T

def dist(a, b):
    return numpy.sqrt(((a-b)**2).sum())

def get_model(input_images, input_position, input_mult, target_var):

    # number of SAX and distance between SAX slices
    #indexes = []
    #for i in range(input_position.shape[0]):
    #    indexes.append(numpy.where(input_position[i][:,0] == 0.)[0][0])
    
    # input layer with unspecified batch size
    layer     = InputLayer(shape=(None, 22, 30, 64, 64), input_var=input_images) #InputLayer(shape=(None, 1, 30, 64, 64), input_var=input_var)
    
    # Z-score?

    # Convolution then batchNormalisation then activation layer, then zero padding layer followed by a dropout layer
    layer         = batch_norm(Conv3DDNNLayer(incoming=layer, num_filters=16, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=rectify))
    shortcut      = layer
    layer         = batch_norm(Conv3DDNNLayer(incoming=layer, num_filters=16, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=rectify))
    layer         = batch_norm(Conv3DDNNLayer(incoming=layer, num_filters=16, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=rectify))
    layer 	  = ElemwiseSumLayer([layer, shortcut])
    shortcut      = layer
    layer         = batch_norm(Conv3DDNNLayer(incoming=layer, num_filters=16, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=rectify))
    layer         = batch_norm(Conv3DDNNLayer(incoming=layer, num_filters=16, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=rectify))
    layer 	  = ElemwiseSumLayer([layer, shortcut])
    shortcut      = layer
    layer         = batch_norm(Conv3DDNNLayer(incoming=layer, num_filters=16, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=rectify))
    layer         = batch_norm(Conv3DDNNLayer(incoming=layer, num_filters=16, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=rectify))
    layer 	  = ElemwiseSumLayer([layer, shortcut])
    shortcut      = layer
    layer         = batch_norm(Conv3DDNNLayer(incoming=layer, num_filters=16, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=rectify))
    layer         = batch_norm(Conv3DDNNLayer(incoming=layer, num_filters=16, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=rectify))
    layer 	  = ElemwiseSumLayer([layer, shortcut])
    shortcut      = layer
    layer         = batch_norm(Conv3DDNNLayer(incoming=layer, num_filters=16, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=rectify))
    layer         = batch_norm(Conv3DDNNLayer(incoming=layer, num_filters=16, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=rectify))
    layer 	  = ElemwiseSumLayer([layer, shortcut])
    shortcut      = layer
    layer         = batch_norm(Conv3DDNNLayer(incoming=layer, num_filters=16, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=rectify))
    layer         = batch_norm(Conv3DDNNLayer(incoming=layer, num_filters=16, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=rectify))
    layer 	  = ElemwiseSumLayer([layer, shortcut])
    layer         = batch_norm(Conv3DDNNLayer(incoming=layer, num_filters=16, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=rectify))
    layer         = Conv3DDNNLayer(incoming=layer, num_filters=22, filter_size=(3,3,3), stride=(1,1,1), pad='same', nonlinearity=sigmoid)

    layer_max     = ExpressionLayer(layer, lambda X: X.max(1), output_shape='auto')
    layer_min     = ExpressionLayer(layer, lambda X: X.min(1), output_shape='auto')
    
    layer_prediction = layer
    # image prediction
    prediction           = get_output(layer_prediction)
        
    loss                 = binary_crossentropy(prediction, target_var).mean()

    #Updates : Stochastic Gradient Descent (SGD) with Nesterov momentum
    params               = get_all_params(layer_prediction, trainable=True)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network, disabling dropout layers.
    test_prediction      = get_output(layer_prediction, deterministic=True)
    test_loss            = binary_crossentropy(test_prediction, target_var).mean()

    return test_prediction, prediction, loss, params

"""    distance = []
    for i in range(input_position.shape[0]):
        distance_per_case = []
        for j in range(indexes[i] - 1):
            distance_per_case.append(dist(input_position[i][j+1], input_position[i][j]))
        distance.append(distance_per_case)
    
    # Maximum and minimum surface images
    max_surface_images = []
    min_surface_images = []
    for i in range(input_position.shape[0]):
        max_surface_images_per_case = []
        min_surface_images_per_case = []
        for j in range(30):
            argmax        = numpy.argmax(numpy.sum(prediction[i,:,j], axis=(1,2)))
            maximum_image = prediction[i,argmax,j]
            argmin = numpy.argmin(numpy.array([numpy.sum(prediction[i,k,j] * maximum_image) for k in range(indexes[i])]))
            max_surface_images_per_case.append(maximum_image)
            min_surface_images_per_case.append(prediction[i, argmin, j])"""
