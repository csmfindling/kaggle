# source = https://github.com/Lasagne/Recipes/blob/master/modelzoo/vgg16.py


# VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
# License: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl

# for preprocessing : https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py

from config import basepath
import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
try:
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
except ImportError:
    from lasagne.layers import Conv2DLayer as ConvLayer

from lasagne.nonlinearities import softmax, sigmoid
import pickle
from theano import tensor, shared, config
import numpy

def build_model(input_var):
    net = {}
    layers_names = {}
    net['mean value'] = shared(numpy.zeros((3,), dtype=config.floatX))
    X = input_var * numpy.float32(255)
    X -= net['mean value'][None, :, None, None]

    net['input'] = InputLayer((None, 3, 256, 256), X)
    layers_names[0] = 'input'
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1, flip_filters=False)
    layers_names[1] = 'conv1_1'
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    layers_names[2] = 'conv1_2'
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    layers_names[3] = 'pool1'
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=False)
    layers_names[4] = 'conv2_1'
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    layers_names[5] = 'conv2_2'
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    layers_names[6] = 'pool2'
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=False)
    layers_names[7] = 'conv3_1'
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    layers_names[8] = 'conv3_2'
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    layers_names[9] = 'conv3_3'
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    layers_names[10] = 'pool3'
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=False)
    layers_names[11] = 'conv4_1'
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    layers_names[12] = 'conv4_2'
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    layers_names[13] = 'conv4_3'
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    layers_names[14] = 'pool4'
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, flip_filters=False)
    layers_names[15] = 'conv5_1'
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    layers_names[16] = 'conv5_2'
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    layers_names[17] = 'conv5_3'
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    layers_names[18] = 'pool5'
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    layers_names[19] = 'fc6'
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    layers_names[20] = 'fc6_dropout'
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    layers_names[21] = 'fc7'
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    layers_names[22] = 'fc7_dropout'
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=17, nonlinearity=None)
    layers_names[23] = 'fc8'
    net['prob'] = NonlinearityLayer(net['fc8'], sigmoid)
    layers_names[24] = 'prob'

    return net, layers_names

def get_model(input_var):
    net, layers_names = build_model(input_var)

    f_pretrained     = open(basepath + 'vgg16.pkl')
    model_pretrained = pickle.load(f_pretrained)
    w_pretrained     = model_pretrained['param values']
    net['mean value'].set_value(model_pretrained['mean value'].astype(config.floatX))

    count = -2
    for layer_idx in range(0,len(layers_names)):
        if len(net[layers_names[layer_idx]].get_params()) == 2:
            try:
                count += 2
                if not layers_names[layer_idx].startswith('fc'):
                    net[layers_names[layer_idx]].W.set_value(w_pretrained[count], borrow=True)
                    net[layers_names[layer_idx]].b.set_value(w_pretrained[count + 1], borrow=True)  
            except:
                print layers_names[layer_idx]
                pass
    print('The weights of {0} layers were updated'.format((count+2)/2))
    prediction_train = lasagne.layers.get_output(net['prob'])
    prediction_test = lasagne.layers.get_output(net['prob'], deterministic=True)
    params = lasagne.layers.get_all_params(net['prob'], trainable=True)

    return prediction_train, prediction_test, params

if __name__=="__main__":
    input_var = tensor.tensor4('X')
    get_model(input_var)
