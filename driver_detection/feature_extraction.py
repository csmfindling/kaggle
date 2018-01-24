from sklearn_theano.feature_extraction.caffe.balanced_vgg import create_theano_expressions
from blocks.bricks import Brick, Feedforward, application, Initializable
from blocks.graph import ComputationGraph
from blocks.roles import add_role, WEIGHT

import theano
from theano import config, tensor

class VGG(Feedforward, Initializable):

    def __init__(self, layer='conv4_4', **kwargs):
        super(VGG, self).__init__(**kwargs)

        self.layer_name = layer
        self.parameters = []

    @application
    def apply(self, images, application_call):

        images = images[:,::-1,:,:] # convert to bgr
        images = images * 256.
        images -= tensor.as_tensor_variable([103.939, 116.779, 123.68])[None,:,None,None]
        images = images.astype(config.floatX)

        layers_features, data_input = create_theano_expressions()

        output = theano.clone(self.layers_features[self.layer_name], replace={self.data_input: images})

        return output


    def _initialize(self):
        self.layers_features, self.data_input = create_theano_expressions()

        cg = ComputationGraph(self.layers_features[self.layer_name])
        i = 0
        for v in cg.shared_variables:
            v.name = str(i)
            self.parameters.append(v)
            add_role(v, WEIGHT)
            i += 1


class SubstractBatch(Feedforward):

    @application
    def apply(self, images, application_call):

        mean = images.mean(axis=0, keepdims=True)
        return images - mean
