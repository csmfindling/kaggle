import theano
from theano import tensor, config
from blocks.bricks import Brick, Feedforward, application, Initializable, BatchNormalization, Rectifier, Linear, Softmax, MLP, BatchNormalizedMLP, FeedforwardSequence, Rectifier, Logistic
from blocks.graph import ComputationGraph, apply_dropout
from blocks.roles import add_role, WEIGHT, COST, PARAMETER
from blocks.bricks.conv import Convolutional, ConvolutionalSequence, Flattener, MaxPooling
from blocks.initialization import IsotropicGaussian, Uniform, Constant
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.select import Selector
import numpy
from sklearn_theano.feature_extraction.caffe.balanced_vgg import create_theano_expressions
from blocks.bricks.cost import Cost, BinaryCrossEntropy
from theano.tensor.nnet.nnet import softmax
from theano.sandbox.rng_mrg import MRG_RandomStreams


srng = MRG_RandomStreams(seed=234)

class StructuredCost(Cost):

    @application(outputs=["cost"])
    def apply(self, y, y_hat):
        epsilon    = 1e-5 # to avoid nan
        mask = (tensor.le(srng.uniform(size=y[:,-1:].shape, dtype=config.floatX), .0005))*1.
        cost = 0
        for i in range(15):
            cost +=  tensor.nnet.binary_crossentropy(y_hat[:,i,:,:], y[:,i,:,:]).mean()

        cost += ( tensor.nnet.binary_crossentropy(y_hat[:,15,:,:], tensor.eq(y[:,15,:,:],1)*1.) * mask ).mean()
        return cost

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

        #layers_features, data_input = create_theano_expressions()

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

class ParallelSum2(Initializable):

    @application(inputs=['input1_', 'input2_'], outputs=['output'])
    def apply(self, input1_, input2_):
         output = tensor.add(input1_, input2_)
         return output

class ParallelSum3(Initializable):

    @application(inputs=['input1_', 'input2_', 'input3_'], outputs=['output'])
    def apply(self, input1_, input2_, input3_):
         output = tensor.add(input1_, input2_, input3_)
         return output

class top_direction_block(Initializable, Feedforward):

    def __init__(self, **kwargs):

        children = []

        self.list_simple_joints = [19, 20, 8, 7] + range(16,19)[::-1] + range(4, 7)[::-1] + [1] + [15, 3, 2] + [0]

        self.simple_joints = {}
        for simple_joint_idx in self.list_simple_joints:
            self.simple_joints[simple_joint_idx] = [Convolutional(filter_size=(1,1), num_filters = 512, border_mode = (0,0), use_bias=True, tied_biases=True, name='fconv_' + str(simple_joint_idx), biases_init=Constant(0.), weights_init=IsotropicGaussian(0.01), num_channels = 512), \
                                                    Rectifier(name='fconv_relu' + str(simple_joint_idx)), \
                                                    Convolutional(filter_size=(1,1), num_filters = 1, border_mode = (0,0), use_bias=True, tied_biases=True, name='fconv1_' + str(simple_joint_idx), biases_init=Constant(0.), weights_init=IsotropicGaussian(0.01), num_channels = 512), \
                                                    Logistic(name = 'flogistic_' + str(simple_joint_idx))]
            children += self.simple_joints[simple_joint_idx]

        kwargs.setdefault('children', []).extend(children)
        super(top_direction_block, self).__init__(**kwargs)
        

    def __allocate(self, conv_layer, number):
        conv_layer.allocate(); [W,b] = conv_layer.parameters;
        W.name = 'W' + str(number); b.name = 'b' + str(number)
        add_role(W, WEIGHT); add_role(b, WEIGHT)
        #self.parameters.append(W); self.parameters.append(b)

    def _allocate(self):
        for k in self.simple_joints.keys():
            self.__allocate(self.simple_joints[k][0], int(str(k)+str(0))); self.__allocate(self.simple_joints[k][2], int(str(k)+str(2)));

    def _initialize(self):
        for k in self.simple_joints.keys():
            self.simple_joints[k][0].initialize(); self.simple_joints[k][2].initialize();

    @application
    def apply(self, inputs):

        # do simple joints
        outputs_conv_simple_joints  = {}; outputs_relu_simple_joints = {}; output_conv2_simple_joints = {}

        outputs_tree = []
        for k in self.list_simple_joints:
            outputs_conv_simple_joints[k] = self.simple_joints[k][0].apply(inputs)
            outputs_relu_simple_joints[k] = self.simple_joints[k][1].apply(outputs_conv_simple_joints[k])
            output_conv2_simple_joints[k] = self.simple_joints[k][2].apply(outputs_relu_simple_joints[k])
            outputs_tree.append(self.simple_joints[k][3].apply(output_conv2_simple_joints[k]))
        
        assert(len(outputs_relu_simple_joints) == len(self.simple_joints))
        # concatenate
        output = tensor.concatenate(outputs_tree, axis=1)

        return output
