import theano
from theano import tensor, config
from blocks.bricks import Brick, Feedforward, application, Initializable, BatchNormalization, Rectifier, Linear, Softmax, MLP, BatchNormalizedMLP, FeedforwardSequence, Rectifier
from blocks.graph import ComputationGraph, apply_dropout
from blocks.roles import add_role, WEIGHT, COST, PARAMETER
from blocks.bricks.conv import Convolutional, ConvolutionalSequence, Flattener, MaxPooling
from blocks.initialization import IsotropicGaussian, Uniform, Constant
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.select import Selector
import numpy
from sklearn_theano.feature_extraction.caffe.balanced_vgg import create_theano_expressions
from blocks.bricks.cost import Cost
from theano.tensor.nnet.nnet import softmax
from theano.sandbox.rng_mrg import MRG_RandomStreams


srng = MRG_RandomStreams(seed=234)

class StructuredCost(Cost):

    @application(outputs=["cost"])
    def apply(self, y, y_hat):
        epsilon = 1e-20 # to avoid nan
        e_y_hat  = tensor.exp(y_hat - y_hat.max(axis=1, keepdims=True))
        out      = tensor.log(e_y_hat / e_y_hat.sum(axis=1, keepdims=True) + epsilon) * y
        background = tensor.addbroadcast(y[:,-1:], 1)
        out      = ((srng.uniform(size=y.shape, dtype=config.floatX) < .0005) * background + (1 - background)) * out # problem
        cost     = out.sum(axis=0).mean()
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

        self.list_simple_joints = [19, 20, 8, 7]#[19, 26, 14, 7]
        self.list_double_joints = range(16,19)[::-1] + range(4, 7)[::-1] + [1]
        self.list_triple_joints = [15, 3, 2]
        self.mapping       = {19:18, 18:17, 17:16, 16:15, 26:25, 25:24, 24:23, 23:22, 22:21, 21:20, 20:15, 15:2, \
                    14:13, 13:12, 12:11, 11:10, 10:9, 9:8, 8:3, 7:6, 6:5, 5:4, 4:3, 3:2, 2:1, 1:0}
        self.rev_mapping       = {18:19, 17:18, 16:17, 15:[16, 20], 25:26, 24:25, 23:24, 22:23, 21:22, 20:21, 2:[15, 3], \
                    13:14, 12:13, 11:12, 10:11, 9:10, 8:9, 6:7, 5:6, 4:5, 3:[4, 8], 1:2, 0:1}

        assert(len(self.list_simple_joints) + len(self.list_double_joints) + len(self.list_triple_joints) == 14)

        self.simple_joints = {}
        for simple_joint_idx in self.list_simple_joints:
            self.simple_joints[simple_joint_idx] = self.generate_elementary_block1(simple_joint_idx, self.mapping[simple_joint_idx])
            children += self.simple_joints[simple_joint_idx]

        self.double_joints = {}
        for double_joint_idx in self.list_double_joints:
            if double_joint_idx != 1:
                self.double_joints[double_joint_idx] = self.generate_elementary_block2(double_joint_idx, self.mapping[double_joint_idx])
            else:
                self.double_joints[double_joint_idx] = self.generate_elementary_block4(double_joint_idx)
            children += self.double_joints[double_joint_idx]

        self.triple_joints = {}
        for triple_joint_idx in self.list_triple_joints:
            self.triple_joints[triple_joint_idx] = self.generate_elementary_block3(triple_joint_idx, self.mapping[triple_joint_idx])
            children += self.triple_joints[triple_joint_idx]

        self.negative_conv = []
        self.negative_conv.append(Convolutional(filter_size=(1,1), num_filters = 512, border_mode = (0,0), use_bias=True, tied_biases=True, name='neg_conv', biases_init=Constant(0.), weights_init=IsotropicGaussian(0.01), num_channels = 512))
        self.negative_conv.append(Rectifier(name='neg_relu'))
        self.negative_conv.append(Convolutional(filter_size=(1,1), num_filters = 1, border_mode = (0,0), use_bias=True, tied_biases=True, name='neg_conv_0', biases_init=Constant(0.), weights_init=IsotropicGaussian(0.01), num_channels = 512))
        children += self.negative_conv

        kwargs.setdefault('children', []).extend(children)
        super(top_direction_block, self).__init__(**kwargs)
        
    def generate_elementary_block1(self, index, to_index):
        number_of_channels = 512
        name_conv_0        = 'fconv7_' + str(index)
        name_relu_0        = 'relu7_' + str(index)
        name_conv_1        = 'fconv7_' + str(index) + 'to' + str(to_index) + '_step1'
        name_relu_1        = 'relu7_' + str(index) + 'to' + str(to_index) + '_step1'
        name_conv_2        = 'fconv7_' + str(index) + 'to' + str(to_index) + '_step2'
        name_conv_3        = 'fconv7_output_' + str(index)
        return [Convolutional(filter_size=(1,1), num_filters = 128, border_mode = (0,0), use_bias=True, tied_biases=True, name=name_conv_0, biases_init=Constant(0.), weights_init=IsotropicGaussian(0.01), num_channels = number_of_channels), \
            Rectifier(name=name_relu_0), \
            Convolutional(filter_size=(7,7), num_filters = 64, border_mode = (3,3), use_bias=True, tied_biases=True, name=name_conv_1, biases_init=Constant(0.), weights_init=IsotropicGaussian(0.01), num_channels = 128), \
            Rectifier(name=name_relu_1), \
            Convolutional(filter_size=(7,7), num_filters = 128, border_mode = (3,3), use_bias=True, tied_biases=True, name=name_conv_2, biases_init=Constant(0.), weights_init=IsotropicGaussian(0.01), num_channels = 64), \
            Convolutional(filter_size=(1,1), num_filters = 1, border_mode = (0,0), use_bias=True, tied_biases=True, name=name_conv_3, biases_init=Constant(0.), weights_init=IsotropicGaussian(0.01), num_channels = 128)]

    def generate_elementary_block2(self, index, to_index):
        number_of_channels = 512
        name_conv_0        = 'fconv7_' + str(index)
        name_relu_0        = 'relu7_' + str(index)
        name_conv_1        = 'fconv7_' + str(index) + 'to' + str(to_index) + '_step1'
        name_relu_1        = 'relu7_' + str(index) + 'to' + str(to_index) + '_step1'
        name_conv_2        = 'fconv7_' + str(index) + 'to' + str(to_index) + '_step2'
        name_conv_3        = 'fconv7_output_' + str(index)
        return [Convolutional(filter_size=(1,1), num_filters = 128, border_mode = (0,0), use_bias=True, tied_biases=True, name=name_conv_0, biases_init=Constant(0.), weights_init=IsotropicGaussian(0.01), num_channels = number_of_channels), \
            ParallelSum2(), Rectifier(name=name_relu_0), \
            Convolutional(filter_size=(7,7), num_filters = 64, border_mode = (3,3), use_bias=True, tied_biases=True, name=name_conv_1, biases_init=Constant(0.), weights_init=IsotropicGaussian(0.01), num_channels = 128), \
            Rectifier(name=name_relu_1), \
            Convolutional(filter_size=(7,7), num_filters = 128, border_mode = (3,3), use_bias=True, tied_biases=True, name=name_conv_2, biases_init=Constant(0.), weights_init=IsotropicGaussian(0.01), num_channels = 64), \
            Convolutional(filter_size=(1,1), num_filters = 1, border_mode = (0,0), use_bias=True, tied_biases=True, name=name_conv_3, biases_init=Constant(0.), weights_init=IsotropicGaussian(0.01), num_channels = 128)]

    def generate_elementary_block3(self, index, to_index):
        number_of_channels = 512
        name_conv_0        = 'fconv7_' + str(index)
        name_relu_0        = 'relu7_' + str(index)
        name_conv_1        = 'fconv7_' + str(index) + 'to' + str(to_index) + '_step1'
        name_relu_1        = 'relu7_' + str(index) + 'to' + str(to_index) + '_step1'
        name_conv_2        = 'fconv7_' + str(index) + 'to' + str(to_index) + '_step2'
        name_conv_3        = 'fconv7_output_' + str(index)
        return [Convolutional(filter_size=(1,1), num_filters = 128, border_mode = (0,0), use_bias=True, tied_biases=True, name=name_conv_0, biases_init=Constant(0.), weights_init=IsotropicGaussian(0.01), num_channels = number_of_channels), \
            ParallelSum3(), Rectifier(name=name_relu_0), \
            Convolutional(filter_size=(7,7), num_filters = 64, border_mode = (3,3), use_bias=True, tied_biases=True, name=name_conv_1, biases_init=Constant(0.), weights_init=IsotropicGaussian(0.01), num_channels = 128), \
            Rectifier(name=name_relu_1), \
            Convolutional(filter_size=(7,7), num_filters = 128, border_mode = (3,3), use_bias=True, tied_biases=True, name=name_conv_2, biases_init=Constant(0.), weights_init=IsotropicGaussian(0.01), num_channels = 64), \
            Convolutional(filter_size=(1,1), num_filters = 1, border_mode = (0,0), use_bias=True, tied_biases=True, name=name_conv_3, biases_init=Constant(0.), weights_init=IsotropicGaussian(0.01), num_channels = 128)]

    def generate_elementary_block4(self, index):
        number_of_channels = 512
        name_conv_0        = 'fconv7_' + str(index)
        name_relu_0        = 'relu7_' + str(index)
        name_conv_3        = 'fconv7_output_' + str(index)
        return [Convolutional(filter_size=(1,1), num_filters = 128, border_mode = (0,0), use_bias=True, tied_biases=True, name=name_conv_0, biases_init=Constant(0.), weights_init=IsotropicGaussian(0.01), num_channels = number_of_channels), \
            ParallelSum2(), Rectifier(name=name_relu_0), \
            Convolutional(filter_size=(1,1), num_filters = 1, border_mode = (0,0), use_bias=True, tied_biases=True, name=name_conv_3, biases_init=Constant(0.), weights_init=IsotropicGaussian(0.01), num_channels = 128)]

    def __allocate(self, conv_layer, number):
        conv_layer.allocate(); [W,b] = conv_layer.parameters;
        W.name = 'W' + str(number); b.name = 'b' + str(number)
        add_role(W, WEIGHT); add_role(b, WEIGHT)
        #self.parameters.append(W); self.parameters.append(b)

    def _allocate(self):
        self.__allocate(self.negative_conv[0], 0), self.__allocate(self.negative_conv[2], 1000)
        for k in self.simple_joints.keys():
            self.__allocate(self.simple_joints[k][0], k); self.__allocate(self.simple_joints[k][2], int(str(k) + str(self.mapping[k]) + str(1))); self.__allocate(self.simple_joints[k][4], int(str(k) + str(self.mapping[k]) + str(2))); self.__allocate(self.simple_joints[k][5], int(str(k) + str(3)))
        for k in self.double_joints.keys():
            if k != 1:
                self.__allocate(self.double_joints[k][0], k); self.__allocate(self.double_joints[k][3], int(str(k) + str(self.mapping[k]) + str(1))); self.__allocate(self.double_joints[k][5], int(str(k) + str(self.mapping[k]) + str(2))); self.__allocate(self.double_joints[k][6], int(str(k) + str(3)))
            else:
                self.__allocate(self.double_joints[k][0], k); self.__allocate(self.double_joints[k][3], int(str(k) + str(3)))
        for k in self.triple_joints.keys():
            self.__allocate(self.triple_joints[k][0], k); self.__allocate(self.triple_joints[k][3], int(str(k) + str(self.mapping[k]) + str(1))); self.__allocate(self.triple_joints[k][5], int(str(k) + str(self.mapping[k]) + str(2))); self.__allocate(self.triple_joints[k][6], int(str(k) + str(3)))

    def _initialize(self):
        self.negative_conv[0].initialize(); self.negative_conv[2].initialize()
        for k in self.simple_joints.keys():
            self.simple_joints[k][0].initialize(); self.simple_joints[k][2].initialize(); self.simple_joints[k][4].initialize(); self.simple_joints[k][5].initialize()
        for k in self.double_joints.keys():
            if k != 1:
                self.double_joints[k][0].initialize(); self.double_joints[k][3].initialize(); self.double_joints[k][5].initialize(); self.double_joints[k][6].initialize()
            else:
                self.double_joints[k][0].initialize(); self.double_joints[k][3].initialize();
        for k in self.triple_joints.keys():
            self.triple_joints[k][0].initialize(); self.triple_joints[k][3].initialize(); self.triple_joints[k][5].initialize(); self.triple_joints[k][6].initialize()

    @application
    def apply(self, inputs):

        # do simple joints
        outputs_conv_simple_joints  = []; outputs_relu_simple_joints = []
        outputs_step1_simple_joints = []; outputs_step1relu_simple_joints = []
        outputs_step2_simple_joints = {}
        count = 0
        for k in self.list_simple_joints:
            outputs_conv_simple_joints.append(self.simple_joints[k][0].apply(inputs))
            outputs_relu_simple_joints.append(self.simple_joints[k][1].apply(outputs_conv_simple_joints[count]))
            outputs_step1_simple_joints.append(self.simple_joints[k][2].apply(outputs_relu_simple_joints[count]))
            outputs_step1relu_simple_joints.append(self.simple_joints[k][3].apply(outputs_step1_simple_joints[count]))
            outputs_step2_simple_joints[k] = self.simple_joints[k][4].apply(outputs_step1relu_simple_joints[count])
            count += 1

        # do double joints except for joint 1
        outputs_conv_double_joints = []; outputs_pass_double_joints = []
        outputs_relu_double_joints = []; outputs_step1_double_joints = []
        outputs_step1relu_double_joints = []; outputs_step2_double_joints = {}       
        count = 0
        for k in self.list_double_joints:
            if k != 1:
                outputs_conv_double_joints.append(self.double_joints[k][0].apply(inputs))
                if self.rev_mapping[k] in self.list_simple_joints:
                    outputs_pass_double_joints.append(self.double_joints[k][1].apply(outputs_conv_double_joints[count], outputs_step2_simple_joints[self.rev_mapping[k]]))
                else:
                    outputs_pass_double_joints.append(self.double_joints[k][1].apply(outputs_conv_double_joints[count], outputs_step2_double_joints[self.rev_mapping[k]]))
                outputs_relu_double_joints.append(self.double_joints[k][2].apply(outputs_pass_double_joints[count]))
                outputs_step1_double_joints.append(self.double_joints[k][3].apply(outputs_relu_double_joints[count]))
                outputs_step1relu_double_joints.append(self.double_joints[k][4].apply(outputs_step1_double_joints[count]))
                outputs_step2_double_joints[k] = self.double_joints[k][5].apply(outputs_step1relu_double_joints[count])
                count += 1

        # do triple joints
        outputs_conv_triple_joints = []; outputs_pass_triple_joints = []
        outputs_relu_triple_joints = []; outputs_step1_triple_joints = []
        outputs_step1relu_triple_joints = []; outputs_step2_triple_joints = {}       
        count = 0
        for k in self.list_triple_joints:
            outputs_conv_triple_joints.append(self.triple_joints[k][0].apply(inputs))
            if self.rev_mapping[k][0] in self.list_double_joints:
                outputs_pass_triple_joints.append(self.triple_joints[k][1].apply(outputs_conv_triple_joints[count], outputs_step2_double_joints[self.rev_mapping[k][0]], outputs_step2_simple_joints[self.rev_mapping[k][1]]))
            else:
                outputs_pass_triple_joints.append(self.triple_joints[k][1].apply(outputs_conv_triple_joints[count], outputs_step2_triple_joints[self.rev_mapping[k][0]], outputs_step2_triple_joints[self.rev_mapping[k][1]]))
            outputs_relu_triple_joints.append(self.triple_joints[k][2].apply(outputs_pass_triple_joints[count]))
            outputs_step1_triple_joints.append(self.triple_joints[k][3].apply(outputs_relu_triple_joints[count]))
            outputs_step1relu_triple_joints.append(self.triple_joints[k][4].apply(outputs_step1_triple_joints[count]))
            outputs_step2_triple_joints[k] = self.triple_joints[k][5].apply(outputs_step1relu_triple_joints[count])
            count += 1

        # do double joint 1
        k = 1
        outputs_conv_double_joints_1 = self.double_joints[k][0].apply(inputs)
        outputs_pass_double_joints_1 = self.double_joints[k][1].apply(outputs_conv_double_joints_1, outputs_step2_triple_joints[self.rev_mapping[k]])
        outputs_relu_double_joints_1 = self.double_joints[k][2].apply(outputs_pass_double_joints_1)

        # do negative joint
        output_conv_neg = self.negative_conv[0].apply(inputs)
        output_relu_neg = self.negative_conv[1].apply(output_conv_neg)

        # apply last convolution
        outputs_tree = []
        assert(len(outputs_relu_simple_joints) == len(self.simple_joints))
        for k in range(len(self.list_simple_joints)):
            outputs_tree.append(self.simple_joints[self.list_simple_joints[k]][5].apply(outputs_relu_simple_joints[k]))
        assert(len(outputs_relu_double_joints) == len(self.double_joints) - 1)
        for k in range(len(self.list_double_joints)):
            if self.list_double_joints[k] != 1:
                outputs_tree.append(self.double_joints[self.list_double_joints[k]][6].apply(outputs_relu_double_joints[k]))
            else:
                outputs_tree.append(self.double_joints[self.list_double_joints[k]][3].apply(outputs_relu_double_joints_1))
        assert(len(outputs_relu_triple_joints) == len(self.triple_joints))
        for k in range(len(self.list_triple_joints)):
            outputs_tree.append(self.triple_joints[self.list_triple_joints[k]][6].apply(outputs_relu_triple_joints[k]))
        outputs_tree.append(self.negative_conv[2].apply(output_relu_neg))

        # concatenate
        output = tensor.concatenate(outputs_tree, axis=1)

        return output
