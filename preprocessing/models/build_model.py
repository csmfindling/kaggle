from blocks.bricks import MLP, Rectifier, application, Logistic, Linear, Initializable, Sequence
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks.cost import AbsoluteError, Cost, SquaredError, BinaryCrossEntropy
from blocks.graph import ComputationGraph, apply_dropout
from blocks.filter import VariableFilter
from blocks.roles import PARAMETER, OUTPUT, INPUT
from theano.tensor import abs_
from blocks.roles import add_role, WEIGHT, COST, PARAMETER


# from theano import tensor; labels_num = tensor.dmatrix('labels_num'); labels_cat = tensor.dmatrix('labels_cat')

def build_autoencoder(features, labels_num, labels_cat):

    mlp_bottom  = MLP(activations=[Rectifier(), Rectifier(), Rectifier(), Rectifier(), Rectifier()], dims=[24033, 5000, 1000, 100, 1000, 5000], weights_init=IsotropicGaussian(), biases_init=Constant(1))
    mlp_bottom.initialize()

    mlp_top = build_top_mlp()
    mlp_top.push_initialization_config()
    mlp_top.initialize()

    # a = mlp_bottom.apply(features)
    # b = mlp_top.apply(a)

    # Construct feedforward sequence
    ss_seq = Sequence([mlp_bottom.apply, mlp_top.apply])
    ss_seq.push_initialization_config()
    ss_seq.initialize()

    [outputs_numerical, outputs_categorical] = ss_seq.apply(features)

    cost        = SquaredError().apply(labels_num, outputs_numerical) + BinaryCrossEntropy().apply(labels_cat, outputs_categorical)

    cg          = ComputationGraph(cost)

    #cg_dropout0   = apply_dropout(cg, [VariableFilter(roles=[INPUT])(cg.variables)[1]], .2)
    #cg_dropout1   = apply_dropout(cg, [VariableFilter(roles=[OUTPUT])(cg.variables)[1], VariableFilter(roles=[OUTPUT])(cg.variables)[3]], .2)
    #cost_dropout1 = cg_dropout1.outputs[0]

    return cost, cg.parameters

class build_top_mlp(Initializable):

    def __init__(self, **kwargs):

        children = []

        self.layers_numerical   = []
        self.layers_numerical.append(Linear(name='input_to_numerical_linear', input_dim=5000, output_dim=17, weights_init=IsotropicGaussian(), biases_init=Constant(1)))

        self.layers_categorical = []
        self.layers_categorical.append(Linear(name='input_to_categorical_linear', input_dim=5000, output_dim=24016, weights_init=IsotropicGaussian(), biases_init=Constant(1)))
        self.layers_categorical.append(Logistic(name = 'input_to_categorical_sigmoid'))

        children += self.layers_numerical
        children += self.layers_categorical
        kwargs.setdefault('children', []).extend(children)

        super(build_top_mlp, self).__init__(**kwargs)
        

    def __allocate(self, layer, number):
        layer.allocate(); [W,b] = layer.parameters;
        W.name = 'W' + str(number); b.name = 'b' + str(number)
        add_role(W, WEIGHT); add_role(b, WEIGHT)
        #self.parameters.append(W); self.parameters.append(b)

    def _allocate(self):
        self.__allocate(self.layers_numerical[0], 1)
        #self.__allocate(self.layers_categorical[0], 1)

    def _initialize(self):
        self.layers_numerical[0].initialize()
        #self.layers_categorical[0].initialize()

    @application
    def apply(self, inputs):

        outputs_numerical   = self.layers_numerical[0].apply(inputs)
        outputs_categorical = self.layers_categorical[0].apply(inputs)
        outputs_categorical = self.layers_categorical[1].apply(outputs_categorical)

        #output = tensor.concatenate([outputs_numerical, outputs_categorical][outputs_numerical, outputs_categorical], axis=1)

        return [outputs_numerical, outputs_categorical]