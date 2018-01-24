from blocks.bricks import MLP, Rectifier, application, Initializable
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks.cost import AbsoluteError, Cost
from blocks.graph import ComputationGraph, apply_dropout
from blocks.filter import VariableFilter
from blocks.roles import PARAMETER, OUTPUT, INPUT, add_role, WEIGHT
from theano.tensor import abs_
from theano import tensor
from theano.tensor.nnet.nnet import sigmoid

from theano import tensor; features_car_cat = tensor.dmatrix('features_car_cat'); features_car_int = tensor.dmatrix('features_car_int')
features_nocar_cat = tensor.dmatrix('features_nocar_cat'); features_nocar_int = tensor.dmatrix('features_nocar_int')
features_hascar = tensor.dmatrix('features_hascar'); means = tensor.dmatrix('means'); labels = tensor.dmatrix('labels'); features_cp = tensor.imatrix('features_cp');

# from theano import tensor; features_int = tensor.dmatrix('features_cp'); features_cat = tensor.dmatrix('features_cat')
# from theano import tensor; labels = tensor.dmatrix('labels'); labels_mean = tensor.dmatrix('labels_mean')

class MAPECost(Cost):

    @application(outputs=["cost"])
    def apply(self, y, y_hat, labels_mean):
        e_y_hat  =  abs_(y - (y_hat + labels_mean))/(y_hat + labels_mean)
        return 100*e_y_hat.mean()

def build_mlp(features_int, features_cat, labels, labels_mean):

    inputs = tensor.concatenate([features_int, features_cat], axis=1)

    mlp  = MLP(activations=[Rectifier(), Rectifier(), Rectifier(), None], dims=[337, 800, 1200, 1], weights_init=IsotropicGaussian(), biases_init=Constant(1))
    mlp.initialize()

    prediction = mlp.apply(inputs)
    cost       = MAPECost().apply(prediction, labels, labels_mean)

    cg            = ComputationGraph(cost)
    #cg_dropout0   = apply_dropout(cg, [VariableFilter(roles=[INPUT])(cg.variables)[1]], .2)
    cg_dropout1   = apply_dropout(cg, [VariableFilter(roles=[OUTPUT])(cg.variables)[1], VariableFilter(roles=[OUTPUT])(cg.variables)[3], VariableFilter(roles=[OUTPUT])(cg.variables)[5]], .2)
    cost_dropout1 = cg_dropout1.outputs[0]

    return cost_dropout1, cg_dropout1.parameters, cost #cost, cg.parameters, cost #

# class sum_layer(Initializable):

#     def __init__(self, **kwargs):
#         super(sum_layer, self).__init__(**kwargs)
#         self.W          = tensor.scalar('mixture_W', dtype='float32')
#         self.parameters = [self.W]

#     def _allocate(self):
#         self.W.name = 'mixture_W'
#         add_role(self.W, WEIGHT)
#         self.W.value = .2

#     @application
#     def apply(self, inputs_mean, inputs_number, inputs, overal_mean):
#         w = 2 * sigmoid(inputs_number * self.W) - 1.
#         return inputs + w * inputs_mean + (1 - w) * overal_mean 




