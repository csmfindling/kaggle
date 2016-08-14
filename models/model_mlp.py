from blocks.bricks import MLP, Rectifier, application
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks.cost import AbsoluteError, Cost
from blocks.graph import ComputationGraph, apply_dropout
from blocks.filter import VariableFilter
from blocks.roles import PARAMETER, OUTPUT, INPUT
from theano.tensor import abs_

# from theano import tensor; features = tensor.dmatrix('features'); labels = tensor.dmatrix('labels')

class MAPECost(Cost):

    @application(outputs=["cost"])
    def apply(self, y, y_hat):
        e_y_hat  =  abs_(y - y_hat)/y_hat
        return 100*e_y_hat.mean()

def build_mlp(features, labels):

    mlp  = MLP(activations=[Rectifier(), Rectifier(), None], dims=[233, 800, 800, 1], weights_init=IsotropicGaussian(), biases_init=Constant(1))
    mlp.initialize()

    prediction = mlp.apply(features)
    cost       = MAPECost().apply(prediction, labels)

    cg            = ComputationGraph(cost)
    cg_dropout0   = apply_dropout(cg, [VariableFilter(roles=[INPUT])(cg.variables)[1]], .2)
    cg_dropout1   = apply_dropout(cg_dropout0, [VariableFilter(roles=[OUTPUT])(cg_dropout0.variables)[1], VariableFilter(roles=[OUTPUT])(cg_dropout0.variables)[3]], .5)
    cost_dropout1 = cg_dropout1.outputs[0]

    return cost_dropout1, cg_dropout1.parameters, cost
