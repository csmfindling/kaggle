from blocks.bricks import MLP, Rectifier
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks.cost import AbsoluteError
from blocks.graph import ComputationGraph, apply_dropout
from blocks.filter import VariableFilter
from blocks.roles import PARAMETER, OUTPUT, INPUT

# from theano import tensor; features = tensor.matrix('features'); labels = tensor.matrix('labels')

def build_mlp(features, labels):

    mlp  = MLP(activations=[Rectifier(), None], dims=[233, 800, 1], weights_init=IsotropicGaussian(), biases_init=Constant(1))
    mlp.initialize()

    prediction = mlp.apply(features)
    cost       = AbsoluteError().apply(prediction, labels)

    cg            = ComputationGraph(cost)
    cg_dropout0   = apply_dropout(cg, [VariableFilter(roles=[INPUT])(cg.variables)[2]], .2)
    cg_dropout1   = apply_dropout(cg_dropout0, [VariableFilter(roles=[OUTPUT])(cg_dropout0.variables)[1], VariableFilter(roles=[OUTPUT])(cg_dropout0.variables)[2]], .5)
    cost_dropout1 = cg_dropout1.outputs[0]

    return cost_dropout1, cg_dropout1.parameters, cost
