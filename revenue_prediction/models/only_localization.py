from blocks.bricks import MLP, Rectifier, application, Logistic, Softmax
from blocks.initialization import IsotropicGaussian, Constant
from blocks.graph import ComputationGraph, apply_dropout
from blocks.filter import VariableFilter
from blocks.roles import PARAMETER, OUTPUT, INPUT
from theano import tensor

from cost import MAPECost

def build_mlp(features_car_cat, features_car_int, features_nocar_cat,
              features_nocar_int, features_cp, features_hascar,
              means, labels):

    features = tensor.concatenate([
        features_hascar,
        means['cp'][features_cp[:, 0]],
        means['dep'][features_cp[:, 1]]
        ], axis=1)

    mlp = MLP(activations=[Rectifier(), Rectifier(), None],
              dims=[5, 50, 50, 1],
              weights_init=IsotropicGaussian(.1),
              biases_init=Constant(0),
              name='mlp')
    mlp.initialize()

    prediction = mlp.apply(features)

    cost = MAPECost().apply(labels, prediction)

    cg = ComputationGraph(cost)
    input_var = VariableFilter(roles=[INPUT])(cg.variables)
    print input_var

    cg_dropout1 = apply_dropout(cg, [input_var[3], input_var[5]], .4)
    cost_dropout1 = cg_dropout1.outputs[0]

    return prediction, cost_dropout1, cg_dropout1.parameters, cost
