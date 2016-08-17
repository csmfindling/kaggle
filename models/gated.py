from blocks.bricks import MLP, Rectifier, application, Logistic
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks.cost import Cost
from blocks.graph import ComputationGraph, apply_dropout
from blocks.filter import VariableFilter
from blocks.roles import PARAMETER, OUTPUT, INPUT
from theano.tensor import abs_


class MAPECost(Cost):

    @application(outputs=["cost"])
    def apply(self, y, y_hat):
        e_y_hat = abs_(y - y_hat)/y_hat
        return 100*e_y_hat.mean()


def build_mlp(features_cat, features_int, labels):

    mlp_int = MLP(activations=[Rectifier(), Rectifier()],
                  dims=[19, 50, 50],
                  weights_init=IsotropicGaussian(),
                  biases_init=Constant(0),
                  name='mlp_interval')
    mlp_int.initialize()
    mlp_cat = MLP(activations=[Logistic()],
                  dims=[320, 50],
                  weights_init=IsotropicGaussian(),
                  biases_init=Constant(0),
                  name='mlp_categorical')
    mlp_cat.initialize()

    mlp = MLP(activations=[Rectifier(), None],
              dims=[50, 50, 1],
              weights_init=IsotropicGaussian(),
              biases_init=Constant(0))
    mlp.initialize()

    gated = mlp_cat.apply(features_cat) * mlp_int.apply(features_int)
    prediction = mlp.apply(gated)
    cost = MAPECost().apply(prediction, labels)

    cg = ComputationGraph(cost)
    print cg.variables

    cg_dropout1   = apply_dropout(cg, [VariableFilter(roles=[OUTPUT])(cg.variables)[1], VariableFilter(roles=[OUTPUT])(cg.variables)[3]], .2)
    cost_dropout1 = cg_dropout1.outputs[0]

    return cost_dropout1, cg_dropout1.parameters, cost
