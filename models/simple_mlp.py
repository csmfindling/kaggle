from blocks.bricks import MLP, Rectifier, application, Logistic, Softmax
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks.cost import Cost
from blocks.graph import ComputationGraph, apply_dropout
from blocks.filter import VariableFilter
from blocks.roles import PARAMETER, OUTPUT, INPUT
from theano import tensor


class MAPECost(Cost):

    @application(outputs=["cost"])
    def apply(self, y, y_hat):
        e_y_hat = tensor.abs_(y - y_hat)/y_hat
        return 100*e_y_hat.mean()


def build_mlp(features_car_cat, features_car_int, features_nocar_cat, features_nocar_int, features_cp, means, labels):

    mlp_car = MLP(activations=[Rectifier(), Rectifier(), None],
                  dims=[8 + 185, 50, 50, 1],
                  weights_init=IsotropicGaussian(.1),
                  biases_init=Constant(0),
                  name='mlp_interval_car')
    mlp_car.initialize()
    mlp_nocar = MLP(activations=[Rectifier(), Rectifier(), None],
                  dims=[5 + 136, 50, 50, 1],
                  weights_init=IsotropicGaussian(.1),
                  biases_init=Constant(0),
                  name='mlp_interval_nocar')
    mlp_nocar.initialize()

    feature_car = tensor.concatenate((features_car_cat, features_car_int), axis=1)
    feature_nocar = tensor.concatenate((features_nocar_cat, features_nocar_int), axis=1)
    prediction = mlp_nocar.apply(feature_nocar)
    # gating with the last feature : does the dude own a car
    prediction += tensor.addbroadcast(features_nocar_cat[:,-1:], 1) * mlp_car.apply(feature_car)

    cost = MAPECost().apply(prediction, labels)

    cg = ComputationGraph(cost)
    var = VariableFilter(roles=[INPUT])(cg.variables)

    cg_dropout1 = apply_dropout(cg, [var[4], var[5], var[8], var[9]], .2)
    cost_dropout1 = cg_dropout1.outputs[0]

    return cost_dropout1, cg_dropout1.parameters, cost
