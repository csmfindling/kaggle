from blocks.bricks import MLP, Rectifier, application, Logistic, Softmax
from only_localization import build_mlp as build_mlp_onlyloc
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks.cost import Cost
from blocks.graph import ComputationGraph, apply_dropout
from blocks.filter import VariableFilter
from blocks.roles import PARAMETER, OUTPUT, INPUT
from theano import tensor

from cost import MAPECost

def build_mlp(features_car_cat, features_car_int, features_nocar_cat, features_nocar_int, features_cp, features_hascar,
              means, labels):

    mlp_car = MLP(activations=[Rectifier(), Rectifier(), None],
                  dims=[8 + 185, 200, 200, 1],
                  weights_init=IsotropicGaussian(.1),
                  biases_init=Constant(0),
                  name='mlp_interval_car')
    mlp_car.initialize()
    mlp_nocar = MLP(activations=[Rectifier(), Rectifier(), None],
                    dims=[5 + 135, 200, 200, 1],
                    weights_init=IsotropicGaussian(.1),
                    biases_init=Constant(0),
                    name='mlp_interval_nocar')
    mlp_nocar.initialize()

    feature_car = tensor.concatenate((features_car_cat, features_car_int), axis=1)
    feature_nocar = tensor.concatenate((features_nocar_cat, features_nocar_int), axis=1)
    prediction = mlp_nocar.apply(feature_nocar)
    # gating with the last feature : does the dude own a car
    prediction += tensor.addbroadcast(features_hascar, 1) * mlp_car.apply(feature_car)

    prediction_loc, _, _, _, = \
            build_mlp_onlyloc(features_car_cat, features_car_int,
                              features_nocar_cat, features_nocar_int,
                              features_cp, features_hascar,
                              means, labels)
    prediction += prediction_loc

    # add crm
    mlp_crm = MLP(activations=[None],
                  dims=[1, 1],
                  weights_init=IsotropicGaussian(.1),
                  biases_init=Constant(0),
                  name='mlp_crm')
    mlp_crm.initialize()
    crm = features_nocar_int[:, 0][:, None]
    prediction = prediction * mlp_crm.apply(crm)

    cost = MAPECost().apply(labels, prediction)

    cg = ComputationGraph(cost)
    input_var = VariableFilter(roles=[INPUT])(cg.variables)
    print input_var

    cg_dropout1 = apply_dropout(cg, [input_var[6], input_var[7]], .4)
    cost_dropout1 = cg_dropout1.outputs[0]

    return prediction, cost_dropout1, cg_dropout1.parameters, cost
