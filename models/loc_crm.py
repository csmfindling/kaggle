from only_localization import build_mlp as build_mlp_onlyloc
from blocks.bricks import MLP, Rectifier, application, Logistic, Softmax
from blocks.initialization import IsotropicGaussian, Constant
from cost import MAPECost
from blocks.graph import ComputationGraph, apply_dropout
from blocks.filter import VariableFilter
from blocks.roles import INPUT


def build_mlp(features_car_cat, features_car_int, features_nocar_cat, features_nocar_int, features_cp, features_hascar,
              means, labels):

    prediction, _, _, _, = \
            build_mlp_onlyloc(features_car_cat, features_car_int,
                              features_nocar_cat, features_nocar_int, features_cp, features_hascar,
                              means, labels)

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

    cg_dropout = apply_dropout(cg, [input_var[7], input_var[5]], .4)
    cost_dropout = cg_dropout.outputs[0]

    return prediction, cost_dropout, cg_dropout.parameters, cost
