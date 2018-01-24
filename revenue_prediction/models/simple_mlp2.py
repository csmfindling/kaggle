from blocks.bricks import MLP, Rectifier, application, Logistic, Softmax, Linear
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks.cost import Cost
from blocks.graph import ComputationGraph, apply_dropout
from blocks.filter import VariableFilter
from blocks.roles import PARAMETER, OUTPUT, INPUT, WEIGHT
from theano import tensor


class MAPECost(Cost):

    @application(outputs=["cost"])
    def apply(self, y, y_hat):
        e_y_hat = tensor.abs_(y - y_hat)/y
        return 100*e_y_hat.mean()


def build_mlp(features_car_cat, features_car_int, features_nocar_cat, features_nocar_int, features_cp, features_hascar,
              means, labels):

    mlp_nocar = MLP(activations=[Rectifier(), None],
                  dims=[5 + 135, 5, 1],
                  weights_init=IsotropicGaussian(.1),
                  biases_init=Constant(0),
                  name='mlp_interval_nocar')
    mlp_nocar.initialize()

    mlp_car = MLP(activations=[Rectifier(), Rectifier(), None],
                  dims=[5 + 135 + 8 + 185, 200, 200, 1],
                  weights_init=IsotropicGaussian(.1),
                  biases_init=Constant(0),
                  name='mlp_interval_car')
    mlp_car.initialize()

    #pred_nocar = Linear(name='no_car_prediction', input_dim=200, output_dim=1, weights_init=IsotropicGaussian(0.1),biases_init=Constant(0))
    #pred_nocar.initialize()

    mlp_weighted_layer = MLP(activations=[Logistic()],
                  dims=[1, 1],
                  weights_init=Constant(.2),
                  biases_init=Constant(0),
                  name='mlp_weighted')
    mlp_weighted_layer.initialize()

    # processing the no car features
    #feature_car   = tensor.concatenate((features_car_cat, features_car_int), axis=1)
    #feature_all   = tensor.concatenate((feature_car, feature_nocar), axis=1)
    feature_nocar    = tensor.concatenate((features_nocar_cat, features_nocar_int), axis=1)
    output_mlp_nocar = mlp_nocar.apply(feature_nocar) #* tensor.addbroadcast(features_hascar - 1, 1)
    #prediction       = pred_nocar.apply(output_mlp_nocar)

    # adding the car features
    feature_car = tensor.concatenate((features_car_cat, features_car_int, features_nocar_cat, features_nocar_int), axis=1)
    # gating with the last feature : does the dude own a car
    prediction = tensor.addbroadcast(features_hascar, 1) * mlp_car.apply(feature_car) + output_mlp_nocar #* tensor.addbroadcast(features_hascar - 1, 1) 

    # add weighted mean
    weights     = 2 * mlp_weighted_layer.apply(tensor.addbroadcast(means[features_cp, 0], 1)) - 1
    prediction += (weights * tensor.addbroadcast(means[features_cp, 1], 1) + (1 - weights) * tensor.addbroadcast(means[features_cp, 2], 1)) * tensor.addbroadcast(features_hascar, 1) \
                                + tensor.addbroadcast(means[features_cp, 3], 1) * tensor.addbroadcast(features_hascar - 1, 1)     #means[features_cp, 2]

    #prediction += tensor.addbroadcast(means[features_cp, 1], 1) # means[features_cp, 1]

    cost = MAPECost().apply(prediction, labels)

    cg = ComputationGraph(cost)
    input_var = VariableFilter(roles=[INPUT])(cg.variables)
    print input_var
    w = VariableFilter(roles=[WEIGHT])(cg.variables)
    print w

    #cost = cost + .1 * (w[-1]**2).sum() + .1 * (w[-2]**2).sum() #+ .01 * (w[0]**2).sum() + .01 * (w[1]**2).sum()
    cg_dropout1 = apply_dropout(cg, [input_var[-5], input_var[-3]], .45)
    cost_dropout1 = cg_dropout1.outputs[0]

    #W1, W2, W3, W4 = VariableFilter(roles=[WEIGHT])(cg.variables)
    #cost_dropout1 = cost_dropout1 + 0.005 * (W1 ** 2).sum() + 0.005 * (W2 ** 2).sum() + 0.005 * (W3 ** 2).sum() + 0.005 * (W4 ** 2).sum()

    return prediction, cost_dropout1, cg_dropout1.parameters, cost

# class WeightedLayer(Feedforward):

#     @application
#     def apply(self, weights, means, prediction, application_call):
#         return prediction + weights * means + (1 - weights) * overal_mean

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
