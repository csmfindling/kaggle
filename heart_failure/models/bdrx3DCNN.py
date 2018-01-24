from sunnybrook.models.bdrx3DCNN import get_model as bottom_model
import theano
from theano import tensor
import numpy

def get_model(input_var, multiply_var):

	# on va arbitrairement choisir le 5eme qui correspond
	# a une coupe pas trop degueu normalement

	test_prediction_bottom, prediction_bottom, params_bottom = \
		bottom_model(input_var[:,4,:,:,:], multiply_var)

	mult = theano.shared(numpy.array([[1, 1]]).astype('float32'))
	mult_b = tensor.addbroadcast(mult, 0)

	sums = prediction_bottom.sum(axis=(2, 3))
	maxs = sums.max(axis=(0, 1))
	mins = sums.min(axis=(0, 1))
	minmax = tensor.stack((maxs, mins), axis=1)

	test_sums = test_prediction_bottom.sum(axis=(2, 3))
	test_maxs = test_sums.max(axis=(0, 1))
	test_mins = test_sums.min(axis=(0, 1))
	test_minmax = tensor.stack((test_maxs, test_mins), axis=1)


	return test_minmax*mult_b*multiply_var, minmax*mult_b*multiply_var, test_prediction_bottom, params_bottom, [mult]
