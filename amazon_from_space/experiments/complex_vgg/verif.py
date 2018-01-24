import lasagne
import theano
from theano import tensor
from .vgg_16 import get_model as model_vgg16
#from vgg_16 import get_model as model_vgg16

# build model and load weights
input_var = tensor.tensor4('X')
_, test_prediction, _ = model_vgg16(input_var)

# create prediction function
val_fn          = theano.function([input_var], [test_prediction])

# Try for a few data points
n_datapoints = 2

from fuel.streams import ServerDataStream
import numpy as np

train_stream = ServerDataStream(('features', 'labels'),
                                produces_examples=False)

labels_count = np.zeros((17,))
mb_count = 0

for data in train_stream.get_epoch_iterator():
    labels_count += data[1].sum(axis=0)
    mb_count += 1

feat            = np.asarray(data[0][:n_datapoints], dtype=np.float32)
pred            = val_fn(feat)

print('Prediction for the {0} datapoints is : '.format(n_datapoints))
print(pred)
