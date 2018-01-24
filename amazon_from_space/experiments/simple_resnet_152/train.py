#import sys
#sys.path.append('experiments/simple_vgg/')
from .resnet_152 import get_model
from theano import tensor, function
import lasagne
from fuel.streams import ServerDataStream
import numpy
from utils import f2_score
import pickle

num_epochs = 50

train_stream = ServerDataStream(('features', 'labels'),
                                produces_examples=False)

valid_stream = ServerDataStream(('features', 'labels'),
                                produces_examples=False, port=5558)

X = tensor.ftensor4('images')
y = tensor.imatrix('targets')

prediction_train, prediction_test, params = get_model(X)

loss = lasagne.objectives.binary_crossentropy(prediction_train, y)
loss = loss.mean()

prediction_01 = tensor.ge(prediction_train, numpy.float32(.5))
f2            = f2_score(prediction_01, y)
f2_diff       = f2_score(prediction_train, y)
loss          = - f2_diff

updates = lasagne.updates.nesterov_momentum(
    loss, params, learning_rate=1e-4, momentum=0.9)

train_fn = function([X, y], loss, updates=updates)
valid_fn = function([X, y], f2)

best_valid_score = 0
patience         = 0
all_train_loss   = []
for epoch in range(num_epochs):
    f2_valid_loss = []
    f2_train_loss = []
    for imgs, targets in train_stream.get_epoch_iterator():
        f2_train_loss.append(train_fn(imgs, targets))
    all_train_loss.append(f2_train_loss)
    train_score = -numpy.mean(numpy.asarray(f2_train_loss))
    print('train score : {0}'.format(train_score))
    for imgs, targets in valid_stream.get_epoch_iterator():
        f2_valid_loss.append(valid_fn(imgs, targets))
    valid_score = numpy.mean(numpy.asarray(f2_valid_loss))
    print('valid score : {0}'.format(valid_score))
    if best_valid_score < valid_score:
        best_valid_score = valid_score
        patience         = 0
        param_values = [p.get_value() for p in params]
        numpy.savez_compressed('simple_resnetvalid_152.npz', param_values)
        pickle.dump(all_train_loss, open('simple_resnetvalid_152.pkl', 'wb'))
    else:
        patience += 1
        if patience == 5:
            break
    print('patience : {0}'.format(patience))
    print('\n')

