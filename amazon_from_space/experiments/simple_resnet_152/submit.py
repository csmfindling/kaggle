import sys
from vgg_16 import get_model, build_model
from theano import tensor, function, config
import lasagne
from fuel.streams import ServerDataStream
import numpy
import pickle
from config import basepath

submit_stream = ServerDataStream(('features', 'image_name'), produces_examples=False)

# tensor
X = tensor.ftensor4('images')

# build simple vgg model
net, layers_names = build_model(X)
f_pretrained      = open(basepath + 'vgg16.pkl')
model_pretrained  = pickle.load(f_pretrained)
w_pretrained      = model_pretrained['param values']
net['mean value'].set_value(model_pretrained['mean value'].astype(config.floatX))

# load weights
from lasagne.layers import set_all_param_values

with numpy.load('weights/simple_vgg_valid.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]

set_all_param_values(net[layers_names[len(layers_names)-1]], param_values[0])

# create predict function
prediction_test = lasagne.layers.get_output(net[layers_names[len(layers_names)-1]], deterministic=True)
eval_fn = function([X], [prediction_test])

# Classes
classes = {0: 'haze', 1: 'primary', 2: 'agriculture', 3: 'clear',
 			4: 'water', 5: 'habitation', 6: 'road', 7: 'cultivation', 8: 'slash_burn', 9: 'cloudy',
 			10: 'partly_cloudy', 11: 'conventional_mine', 12: 'bare_ground', 13: 'artisinal_mine', 14: 'blooming', 15: 'selective_logging', 16: 'blow_down'}

# submit file
import csv
csvfile = open('submits/submit_simpleresnet152.csv', 'wb')
spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
spamwriter.writerow(['image_name,tags'])

# prediction
import time
print('writing submit file')
for imgs, imgs_name in submit_stream.get_epoch_iterator():
    pred_targets = eval_fn(imgs)
    for pred_idx in range(len(pred_targets[0])):
    	spamwriter.writerow([imgs_name[pred_idx].split('.')[0] + ','] + [classes[k] for k in numpy.arange(len(classes))[(pred_targets[0][pred_idx] > .5)]])
    time.sleep(.5)
    
csvfile.close()

