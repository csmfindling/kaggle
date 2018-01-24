
# coding: utf-8

# In[1]:

# source https://github.com/tfjgeorge/ift6268/blob/master/projet/FF%20texture%20generation%20bricks%20-%20Adam%20blocks%20training.ipynb


# In[1]:

import theano
import socket
import numpy

from theano import tensor
from fuel.datasets.hdf5 import H5PYDataset
from functions.custom_transformers import *
from fuel.streams import DataStream
from fuel.schemes import cross_validation, SequentialScheme

train_set = H5PYDataset(
        '../data/data_1.hdf5',
        which_sets=('train',),
        load_in_memory=True
)

def get_train_stream(scheme):

    stream = DataStream.default_stream(
        train_set,
        iteration_scheme=scheme
    )

    stream = RandomRotate(stream, 20)
    stream = Normalize(stream)
    stream = Cast(stream, 'floatX')

    return stream

def get_valid_stream(scheme):

    stream = DataStream.default_stream(
        train_set,
        iteration_scheme=scheme
    )

    stream = Normalize(stream)
    stream = Cast(stream, 'floatX')

    return stream

# get_ipython().magic(u'matplotlib inline')
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

## choose model
#from models.simple_conv_seq import build_model
from models.vgg_features import build_model

from blocks.algorithms import GradientDescent, Adam
from blocks.graph import ComputationGraph, apply_batch_normalization, get_batch_normalization_updates, apply_dropout

from blocks.model import Model
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT, INPUT


# LOAD NEURAL NETWORK AND LOSS

# In[2]:

# ensemble training

schemes_5_fold = cross_validation(SequentialScheme, train_set.num_examples, 4, batch_size=100)
n_model = 1
for (train_scheme, valid_scheme) in schemes_5_fold:
    print '=====\nModel %d\n=====' % (n_model, )

    images = tensor.ftensor4('images')
    labels = tensor.imatrix('labels')
    cost, all_parameters = build_model(images, labels)


    # LEARN WEIGHTS

    # In[3]:

    train_stream = get_train_stream(train_scheme)
    valid_stream = get_valid_stream(valid_scheme)

    # In[5]:
    alpha = 0.1

    cg    = ComputationGraph(cost)

    cg_bn = apply_batch_normalization(cg)

    # drop out some variables
    inputs = VariableFilter(roles=[INPUT])(cg_bn.variables)
    cg_dropout = apply_dropout(cg_bn, [inputs[11], inputs[0]], .5)

    cost_bn = cg_bn.outputs[0]
    cost_dropout = cg_dropout.outputs[0]
    model = Model(cost)
        

    print 'Optimizing parameters :'
    print all_parameters

    for parameters in all_parameters:

        algorithm = GradientDescent(
                cost=cost_dropout,
                parameters=parameters,
                step_rule=Adam(),
                on_unused_sources='ignore'
        )

        # Add updates for population parameters
        pop_updates = get_batch_normalization_updates(cg_bn)
        extra_updates = [(p, m * alpha + p * (1 - alpha))
                                for p, m in pop_updates]

        algorithm.add_updates(extra_updates)


        # In[6]:

        from blocks.extensions import Printing, Timing, FinishAfter
        from blocks.extensions.training import TrackTheBest
        from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
        from blocks.extensions.stopping import FinishIfNoImprovementAfter
        from blocks.extensions.saveload import Checkpoint
        from blocks_extras.extensions.plot import Plot
        import socket
        import datetime
        import time

        host_plot = 'http://tfjgeorge.com:5006'
        cost.name = 'cost'

        extensions = [
            Timing(),
            TrainingDataMonitoring([cost], after_epoch=True, prefix='train'),
            DataStreamMonitoring(variables=[cost], data_stream=valid_stream, prefix="valid"),
            Plot('%s %s' % (socket.gethostname(), datetime.datetime.now(), ),channels=[['train_cost', 'valid_cost']], after_epoch=True, server_url=host_plot),
            TrackTheBest('valid_cost'),
            Checkpoint('trained_%d' % (n_model,), save_separately=["model","log"]),
            FinishIfNoImprovementAfter('valid_cost_best_so_far', epochs=5),
            #FinishAfter(after_n_epochs=100),
            Printing()
        ]


        # In[7]:

        from blocks.main_loop import MainLoop


        main_loop = MainLoop(model=model, data_stream=train_stream, algorithm=algorithm,
                             extensions=extensions)


        # In[ ]:

        main_loop.run()
        n_model += 1





