import theano
import socket
import numpy

from theano import tensor
from fuel.streams import ServerDataStream


## choose model
from model.vgg_structured import build_model

from blocks.algorithms import GradientDescent, Adam
from blocks.graph import ComputationGraph, apply_batch_normalization, get_batch_normalization_updates, apply_dropout
from blocks.model import Model
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT, INPUT

# BUILD MODEL
images = tensor.ftensor4('images')
labels = tensor.ftensor4('labels')
cost_dropout, parameters = build_model(images, labels)


# LEARN WEIGHTS
train_stream = ServerDataStream(('images', 'labels'), False, hwm=10)
valid_stream = ServerDataStream(('images', 'labels'), False, hwm=10, port=5558)
model = Model(cost_dropout)

# ALGORITHM
alpha = 0.01 # learning rate of Adam
algorithm = GradientDescent(
    cost=cost_dropout,
    parameters=parameters,
    step_rule=Adam(),
    on_unused_sources='ignore'
)

# EXTENSIONS
from blocks.extensions import Printing, Timing
from blocks.extensions.training import TrackTheBest
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.saveload import Checkpoint
from blocks_extras.extensions.plot import Plot
import socket
import datetime
import time

host_plot = 'http://tfjgeorge.com:5006'
cost_dropout.name = 'cost'

extensions = [
    Timing(every_n_batches=50),
    TrainingDataMonitoring([cost_dropout], prefix='train'),
    DataStreamMonitoring(variables=[cost_dropout], data_stream=valid_stream, prefix="valid", every_n_batches=50),
    Plot('%s %s' % (socket.gethostname(), datetime.datetime.now(), ),channels=[['train_cost', 'valid_cost']], every_n_batches=50, server_url=host_plot),
    TrackTheBest('valid_cost'),
    Checkpoint('train', save_separately=["model","log"]),
    FinishIfNoImprovementAfter('valid_cost_best_so_far', epochs=5),
    #FinishAfter(every_n_epochs=100),
    Printing(every_n_batches=50)
]

# MAIN LOOP
from blocks.main_loop import MainLoop

main_loop = MainLoop(model=model, data_stream=train_stream, algorithm=algorithm,
                     extensions=extensions)
main_loop.run()


