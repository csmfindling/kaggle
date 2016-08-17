from theano import tensor

# choose model
from models.gated import build_mlp
from blocks.algorithms import GradientDescent, Adam
from blocks.model import Model


# load model
features_cat = tensor.dmatrix('features_cat')
features_int = tensor.dmatrix('features_num')
labels = tensor.dmatrix('labels')
cost, params, valid_cost = build_mlp(features_cat, features_int, labels)
model = Model(cost)

# load data
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.datasets.hdf5 import H5PYDataset

train_set = H5PYDataset(
    './data/data.hdf5',
    which_sets=('train',),
    subset=slice(0, 290000), #
    load_in_memory=True
)

valid_set = H5PYDataset(
    './data/data.hdf5',
    which_sets=('validation',),
    subset=slice(0, 9081), #
    load_in_memory=True
)


train_stream = DataStream.default_stream(
    train_set,
    iteration_scheme=ShuffledScheme(train_set.num_examples, batch_size=1000)
)


valid_stream = DataStream.default_stream(
    valid_set,
    iteration_scheme=ShuffledScheme(valid_set.num_examples, batch_size=1000)
)


# load algorithm
from blocks.algorithms import Adam

algorithm = GradientDescent(
        cost=cost,
        parameters=params,
        step_rule=Adam(),
        on_unused_sources='ignore'
)


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
valid_cost.name = 'valid_cost'

extensions = [
    Timing(),
    TrainingDataMonitoring([cost], after_epoch=True, prefix='train'),
    DataStreamMonitoring(variables=[valid_cost], data_stream=valid_stream),
    #Plot('%s %s' % (socket.gethostname(), datetime.datetime.now(), ),channels=[['train_cost', 'valid_cost']], after_epoch=True, server_url=host_plot),
    TrackTheBest('valid_cost'),
    Checkpoint('model', save_separately=["model","log"]),
    FinishIfNoImprovementAfter('valid_cost_best_so_far', epochs=5),
    #FinishAfter(after_n_epochs=100),
    Printing()
]


from blocks.main_loop import MainLoop


main_loop = MainLoop(model=model, data_stream=train_stream, algorithm=algorithm,
                     extensions=extensions)


main_loop.run()





