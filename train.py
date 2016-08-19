from theano import tensor, shared, config

# choose model
# from models.simple_mlp2 import build_mlp
# from models.gated import build_mlp
#from models.simple_mlp import build_mlp
#from models.only_localization import build_mlp
from models.loc_crm import build_mlp

from blocks.algorithms import GradientDescent, Adam
from blocks.model import Model
import numpy


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
    which_sets=('train',),
    subset=slice(290000, 300000), #
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

# compute mean target values
print('Computing mean target values...')
cps = []
deps = []
primes = []
hascar = []
cp_index = train_set.provides_sources.index('codepostal')
prime_index = train_set.provides_sources.index('labels')
hascar_index = train_set.provides_sources.index('features_hascar')

for d in train_stream.get_epoch_iterator():
    cps.append(d[cp_index][:, 0])
    deps.append(d[cp_index][:, 1])
    primes.append(d[prime_index])
    hascar.append(d[hascar_index])

cps = numpy.array(cps).flatten()
deps = numpy.array(deps).flatten()
hascar = numpy.array(hascar).flatten()
primes = numpy.array(primes).flatten()
overall_mean = primes[hascar == 1].mean()

means_by_cp = []
means_by_dep = []
freqs = []
s = 0
for cp in range(23712):
    freq = (numpy.logical_and(cps==cp, hascar==1)).sum()
    if freq > 0:
        means_by_cp.append([primes[numpy.logical_and(cps==cp, hascar==1)].mean(), freq]) 
    else:
        means_by_cp.append([overall_mean, freq])
for dep in range(101):
    freq = (numpy.logical_and(deps==dep, hascar==1)).sum()
    if freq > 0:
        means_by_dep.append([primes[numpy.logical_and(deps==dep, hascar==1)].mean(), freq]) 
    else:
        means_by_dep.append([overall_mean, freq])

means_by_cp = numpy.array(means_by_cp).astype(config.floatX)
means_by_dep = numpy.array(means_by_dep).astype(config.floatX)
means = {'cp': shared(value=means_by_cp),
         'dep': shared(value=means_by_dep)}

print('done.')

# load model
features_car_cat = tensor.dmatrix('features_car_cat')
features_car_int = tensor.dmatrix('features_car_int')
features_nocar_cat = tensor.dmatrix('features_nocar_cat')
features_nocar_int = tensor.dmatrix('features_nocar_int')
features_cp = tensor.imatrix('codepostal')
features_hascar = tensor.imatrix('features_hascar')
labels = tensor.dmatrix('labels')
prediction, cost, params, valid_cost = build_mlp(features_car_cat, features_car_int, features_nocar_cat, features_nocar_int,
                                     features_cp, features_hascar, means, labels)
model = Model([cost, prediction])

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
    Plot('%s %s' % (socket.gethostname(), datetime.datetime.now(), ),channels=[['train_cost', 'valid_cost']], after_epoch=True, server_url=host_plot),
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
