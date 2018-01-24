from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.datasets.hdf5 import H5PYDataset
from fuel.server import start_server
import math

train_set = H5PYDataset(
	'../../data/pretrain/data_pretraining.hdf5',
	which_sets=('train',),
	subset=slice(0, 81036), #
	load_in_memory=False
)

index_features = 0
index_labels   = 1

stream = DataStream.default_stream(
    train_set,
    iteration_scheme=ShuffledScheme(train_set.num_examples, 10)
)

start_server(stream, hwm=10)
