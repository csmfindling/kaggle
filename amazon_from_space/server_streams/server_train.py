from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.datasets.hdf5 import H5PYDataset
from fuel.server import start_server
from config import basepath, minibatch_size
from transformers.custom_transformers import Standardize
import argparse
import numpy

parser = argparse.ArgumentParser()
parser.add_argument('--mode')
args = parser.parse_args()

if args.mode == None:
	path = 'data.hdf5'
elif args.mode == 'jpeg':
	path = 'data.hdf5'
elif args.mode == 'tiff':
	path = 'data_tiff.hdf5'
else:
	raise SyntaxError

train_set = H5PYDataset(
    basepath + path,
    which_sets=('train',),
    subset=slice(0, 32383), # 32383 = 80% of training set
    sources=['features', 'labels'],
    load_in_memory=False
)

stream = DataStream.default_stream(
    train_set,
    iteration_scheme=ShuffledScheme(train_set.num_examples, minibatch_size)
)

print('I provide sources ', train_set.sources)
print('Number of examples', train_set.num_examples)

standardized_stream = Standardize(stream, 255.)

start_server(standardized_stream)
