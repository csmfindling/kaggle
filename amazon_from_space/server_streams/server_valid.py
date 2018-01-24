from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.datasets.hdf5 import H5PYDataset
from fuel.server import start_server
from config import basepath, minibatch_size
from transformers.custom_transformers import Standardize
import argparse

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

valid_set = H5PYDataset(
    basepath + path,
    which_sets=('train',),
    subset=slice(32383, 40479), # 40479 20% of training set
    sources=['features', 'labels'],
    load_in_memory=False
)

stream = DataStream.default_stream(
    valid_set,
    iteration_scheme=SequentialScheme(valid_set.num_examples, minibatch_size)
)

print('I provide sources ', valid_set.sources)
print('Number of examples', valid_set.num_examples)

standardized_stream = Standardize(stream, 255.)

start_server(standardized_stream, port=5558)
