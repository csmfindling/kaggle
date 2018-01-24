from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.datasets.hdf5 import H5PYDataset
from fuel.server import start_server
from config import basepath, minibatch_size
from transformers.custom_transformers import Standardize

submit_set = H5PYDataset(
    basepath + 'data.hdf5',
    which_sets=('submit',),
    #subset=slice(0,50),
    sources=['features', 'image_name'],
    load_in_memory=False
)

stream = DataStream.default_stream(
    submit_set,
    iteration_scheme=SequentialScheme(submit_set.num_examples, minibatch_size)
)

print('I provide sources ', submit_set.sources)
print('Number of examples', submit_set.num_examples)

standardized_stream = Standardize(stream, 255)

start_server(standardized_stream)
