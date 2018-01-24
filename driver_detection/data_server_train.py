from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.datasets.hdf5 import H5PYDataset
from fuel.server import start_server
from functions.custom_transformers import RandomDownscale, RandomFixedSizeCrop, RandomRotate, Normalize, Cast
import math

train_set = H5PYDataset(
	'../data/data_1.hdf5',
	which_sets=('train',),
	subset=slice(0, 20000),
	load_in_memory=True
)

index_images = 0
index_labels = 1

stream = DataStream.default_stream(
    train_set,
    iteration_scheme=ShuffledScheme(train_set.num_examples, 125)
)

#downscaled_stream = RandomDownscale(stream, 140)
stream = RandomRotate(stream, 20)
#cropped_stream    = RandomFixedSizeCrop(rotated_stream, (130,130))
stream = Normalize(stream)
stream = Cast(stream, 'floatX')

start_server(stream, hwm=10)
