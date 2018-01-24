from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.datasets.hdf5 import H5PYDataset
from fuel.server import start_server
from functions.custom_transformers import RandomDownscale, RandomFixedSizeCrop, RandomRotate, Normalize, Cast, FixedSizeCrop
import math

train_set = H5PYDataset(
	'../data/data_1.hdf5',
	which_sets=('train',),
	subset=slice(20000, 22424),
	load_in_memory=True
)

index_images = 0
index_labels = 1

stream = DataStream.default_stream(
    train_set,
    iteration_scheme=ShuffledScheme(train_set.num_examples, 32)
)

#downscaled_stream = RandomDownscale(stream, 130) # TAKE OUT
#cropped_stream    = FixedSizeCrop(stream, (130,130))
#rotated_stream    = RandomRotate(cropped_stream, math.pi/10) # TAKE OUT
normalized_stream = Normalize(stream)
casted_stream     = Cast(normalized_stream, 'floatX')

start_server(casted_stream, port=5558, hwm=10)
