import os
import h5py
from fuel.converters.base import progress_bar
import numpy
import math
from fuel.datasets.hdf5 import H5PYDataset
import pickle

# Load pickle data train and test
[X_train, y_train, X_val, y_val] = pickle.load(open('../../data/train.pkl','rb'))

# parameters for dividing the training set
n_examples_train = X_train.shape[0] # 289111
n_examples_val   = X_val.shape[0] # 9970
n_total          = n_examples_train + n_examples_val
n_features       = X_train.shape[1]

# create hdf5 instance
output_path   = '../../data/data.hdf5'
h5file        = h5py.File(output_path, mode='w')
hdf_features  = h5file.create_dataset('features', (n_total, n_features), dtype='float32')
hdf_labels    = h5file.create_dataset('labels', (n_total, 1), dtype='float32')

hdf_features.dims[0].label = 'batch'
hdf_features.dims[1].label = 'features'
hdf_labels.dims[0].label   = 'batch'
hdf_labels.dims[1].label   = 'labels'

# build hdf5 train and submit
with progress_bar('train', n_examples_train) as bar:
	for j in range(n_examples_train):
		hdf_features[j]  = X_train[j]
		hdf_labels[j]    = y_train[j]
		bar.update(j)

with progress_bar('validation', n_examples_val) as bar:
    for j in range(n_examples_val):
    	hdf_features[n_examples_train + j] = X_val[j]
    	hdf_labels[n_examples_train + j]   = y_val[j]
    	bar.update(j)

# Save hdf5 train and submit
split_dict = {}
sources = ['features', 'labels']
for name, slice_ in zip(['train', 'validation'], [(0, n_examples_train), (n_examples_train, n_total)]):
    split_dict[name] = dict(zip(sources, [slice_] * len(sources)))

h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

h5file.flush()
h5file.close()
