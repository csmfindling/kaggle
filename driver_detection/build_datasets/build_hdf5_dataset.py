import os
import h5py
from fuel.converters.base import progress_bar
import numpy
import math
from fuel.datasets.hdf5 import H5PYDataset
import pickle

# Load pickle data train and test
X_train, y_train, driver_id, unique_drivers = pickle.load(open('../../data/train_data_color3.pkl','rb'))
X_test, X_test_id                           = pickle.load(open('../../data/test_data_color3.pkl', 'rb'))

# Create list for validation set
driver_id_int = numpy.zeros(len(driver_id))
for i in range(len(driver_id)):
	driver_id_int[i] = int(driver_id[i][1:])

# y = numpy.asarray(y_train)
# dict_index = {}
# list_id_uniq  = numpy.unique(driver_id_int)
# dict_index['keys'] = list_id_uniq
# for id_uniq in list_id_uniq:
# 	index = numpy.where(driver_id_int==id_uniq)[0]
# 	dict_index[str(int(id_uniq)) + '_indexes'] = index.astype(int)
# 	dict_index[str(int(id_uniq)) + '_labels']  = y[index].astype(int)

# pickle.dump(dict_index, open('../../data/dict_index.pkl', 'wb'))
# Create validation set
driver_id_array = numpy.zeros(len(driver_id), dtype=numpy.int64)
for i in range(len(driver_id)):
	driver_id_array[i] = int(driver_id[i][1:])
unique_drivers_array = numpy.zeros(len(unique_drivers), dtype=numpy.int64)
for i in range(len(unique_drivers)):
	unique_drivers_array[i] = int(unique_drivers[i][1:])

validation_set_indexes = numpy.concatenate((numpy.where(driver_id_array==unique_drivers_array[0])[0], \
													numpy.where(driver_id_array==unique_drivers_array[1])[0],\
													numpy.where(driver_id_array==unique_drivers_array[2])[0]))

# parameters for dividing the training set
n_examples_train  = len(X_train)
n_examples_submit = len(X_test)
n_total           = n_examples_train + n_examples_submit

# create hdf5 instance
output_path   = '../../data/data_1.hdf5'
h5file        = h5py.File(output_path, mode='w')
hdf_images    = h5file.create_dataset('images', (n_total, 3, 160, 160), dtype='float32')
hdf_driver_id = h5file.create_dataset('driver_id', (n_total, 1), dtype='int32')
hdf_labels    = h5file.create_dataset('labels', (n_total, 1), dtype='int32')

hdf_images.dims[0].label    = 'batch'
hdf_images.dims[1].label    = 'height'
hdf_images.dims[2].label    = 'width'
hdf_images.dims[3].label    = 'channels'
hdf_driver_id.dims[0].label = 'batch'
hdf_labels.dims[0].label    = 'batch'

# Create matching for creating valid and train set
validation_set_indexes_list = list(validation_set_indexes)
indexes  = range(n_examples_train)
for ind in validation_set_indexes_list:
	indexes.remove(ind)
indexes  = indexes + validation_set_indexes_list
assert(len(indexes) == n_examples_train)

# build hdf5 train and submit
with progress_bar('train', n_examples_train) as bar:
	for j in range(n_examples_train):
		hdf_images[j]    = numpy.rollaxis(X_train[indexes[j]], 2, 0)
		hdf_labels[j]    = y_train[indexes[j]]
		hdf_driver_id[j] = int(driver_id[indexes[j]][1:])
		bar.update(j)

with progress_bar('submit', n_examples_submit) as bar:
    for j in range(n_examples_submit):
    	hdf_images[n_examples_train + j]    = numpy.rollaxis(X_test[j], 2, 0)
    	hdf_driver_id[n_examples_train + j] = int(X_test_id[j].split('_')[1][:-4])
    	bar.update(j)

# Save hdf5 train and submit
split_dict = {}
sources = ['images', 'driver_id', 'labels']
for name, slice_ in zip(['train', 'submit'], [(0, n_examples_train), (n_examples_train, n_total)]):
    split_dict[name] = dict(zip(sources, [slice_] * len(sources)))

h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

h5file.flush()
h5file.close()
