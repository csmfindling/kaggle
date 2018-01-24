import os
import h5py
from fuel.converters.base import progress_bar
import numpy
import math
from fuel.datasets.hdf5 import H5PYDataset
import pickle
import glob
import cv2
from scipy.io import loadmat
import re
import numpy as np

# paths
features_train_path = '../../../pretraining_data/lsp/train/'
labels_train_path   = '../../../pretraining_data/lsp/train_label/'
features_val_path   = '../../../pretraining_data/lsp/val/'
labels_val_path     = '../../../pretraining_data/lsp/val_label/'

# files
features_train_files = glob.glob(features_train_path + '*.png')
labels_train_files   = glob.glob(labels_train_path + '*.mat')
features_val_files   = glob.glob(features_val_path + '*.png')
labels_val_files     = glob.glob(labels_val_path + '*.mat')

# verification
len(features_train_files) == len(labels_train_files)
len(features_val_files) == len(labels_val_files)

# parameters for dividing the training set
n_examples_train  = len(features_train_files)
n_examples_valid  = len(labels_val_files)
n_total           = n_examples_train + n_examples_valid

print 'number of training examples ' + str(n_examples_train)
print 'number of validation examples ' + str(n_examples_valid)

# function
def get_im_cv2(path, color_type=3):
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    return img

# create hdf5 instance
output_path   = '../../../pretraining_data/data_pretraining.hdf5'
h5file        = h5py.File(output_path, mode='w')

#dtype         = h5py.special_dtype(vlen=numpy.dtype('uint8'))
hdf_features  = h5file.create_dataset('features', (n_total, 3, 336, 336), dtype='uint8')
hdf_labels    = h5file.create_dataset('labels', (n_total, 15, 42, 42), dtype='uint8')

hdf_features.dims[0].label = 'batch'
hdf_features.dims[1].label = 'channel'
hdf_features.dims[2].label = 'height'
hdf_features.dims[3].label = 'width'
hdf_labels.dims[0].label   = 'batch'
hdf_labels.dims[1].label   = 'channel'
hdf_labels.dims[2].label   = 'height'
hdf_labels.dims[3].label   = 'width'

# Create matching for creating valid and train set
# range for selecting upper body joints
r = range(0,8) + range(14,20) + [26]

# build hdf5 train and submit
with progress_bar('train', n_examples_train) as bar:
	for j in range(n_examples_train):
		assert(re.findall(r'\d+', features_train_files[j]) == re.findall(r'\d+', labels_train_files[j])) 
		hdf_features[j]  = np.rollaxis(get_im_cv2(features_train_files[j]), 2)
		hdf_labels[j]    = np.rollaxis(loadmat(labels_train_files[j])['map'], 2)[r]
		bar.update(j)

with progress_bar('valid', n_examples_valid) as bar:
    for j in range(n_examples_valid):
    	assert(re.findall(r'\d+', features_val_files[j]) == re.findall(r'\d+', labels_val_files[j])) 
    	hdf_features[n_examples_train + j] = np.rollaxis(get_im_cv2(features_val_files[j]), 2)
    	hdf_labels[n_examples_train + j]   = np.rollaxis(loadmat(labels_val_files[j])['map'], 2)[r]
    	bar.update(j)

# Save hdf5 train and valid
split_dict = {}
sources = ['features', 'labels']
for name, slice_ in zip(['train', 'valid'], [(0, n_examples_train), (n_examples_train, n_total)]):
    split_dict[name] = dict(zip(sources, [slice_] * len(sources)))

h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

h5file.flush()
h5file.close()