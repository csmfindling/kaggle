########################
#     make features    #
########################

import csv
import numpy as np


input_file = csv.DictReader(open("../../data/ech_apprentissage.csv"), delimiter=';')

lists_elements_categories = {}
#list_strings = ['energie_veh', 'marque', 'profession', 'var14', 'var6', 'var8', 'codepostal']
list_categorical = ['energie_veh', 'marque', 'profession', 'codepostal', 'var2', 'var3', 'var4', 'var5', 'var6', 'var8', 'var13', 'var14', 'var15', 'var16', 'var17', 'var19', 'var20', 'var21', 'var22']
for cat in list_categorical:
	lists_elements_categories[cat] = []

nb_of_training  = 0
nb_of_errors    = 0
features_errors_0  = []
features_errors_NR = []
for row in input_file:
	for k in row.keys():
		boolean = False
		if ((row[k] == '') or (row[k] == 'NR')):
			if (k not in features_errors_0) and (row[k] == ''):
				features_errors_0.append(k)
			if (k not in features_errors_NR) and (row[k] == 'NR'):
				features_errors_NR.append(k)
			nb_of_errors += 1
			boolean = True
		if (not boolean) and (k in list_categorical):
			if row[k] not in lists_elements_categories[k]:
				lists_elements_categories[k].append(row[k])
	nb_of_training += 1

features_errors = features_errors_0 + features_errors_NR

list_keys_0 = row.keys()
list_keys   = []
for key in list_keys_0:
	if key != 'id' and key != 'prime_tot_ttc':
		list_keys.append(key)

nb_features = len(list_keys) - len(list_categorical) + len(features_errors)
nb_features_numerical   = len(list_keys) - len(list_categorical)
nb_features_categorical = 0
for feature in features_errors:
	if feature in list_categorical:
		nb_features_categorical += 1
	else:
		nb_features_numerical += 1

for categorie in list_categorical:
	nb_features_categorical += len(lists_elements_categories[categorie])
	nb_features += len(lists_elements_categories[categorie])

assert(nb_features_numerical + nb_features_categorical == nb_features)

input_file = csv.DictReader(open("../../data/ech_apprentissage.csv"), delimiter=';')

nb_all              = nb_of_training
X_train_categorical = np.zeros([nb_of_training, nb_features_categorical], dtype=np.int8)
X_train_numerical   = np.zeros([nb_of_training, nb_features_numerical])
y_train             = np.zeros(nb_of_training)
label_string        = 'prime_tot_ttc'

val_index    = np.arange(0, nb_all, 30)
train_index  = list(np.arange(0, nb_all))
for ind in val_index:
	train_index.remove(ind)

train_index = np.asarray(train_index)
assert(len(val_index) + len(train_index) == nb_all)

idx = 0
for row in input_file:
	idx_feature_numerical   = 0
	idx_feature_categorical = 0
	for k in list_keys:

		if (k not in features_errors) and (k not in list_categorical):
			X_train_numerical[idx, idx_feature_numerical] = row[k]
			idx_feature_numerical    += 1

		elif (k not in features_errors) and (k in list_categorical):
			X_train_categorical[idx, idx_feature_categorical + lists_elements_categories[k].index(row[k])] = 1
			idx_feature_categorical                                               += len(lists_elements_categories[k])

		elif (k in features_errors) and (k not in list_categorical):
			if (row[k] == 'NR') or (row[k] == ''):
				X_train_numerical[idx, idx_feature_numerical]     = 1
				X_train_numerical[idx, idx_feature_numerical + 1] = 0
			else:
				X_train_numerical[idx, idx_feature_numerical]     = 0
				X_train_numerical[idx, idx_feature_numerical + 1] = row[k]

			idx_feature_numerical             += 2

		elif (k in features_errors) and (k in list_categorical):
			if (row[k] == 'NR') or (row[k] == ''):
				X_train_categorical[idx, idx_feature_categorical]     = 1
			else:
				X_train_categorical[idx, idx_feature_categorical]                                                  = 0
				X_train_categorical[idx, idx_feature_categorical + 1 + lists_elements_categories[k].index(row[k])] = 1

			idx_feature_categorical += len(lists_elements_categories[k]) + 1
		else:
			raise Exception
	y_train[idx] = row[label_string]
	idx += 1

assert(idx_feature_numerical == nb_features_numerical)
assert(idx_feature_categorical == nb_features_categorical)
assert(idx_feature_categorical + idx_feature_numerical == nb_features)
assert(idx == nb_all)

X_train_numerical = X_train_numerical/np.max(X_train_numerical, axis=0)

y_train_0 = np.array(y_train)

X_train_cat = X_train_categorical[train_index]
X_train_num = X_train_numerical[train_index]
y_train     = y_train_0[train_index]

X_val_cat   = X_train_categorical[val_index]
X_val_num   = X_train_numerical[val_index]
y_val       = y_train_0[val_index]

print('features created')

########################
#      make HDF5       #
########################

import os
import h5py
from fuel.converters.base import progress_bar
import numpy
import math
from fuel.datasets.hdf5 import H5PYDataset
import pickle

# Load pickle data train and test
# [X_train_cat, X_train_num, y_train, X_val_cat, X_val_num, y_val] = pickle.load(open('../../data/train.pkl','rb'))

# parameters for dividing the training set
n_examples_train = X_train_cat.shape[0] # 290000
n_examples_val   = X_val_cat.shape[0] # 10000
n_total          = n_examples_train + n_examples_val
n_features_cat   = X_train_cat.shape[1] # 24016
n_features_num   = X_train_num.shape[1] # 17
# n_features = 24033

# create hdf5 instance
output_path       = '../../data/data.hdf5'
h5file            = h5py.File(output_path, mode='w')
hdf_features_cat  = h5file.create_dataset('features_cat', (n_total, n_features_cat), dtype='int8')
hdf_features_num  = h5file.create_dataset('features_num', (n_total, n_features_num), dtype='float32')
hdf_labels        = h5file.create_dataset('labels', (n_total, 1), dtype='float32')

hdf_features_cat.dims[0].label = 'batch'
hdf_features_cat.dims[1].label = 'features_cat'
hdf_features_num.dims[0].label = 'batch'
hdf_features_num.dims[1].label = 'features_num'
hdf_labels.dims[0].label       = 'batch'
hdf_labels.dims[1].label       = 'labels'

# build hdf5 train and submit
with progress_bar('train', n_examples_train) as bar:
	for j in range(n_examples_train):
		hdf_features_cat[j] = X_train_cat[j]
		hdf_features_num[j] = X_train_num[j]
		hdf_labels[j]       = y_train[j]
		bar.update(j)

with progress_bar('validation', n_examples_val) as bar:
    for j in range(n_examples_val):
    	hdf_features_cat[n_examples_train + j] = X_val_cat[j]
    	hdf_features_num[n_examples_train + j] = X_val_num[j]
    	hdf_labels[n_examples_train + j]       = y_val[j]
    	bar.update(j)

# Save hdf5 train and submit
split_dict = {}
sources = ['features_cat', 'features_num', 'labels']
for name, slice_ in zip(['train', 'validation'], [(0, n_examples_train), (n_examples_train, n_total)]):
    split_dict[name] = dict(zip(sources, [slice_] * len(sources)))

h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

h5file.flush()
h5file.close()
