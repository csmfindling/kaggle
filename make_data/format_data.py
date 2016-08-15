import csv
import numpy as np


input_file = csv.DictReader(open("../../data/ech_apprentissage.csv"), delimiter=';')

lists = {}
#list_strings = ['energie_veh', 'marque', 'profession', 'var14', 'var6', 'var8', 'codepostal']
list_strings = ['energie_veh', 'marque', 'profession', 'var14', 'var6', 'var8']
for string in list_strings:
	lists[string] = []

nb_of_training  = 0
nb_of_errors    = 0
errors_training = []
for row in input_file:
	boolean = False
	for k in row.keys():
		if ((row[k] == '') or (row[k] == 'NR')) and (nb_of_training not in errors_training):
			errors_training.append(nb_of_training)
			nb_of_errors += 1
			boolean = True
	nb_of_training += 1
	if not boolean:
		for string in list_strings:
			if row[string] not in lists[string]:
				lists[string].append(row[string])

list_keys_0 = row.keys()
list_keys   = []
for key in list_keys_0:
	if key != 'id' and key != 'prime_tot_ttc':
		list_keys.append(key)

nb_features = len(list_keys) - len(list_strings) - 1 + 1 + 1
for string in list_strings:
	nb_features += len(lists[string])

input_file = csv.DictReader(open("../../data/ech_apprentissage.csv"), delimiter=';')

nb_all       = nb_of_training - nb_of_errors
X_train      = np.zeros([nb_of_training - nb_of_errors, nb_features])
y_train      = np.zeros(nb_of_training - nb_of_errors)
label_string = 'prime_tot_ttc'

val_index    = np.arange(0, nb_all, 30)
train_index  = list(np.arange(0, nb_all))
for ind in val_index:
	train_index.remove(ind)

train_index = np.asarray(train_index)

idx_training = 0
idx = 0
for row in input_file:
	if idx_training not in errors_training:
		idx_feature = 0
		for k in list_keys:
			if (k != label_string) and (k not in list_strings) and (k != 'var16') and (k != 'codepostal'):
				X_train[idx, idx_feature] = row[k]
				idx_feature += 1
			elif (k != label_string) and (k != 'var16') and (k != 'codepostal'):
				X_train[idx, idx_feature + lists[k].index(row[k])] = 1
				idx_feature += len(lists[k])
			elif (k != label_string) and (k != 'codepostal'):
				X_train[idx, idx_feature] = row[k]
				idx_feature += 1
			elif (k != label_string):
				if row[k] == 'NR':
					assert(1 == 0)
				elif row[k] == 'ARMEE':
					X_train[idx, idx_feature]     = 1
					X_train[idx, idx_feature + 1] = 0
				else:
					X_train[idx, idx_feature]     = 0
					X_train[idx, idx_feature + 1] = row[k]
				idx_feature += 2	
		y_train[idx] = row[label_string]
		idx += 1
	idx_training += 1

import pickle
X_train = X_train/np.max(X_train, axis=0)

pickle.dump([X_train[train_index], y_train[train_index], X_train[val_index], y_train[val_index]], open('../../data/train.pkl', 'wb'))

