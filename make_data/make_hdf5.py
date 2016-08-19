# -*- coding: utf-8 -*-

import pandas
import numpy
import h5py
from fuel.converters.base import progress_bar
from fuel.datasets.hdf5 import H5PYDataset

BASEPATH = '../data/'
list_categorical = ['energie_veh', 'marque', 'profession', 'var2', 'var3', 'var4', 'var5',
                    'var6', 'var8', 'var13', 'var14', 'var15', 'var16', 'var17', 'var19', 'var20', 'var21', 'var22']
list_interval = ['annee_naissance', 'annee_permis', 'puis_fiscale', 'anc_veh',
                 'kmage_annuel', 'crm', 'var1', 'var7', 'var9', 'var10', 'var11', 'var12', 'var18']
list_car = ['annee_naissance', 'annee_permis', 'marque', 'puis_fiscale', 'anc_veh', 'energie_veh', 'kmage_annuel',
            'var7', 'var8', 'var9', 'var10']
list_interval_car = [v for v in list_interval if v in list_car]
list_interval_nocar = [v for v in list_interval if v not in list_car]

NA_VALUES = dict((column_name, ['NR']) for column_name in list_interval)

def cp_reader(x):
    if x == 'ARMEE':
        return 100000
    if x == 'NR':
        return 0
    return int(x)

CONVERTERS = {'codepostal': cp_reader}

print('Preprocess training set...')
data_train = pandas.read_csv(BASEPATH + "ech_apprentissage.csv", delimiter=';', na_values=NA_VALUES, converters=CONVERTERS)
data_submit = pandas.read_csv(BASEPATH + "ech_test.csv", delimiter=';', na_values=NA_VALUES, converters=CONVERTERS)
#shuffle data_train
data_train = data_train.sample(frac=1).reset_index(drop=True)

list_uniques = dict((column_name, data_train[column_name].unique())
                    for column_name in list_categorical)
uniques_count = dict((column_name, len(v))
                     for column_name, v in list_uniques.items())
uniques_count_car = dict((column_name, val) for column_name, val in uniques_count.iteritems()
                         if column_name in list_car)
uniques_count_nocar = dict((column_name, val) for column_name, val in uniques_count.iteritems()
                         if column_name not in list_car)
total_uniques = sum(uniques_count.values())
total_uniques_car = sum(uniques_count_car.values())
total_uniques_nocar = sum(uniques_count_nocar.values())
cumsums_car = numpy.cumsum(uniques_count_car.values())
cumsums_nocar = numpy.cumsum(uniques_count_nocar.values())
uniques_startindex_car = dict(
    zip(uniques_count_car.keys(), numpy.append([0], cumsums_car)))
uniques_startindex_nocar = dict(
    zip(uniques_count_nocar.keys(), numpy.append([0], cumsums_nocar)))
unique_to_index = dict((col, dict((v, i) for i, v in enumerate(values)))
                       for col, values in list_uniques.items())

codepostaux = data_train['codepostal'].unique()
codepostaux = numpy.append([0], codepostaux) # used for unknown CP
cp_to_id = dict((cp, i) for i, cp in enumerate(codepostaux))

departements = numpy.unique([c/1000 for c in codepostaux])
dep_to_id = dict((dep, i) for i, dep in enumerate(departements))
print('Found départements:')
print(departements)

min_max = dict((key, (numpy.nanmin(data_train[key].values), numpy.nanmax(data_train[key].values)))
               for key in list_interval)
print('Min/max values for each category:')
print min_max

print('Found categories:')
for column_name, count in uniques_count.items():
    print('%12s: %3d different values' % (column_name, count))
print('Total: %d values in %d columns' %
      (total_uniques, len(list_uniques.keys())))
print('Category features:')
print('Car: %d, Nocar: %d' % (total_uniques_car, total_uniques_nocar))
print('Interval features:')
print('Car: %d, Nocar: %d' % (len(list_interval_car), len(list_interval_nocar)))
print('%d different code postaux' % (len(codepostaux),))

print('Feature ordering:')
print('Car related features')
print([v for v in list_interval if v in list_car])
print('Other features')
print([v for v in list_interval if v not in list_car])

print('Creating hdf5...')
output_path = BASEPATH + 'data.hdf5'
h5file = h5py.File(output_path, mode='w')
n_total = len(data_train) + len(data_submit)
hdf_features_car_cat = h5file.create_dataset(
    'features_car_cat', (n_total, total_uniques_car), dtype='int8')
hdf_features_car_int = h5file.create_dataset(
    'features_car_int', (n_total, len(list_interval_car)), dtype='float32')
hdf_features_nocar_cat = h5file.create_dataset(
    'features_nocar_cat', (n_total, total_uniques_nocar), dtype='int8')
hdf_features_nocar_int = h5file.create_dataset(
    'features_nocar_int', (n_total, len(list_interval_nocar)), dtype='float32')
hdf_labels = h5file.create_dataset('labels', (n_total, 1), dtype='float32')
hdf_cp = h5file.create_dataset('codepostal', (n_total, 2), dtype='int32')
hdf_hascar= h5file.create_dataset('features_hascar', (n_total, 1), dtype='int8')

hdf_features_car_cat.dims[0].label = 'batch'
hdf_features_car_cat.dims[1].label = 'features_car_cat'
hdf_features_car_int.dims[0].label = 'batch'
hdf_features_car_int.dims[1].label = 'features_car_int'
hdf_features_nocar_cat.dims[0].label = 'batch'
hdf_features_nocar_cat.dims[1].label = 'features_nocar_cat'
hdf_features_nocar_int.dims[0].label = 'batch'
hdf_features_nocar_int.dims[1].label = 'features_nocar_int'
hdf_labels.dims[0].label = 'batch'
hdf_labels.dims[1].label = 'labels'
hdf_cp.dims[0].label = 'batch'
hdf_cp.dims[1].label = 'codepostal'
hdf_hascar.dims[0].label = 'batch'
hdf_hascar.dims[1].label = 'hascar'

missing_codepostaux = []
missing_departements = []

for set_label, data in [('train', data_train), ('submit', data_submit)]:
    start_i = 0 if set_label == 'train' else len(data_train)

    with progress_bar(set_label, len(data)) as bar:
        for i, row in data.iterrows():
            # does the dude have a car ?
            has_car = row['marque'] != 'NR'
            hdf_hascar[start_i + i] = 1 if has_car else 0

            # categorical features
            feature_onehot_car_cat = numpy.zeros(total_uniques_car)
            feature_onehot_nocar_cat = numpy.zeros(total_uniques_nocar)
            for column_name in list_categorical:
                try:
                    if column_name in list_car:
                        feature_onehot_car_cat[uniques_startindex_car[column_name]
                                           + unique_to_index[column_name][row[column_name]]] = 1
                    else:
                        feature_onehot_nocar_cat[uniques_startindex_nocar[column_name]
                                           + unique_to_index[column_name][row[column_name]]] = 1
                except:
                    print('Found a category that was not in the train set at index %d: %s for column %s' %
                          (i, row[column_name], column_name)) 
            hdf_features_car_cat[start_i + i] = feature_onehot_car_cat
            hdf_features_nocar_cat[start_i + i] = feature_onehot_nocar_cat

            # interval features
            feature_car_interval = numpy.zeros(len(list_interval_car))
            feature_nocar_interval = numpy.zeros(len(list_interval_nocar))
            for j, column_name in enumerate([v for v in list_interval if v in list_car]):
                if not numpy.isnan(row[column_name]):
                    feature_car_interval[j] = (float(row[column_name]) - min_max[column_name][0]) \
                            / (min_max[column_name][1] - min_max[column_name][0])
            for j, column_name in enumerate([v for v in list_interval if v not in list_car]):
                if not numpy.isnan(row[column_name]):
                    feature_nocar_interval[j] = (float(row[column_name]) - min_max[column_name][0]) \
                            / (min_max[column_name][1] - min_max[column_name][0])
            hdf_features_car_int[start_i + i] = feature_car_interval
            hdf_features_nocar_int[start_i + i] = feature_nocar_interval

            # code postal
            cp = row['codepostal']
            cp_feat = numpy.zeros(2)
            if cp in cp_to_id:
                cp_feat[0] = cp_to_id[cp]
            else:
                cp_feat[0] = cp_to_id[0]
                missing_codepostaux.append(cp)

            # departement
            dep = cp/1000 if type(cp) == int else cp[:2]
            if dep in dep_to_id:
                cp_feat[1] = dep_to_id[dep]
            else:
                cp_feat[1] = dep_to_id[0]
                missing_departements.append(dep)


            hdf_cp[start_i + i] = cp_feat

            # target
            if set_label == 'train':
                hdf_labels[start_i + i] = row['prime_tot_ttc']

            bar.update(i)

print('Code postaux not in the train set:', numpy.unique(missing_codepostaux, return_counts=True))
print('Départements not in the train set:', numpy.unique(missing_departements, return_counts=True))

# Save hdf5 train and submit
split_dict = {}
sources = ['features_car_cat', 'features_car_int', 'features_nocar_cat', 'features_nocar_int',
           'labels', 'codepostal', 'features_hascar']
for name, slice_ in zip(['train', 'submit'], [(0, len(data_train)), (len(data_train), n_total)]):
    split_dict[name] = dict(zip(sources, [slice_] * len(sources)))

h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

h5file.flush()
h5file.close()
