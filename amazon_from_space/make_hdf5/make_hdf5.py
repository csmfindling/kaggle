import numpy as np
import h5py
from fuel.converters.base import progress_bar
from fuel.datasets.hdf5 import H5PYDataset
import csv
import glob
from PIL import Image
from config import basepath

# get Classes
CLASSES = {}
nb_training_examples = 0
with open(basepath + 'train_v2.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        if row[0].startswith('train_'):
            nb_training_examples += 1
            labels_example        = row[1].split(' ')
            for lab in labels_example:
                if lab not in CLASSES.values():
                    if len(CLASSES.keys()) > 0:
                        CLASSES[max(CLASSES.keys()) + 1] = lab
                    else:
                        CLASSES[0] = lab

nb_labels = len(CLASSES.keys())

print('Number of labels is {0}'.format(nb_labels))
print('Number of training examples is {0}'.format(nb_training_examples))

# image size
images_paths_train = glob.glob(basepath + 'train-jpg/*.jpg')
assert(len(images_paths_train) == nb_training_examples)
jpgfile            = Image.open(images_paths_train[0])
a                  = np.asarray(jpgfile)
length, width, _   = a.shape
nb_channels        = 3

# test data
images_path_test =  glob.glob(basepath + 'test-jpg*/*.jpg')
nb_test_examples = len(images_path_test)
print('Number of testing examples is {0}'.format(nb_test_examples))

# hdf5
print('Creating hdf5...')
output_path = basepath + 'data.hdf5'
h5file = h5py.File(output_path, mode='w')

hdf_features                = h5file.create_dataset('features', (nb_test_examples + nb_training_examples, nb_channels, length, width), dtype='int32')
hdf_labels                  = h5file.create_dataset('labels', (nb_test_examples + nb_training_examples, nb_labels), dtype='int8')
hdf_names                   = h5file.create_dataset('image_name', (nb_test_examples + nb_training_examples,), dtype='S20')

hdf_features.dims[0].label    = 'batch'
hdf_features.dims[2].label    = 'height'
hdf_features.dims[3].label    = 'width'
hdf_features.dims[1].label    = 'channels'
hdf_labels.dims[0].label      = 'batch'
hdf_labels.dims[1].label      = 'labels'

with progress_bar('train', nb_training_examples) as bar:

    with open(basepath + 'train_v2.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        count      = 0
        for row in spamreader:
            if row[0].startswith('train_'):
                train_idx           = int(row[0].split('_')[-1])
                p                   = glob.glob(basepath + 'train-jpg/train_{0}.jpg'.format(train_idx))
                hdf_names[count]    = 'train_{0}.jpg'.format(train_idx)
                jpgfile             = Image.open(p[0])
                a = np.asarray(jpgfile)
                assert(np.sum(a[:,:,nb_channels] != 0)==0)
                img                 = np.moveaxis(a[:,:,:nb_channels], 2, 0)
                hdf_features[count] = img
                labels_example      = row[-1].split(' ')
                lab                 = np.zeros(nb_labels, dtype=np.int8)
                lab[CLASSES.keys()] = [CLASSES[key] in labels_example for key in CLASSES.keys()]*1
                hdf_labels[count]   = lab
                count              += 1
                bar.update(count)

with progress_bar('test', nb_test_examples) as bar:
        for p in images_path_test:
            jpgfile             = Image.open(p)
            a                   = np.asarray(jpgfile)
            hdf_names[count]    = p.split('/')[-1]
            img                 = np.moveaxis(a[:,:,:nb_channels], 2, 0)
            hdf_features[count] = img
            labels_example      = row[-1].split(' ')
            lab                 = np.zeros(nb_labels, dtype=np.int8)
            lab[CLASSES.keys()] = -1
            hdf_labels[count]   = lab
            count              += 1
            bar.update(count - nb_training_examples)

# Save hdf5 train and submit
split_dict = {}
sources = ['features', 'labels', 'image_name']
for name, slice_ in zip(['train', 'submit'], [(0, nb_training_examples), (nb_training_examples, nb_training_examples + nb_test_examples)]):
    split_dict[name] = dict(zip(sources, [slice_] * len(sources)))

h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

h5file.flush()
h5file.close()
