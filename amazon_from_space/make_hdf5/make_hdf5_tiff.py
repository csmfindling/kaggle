# some useful code https://www.kaggle.com/fppkaggle/making-tifs-look-normal-using-spectral-fork

import numpy as np 
import h5py
from fuel.converters.base import progress_bar
from fuel.datasets.hdf5 import H5PYDataset
import csv
import glob
from skimage import io
from config import basepath
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

# scaler 
scaler = MinMaxScaler(feature_range=(0, 255))

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
images_paths_train = glob.glob(basepath + 'train-tif-v2/*.tif')
# assert(len(images_paths_train) == nb_training_examples)
tifffile           = io.imread(images_paths_train[0])
a                  = np.asarray(tifffile) #rescaleIMG = scaler.fit_transform(rescaleIMG)
length, width, _   = a.shape
nb_channels        = 4

# test data
images_path_test =  glob.glob(basepath + 'test-tif*/*.tif')
nb_test_examples = len(images_path_test)
print('Number of testing examples is {0}'.format(nb_test_examples))

# hdf5
print('Creating hdf5...')
output_path = basepath + 'data_tiff.hdf5'
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

maximums = np.zeros(4)
with progress_bar('train', nb_training_examples) as bar:

    with open(basepath + 'train_v2.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        count      = 0
        for row in spamreader:
            if row[0].startswith('train_'):
                train_idx           = int(row[0].split('_')[-1])
                p                   = glob.glob(basepath + 'train-tif-v2/train_{0}.tif'.format(train_idx))
                hdf_names[count]    = 'train_{0}.tif'.format(train_idx)
                # tiff image
                img                 = io.imread(p[0])
                rescaleIMG          = np.reshape(img[:,:,-1], (-1, 1))
                rescaleIMG          = scaler.fit_transform(rescaleIMG.astype(np.float32))
                img_scaled          = (np.reshape(rescaleIMG, img[:,:,-1].shape)).astype(np.uint8)
                #img                 = np.moveaxis(img_scaled[:,:,:nb_channels], 2, 0)[np.asarray([2,1,0,3])] # move channel axis + rgb
                # jpeg image
                p                   = glob.glob(basepath + 'train-jpg/train_{0}.jpg'.format(train_idx))
                jpgfile             = Image.open(p[0])
                a                   = np.array(jpgfile)
                assert(np.sum(a[:,:,-1] != 0)==0)
                a[:,:,-1]           = img_scaled
                img                 = np.moveaxis(a, 2, 0)
                hdf_features[count] = img
                labels_example      = row[-1].split(' ')
                lab                 = np.zeros(nb_labels, dtype=np.int8)
                lab[CLASSES.keys()] = [CLASSES[key] in labels_example for key in CLASSES.keys()]*1
                hdf_labels[count]   = lab
                count              += 1
                bar.update(count)
                maximums            = np.maximum(np.max(img, axis=(1,2)), maximums)

# with progress_bar('test', nb_test_examples) as bar:
#         for p in images_path_test:
#             raise NotImplementedError('Not implemented yet for testing set')
#             img                 = io.imread(p)
#             hdf_names[count]    = p.split('/')[-1]
#             rescaleIMG          = np.reshape(img[:,:,-1], (-1, 1))
#             rescaleIMG          = scaler.fit_transform(rescaleIMG.astype(np.float32))
#             img_scaled          = (np.reshape(rescaleIMG, img[:,:,-1].shape)).astype(np.uint8)
#             #img                 = np.moveaxis(img_scaled[:,:,:nb_channels], 2, 0)[np.asarray([2,1,0,3])] # move channel axis + rgb
#             hdf_features[count] = img
#             labels_example      = row[-1].split(' ')
#             lab                 = np.zeros(nb_labels, dtype=np.int8)
#             lab[CLASSES.keys()] = -1
#             hdf_labels[count]   = lab
#             count              += 1
#             bar.update(count - nb_training_examples)
# 
# print('maximums is {0}'.format(maximums)) #  result is : [ 29443.  24650.  34166.  39786.]

# Save hdf5 train and submit
split_dict = {}
sources = ['features', 'labels', 'image_name']
for name, slice_ in zip(['train', 'submit'], [(0, nb_training_examples), (nb_training_examples, nb_training_examples + nb_test_examples)]):
    split_dict[name] = dict(zip(sources, [slice_] * len(sources)))

h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

h5file.flush()
h5file.close()









