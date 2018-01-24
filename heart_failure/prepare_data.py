import os
import h5py
from fuel.converters.base import progress_bar
import numpy
import dicom
from PIL import Image
import math
from fuel.datasets.hdf5 import H5PYDataset
import re

def get_features(root_path):
   """Get path to all the frame in view SAX and contain complete features"""
   ret = []
   for root, _, files in os.walk(root_path):
       root=root.replace('\\','/')
       files=[s for s in files if ".dcm" in s]
       if len(files) == 0 or not files[0].endswith(".dcm") or root.find("sax") == -1:
           continue
       prefix = files[0].rsplit('-', 1)[0]
       fileset = set(files)
       expected = ["%s-%04d.dcm" % (prefix, i + 1) for i in range(30)]
       if all(x in fileset for x in expected):
           ret.append([root + "/" + x for x in expected])
   # sort for reproduciblity
   return sorted(ret, key = lambda x: x[0])

def get_label_map(fname):
   labelmap = {}
   fi = open(fname)
   fi.readline()
   for line in fi:
       arr = line.split(',')
       labelmap[int(arr[0])] = (float(arr[1]), float(arr[2]))
   return labelmap

def get_data(lst,preproc):
   data = []
   result = []
   for path in lst:
       f = dicom.read_file(path)
       img = f.pixel_array
       dst_path = path.rsplit(".", 1)[0] + ".64x64.jpg"
       # scipy.misc.imsave(dst_path, img)

       pixel_mult = float(f.PixelSpacing[0]) * float(f.PixelSpacing[1])

       img_pos = [float(i) for i in f.ImagePositionPatient]

       # resize to 70 px imgs
       original_height, original_width = img.shape[-2:]
       multiplier = max(80./ original_width, 80./ original_height)

       width = int(math.ceil(original_width * multiplier))
       height = int(math.ceil(original_height * multiplier))

       im = Image.fromarray(img.astype('int16'))
       im = numpy.array(im.resize((width, height))).astype('int16')

       result.append(dst_path)
       data.append(im)
   return [data,result], 1/multiplier**2 * pixel_mult, img_pos

def crop_sequence(seq):
  sizes = []
  for m in range(len(seq)):
    sizes.append(numpy.array(seq[m]).shape)
  sizes = numpy.array(sizes)
  depth, min_width, min_height = numpy.min(sizes,axis=0)
  new_width_2  = min_width/2
  new_height_2 = min_height/2
  cropped_sequence = numpy.zeros([len(sizes), depth, min_width, min_height])
  for m in range(len(seq)):
    if list(numpy.array(seq[m]).shape) != list(numpy.min(sizes,axis=0)):
      half_the_width  = seq[m].shape[1] / 2
      half_the_height = seq[m].shape[2] / 2
      assert(seq[m].shape[0] == 30)
      for t in range(30):
        im = Image.fromarray(seq[m][t].astype('int16'))
        if min_width%2==1:
          img = numpy.array(im.crop(( half_the_height - new_height_2,half_the_width - new_width_2, half_the_height + new_height_2, half_the_width + new_width_2 + 1)))
        elif min_height%2==1:
          img = numpy.array(im.crop(( half_the_height - new_height_2,half_the_width - new_width_2, half_the_height + new_height_2+1, half_the_width + new_width_2)))
        else:
          img = numpy.array(im.crop(( half_the_height - new_height_2,half_the_width - new_width_2, half_the_height + new_height_2, half_the_width + new_width_2 )))
        cropped_sequence[m][t] = img
  return cropped_sequence

labels = get_label_map("./data_kaggle/train.csv")
train_features = get_features('./data_kaggle/train')
submit_features = get_features('./data_kaggle/validate')

n_examples_train = 5293
n_examples_submit = 2128
n_total = n_examples_train + n_examples_submit

output_path = './data_kaggle/kaggle_heart.hdf5'
h5file = h5py.File(output_path, mode='w')
dtype = h5py.special_dtype(vlen=numpy.dtype('uint16'))
dtype_pos = h5py.special_dtype(vlen=numpy.dtype('float32'))

hdf_features = h5file.create_dataset('sax_features', (n_total,), dtype=dtype)
hdf_shapes = h5file.create_dataset('sax_features_shapes', (n_total, 4), dtype='int32')
hdf_cases = h5file.create_dataset('cases', (n_total, 1), dtype='int32')
hdf_sax = h5file.create_dataset('sax', (n_total, ), dtype=dtype)
hdf_shapes_sax = h5file.create_dataset('sax_shapes', (n_total, 1), dtype='int32')
hdf_labels = h5file.create_dataset('targets', (n_total, 2), dtype='float32')
hdf_mult = h5file.create_dataset('multiplier', (n_total, 1), dtype='float32')
hdf_position = h5file.create_dataset('image_position', (n_total, ), dtype=dtype_pos)
hdf_shapes_position = h5file.create_dataset('image_position_shapes', (n_total, 2), dtype='int32')

# Attach shape annotations and scales
hdf_features.dims.create_scale(hdf_shapes, 'shapes')
hdf_features.dims[0].attach_scale(hdf_shapes)

hdf_shapes_labels = h5file.create_dataset('sax_features_labels', (4,), dtype='S7')
hdf_shapes_labels[...] = ['depth'.encode('utf8'),
                          'features'.encode('utf8'),
                          'height'.encode('utf8'),
                          'width'.encode('utf8')]
hdf_features.dims.create_scale(hdf_shapes_labels, 'shape_labels')
hdf_features.dims[0].attach_scale(hdf_shapes_labels)

# Attach shape annotations and scales for image position
hdf_position.dims.create_scale(hdf_shapes_position, 'shapes')
hdf_position.dims[0].attach_scale(hdf_shapes_position)

hdf_shapes_position_labels = h5file.create_dataset('image_position_labels', (2,), dtype='S7')
hdf_shapes_position_labels[...] = ['depth'.encode('utf8'),
                          'dimension'.encode('utf8')]
hdf_position.dims.create_scale(hdf_shapes_position_labels, 'shape_labels')
hdf_position.dims[0].attach_scale(hdf_shapes_position_labels)

# Attach shape annotations and scales for sax number
hdf_sax.dims.create_scale(hdf_shapes_sax, 'shapes')
hdf_sax.dims[0].attach_scale(hdf_shapes_sax)

hdf_sax_labels = h5file.create_dataset('sax_labels', (1,), dtype='S7')
hdf_sax_labels[...] = ['depth'.encode('utf8')]
hdf_sax.dims.create_scale(hdf_sax_labels, 'shape_labels')
hdf_sax.dims[0].attach_scale(hdf_sax_labels)

# Add axis annotations
hdf_features.dims[0].label = 'batch'
hdf_labels.dims[0].label   = 'batch'
hdf_sax.dims[0].label      = 'batch'
hdf_labels.dims[1].label   = 'index'
hdf_cases.dims[0].label    = 'batch'
hdf_cases.dims[1].label    = 'index'
hdf_mult.dims[0].label     = 'batch'
hdf_mult.dims[1].label     = 'index'
hdf_position.dims[0].label     = 'batch'

### loading train


# def classify_images_per_cases(liste):
#   output      = []
#   output_case = []
#   index       = -1
#   for i in range(len(liste)):
#     for j in range(len(liste[i])):
#       stri       = train_features[i][j]
#       m          = re.search('train/(.+?)/study', stri)
#       case_index = int(m.group(1))
#       if (index == -1):
#         index = case_index
#       if case_index != index:
#         output.append(output_case)
#         output_case = []
#         index = case_index
#       d, multiplier = get_data(sequence, lambda x: x)
#       output_case.append(d[0])
#   output.append(output_case)
#   return output
index_list        = []
index             = 1
index_list.append(index)
images_output     = []
multiplier_output = []
cases_output      = []
output_ind        = []
sax_indexes       = []
sax_indexes_tmp   = []
positions = []
positions_tmp = []
i = 0
with progress_bar('train', n_examples_train) as bar:
    for sequence in train_features:
        stri          = sequence[0]
        m             = re.search('train/(.+?)/study', stri)
        case_index    = int(m.group(1))
        if case_index != index:
            sax_indexes.append(list(numpy.unique(numpy.array(sax_indexes_tmp))))
            cases_output.append(index)
            images_output.append(output_ind)
            output_ind   = []
            sax_indexes_tmp = []
            positions.append(positions_tmp)
            positions_tmp = []
            index       = case_index
            index_list.append(index)
            multiplier_output.append(multiplier)
        m_sax         = re.search('study/sax_(.+?)/IM-', stri)
        d, multiplier, position = get_data(sequence, lambda x: x)
        sax_indexes_tmp.append(int(m_sax.group(1)))
        positions_tmp.append(position)
        images        = numpy.array(d[0])
        output_ind.append(images)
        bar.update(i)
        i+=1
images_output.append(output_ind)
sax_indexes.append(list(numpy.unique(numpy.array(sax_indexes_tmp))))
cases_output.append(case_index)
multiplier_output.append(multiplier)
positions.append(positions_tmp)

j = 0
for i in range(len(images_output)):
  if len(sax_indexes[i]) > 1:
        try:
            im              = numpy.array(images_output[i])
        except ValueError:
            im              = crop_sequence(images_output[i])
        hdf_features[j]   = im.flatten().astype(numpy.dtype('uint16'))
        hdf_shapes[j]     = im.shape
        hdf_sax[j]        = sax_indexes[i]
        hdf_shapes_sax[j] = im.shape[0]
        hdf_mult[j]       = multiplier_output[i]
        hdf_labels[j]     = numpy.array(labels[cases_output[i]])
        hdf_cases[j]      = cases_output[i]
        hdf_position[j]   = numpy.array(positions[i]).ravel()
        hdf_shapes_position[j] = numpy.array(positions[i]).shape
        j += 1
n_examples_train = j

### loading submit
index             = numpy.max(index_list) + 1
index_list        = []
index_list.append(index)
images_output     = []
multiplier_output = []
cases_output      = []
output_ind        = []
sax_indexes       = []
sax_indexes_tmp   = []
positions = []
positions_tmp = []

i = 0
with progress_bar('submit', n_examples_submit) as bar:
    for sequence in submit_features:
        stri          = sequence[0]
        m             = re.search('validate/(.+?)/study', stri)
        case_index    = int(m.group(1))
        if case_index != index:
            sax_indexes.append(list(numpy.unique(numpy.array(sax_indexes_tmp))))
            cases_output.append(index)
            images_output.append(output_ind)
            output_ind   = []
            sax_indexes_tmp = []
            positions.append(positions_tmp)
            positions_tmp = []
            index       = case_index
            index_list.append(index)
            multiplier_output.append(multiplier)
        m_sax         = re.search('study/sax_(.+?)/IM-', stri)
        d, multiplier, position = get_data(sequence, lambda x: x)
        sax_indexes_tmp.append(int(m_sax.group(1)))
        positions_tmp.append(position)
        images        = numpy.array(d[0])
        output_ind.append(images)
        bar.update(i)
        i+=1
images_output.append(output_ind)
sax_indexes.append(list(numpy.unique(numpy.array(sax_indexes_tmp))))
cases_output.append(case_index)
multiplier_output.append(multiplier)
positions.append(positions_tmp)

for i in range(len(images_output)):
  if len(sax_indexes[i]) > 1:
        try:
            im              = numpy.array(images_output[i])
        except ValueError:
            im              = crop_sequence(images_output[i])
        hdf_features[j]   = im.flatten().astype(numpy.dtype('uint16'))
        hdf_shapes[j]     = im.shape
        hdf_sax[j]        = sax_indexes[i]
        hdf_shapes_sax[j] = im.shape[0]
        hdf_mult[j]       = multiplier_output[i]
        hdf_cases[j]      = cases_output[i]
        hdf_position[j]   = numpy.array(positions[i]).ravel()
        hdf_shapes_position[j] = numpy.array(positions[i]).shape
        j += 1
n_total = j-1
# Add the labels
split_dict = {}
sources = ['sax_features', 'targets', 'cases', 'sax', 'multiplier', 'image_position']
for name, slice_ in zip(['train', 'submit'],
                        [(0, n_examples_train), (n_examples_train, n_total)]):
    split_dict[name] = dict(zip(sources, [slice_] * len(sources)))
h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

h5file.flush()
h5file.close()

# n_total = 691
# number_train = 494 (counting valid set)
