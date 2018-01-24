import os
import h5py
from fuel.converters.base import progress_bar
import numpy
from PIL import Image
import math
from fuel.datasets.hdf5 import H5PYDataset
import numpy as np
import dicom
import os
import fnmatch
import re
import cv2



SAX_SERIES = {
    # challenge online
    "SC-HF-I-9"  : "0241",  # OBVIOUS           ['IM-0241-0278.dcm']
    "SC-HF-I-10"  : "0024",  # DONE              ['IM-0024-0220.dcm', 'IM-0034-0220.dcm']"
    "SC-HF-I-11"  : "0043",  # DONE              ['IM-0043-0220.dcm', 'IM-0047-0220.dcm']"
    "SC-HF-I-12"  : "0062",  # OBVIOUS           ['IM-0062-0220.dcm']
    "SC-HF-NI-12" : "0286",  # OBVIOUS           ['IM-0286-0260.dcm']
    "SC-HF-NI-13" : "0304", # OBVIOUS           ['IM-0304-0220.dcm']
    "SC-HF-NI-14" : "0331", # DONE              ['IM-0330-0216.dcm', 'IM-0331-0216.dcm', 'IM-0332-0216.dcm']"
    "SC-HF-NI-15" : "0359", # OBVIOUS           ['IM-0359-0220.dcm']
    "SC-HYP-9"   : "0003",   # DONE              ['IM-0003-0220.dcm', 'IM-0004-0220.dcm', 'IM-0007-0220.dcm']"
    "SC-HYP-10"   : "0579",   # TO VERIFY !! DONE ['IM-0579-0180.dcm', 'IM-0583-0180.dcm', 'IM-0592-0180.dcm']"
    "SC-HYP-11"   : "0606",   # DONE              ['IM-0601-0200.dcm', 'IM-0606-0200.dcm']"
    "SC-HYP-12"   : "0629",   # DONE              ['IM-0629-0180.dcm', 'IM-0645-0180.dcm']"
    "SC-N-9"     : "1031",     # DONE              ['IM-1031-0160.dcm', 'IM-1032-0160.dcm', 'IM-1034-0160.dcm']"
    "SC-N-10"     : "0851",     # DONE              ['IM-0851-0200.dcm', 'IM-0853-0200.dcm', 'IM-0855-0200.dcm']"
    "SC-N-11"     : "0878",     # DONE              ['IM-0877-0180.dcm', 'IM-0878-0180.dcm', 'IM-0882-0180.dcm']"
    # challenge training
    "SC-HF-I-1": "0004", #,"['IM-0004-0219.dcm', 'IM-0005-0219.dcm']"
    "SC-HF-I-2": "0106",
    "SC-HF-I-4": "0116",
    "SC-HF-I-40": "0134",
    "SC-HF-NI-3": "0379",
    "SC-HF-NI-4": "0501",
    "SC-HF-NI-34": "0446",
    "SC-HF-NI-36": "0474",
    "SC-HYP-1": "0550",
    "SC-HYP-3": "0650",
    "SC-HYP-38": "0734",
    "SC-HYP-40": "0755",
    "SC-N-2": "0898",
    "SC-N-3": "0915",
    "SC-N-40": "0944",    
    # challenge validation
    "SC-HF-I-5" :  "0156", # "0174" ok
    "SC-HF-I-6" :  "0180", # "0187" ok
    "SC-HF-I-7" :  "0209", # "0211" "0217" idk
    "SC-HF-I-8" :  "0226",
    "SC-HF-NI-7" :  "0523",
    "SC-HF-NI-11" : "0270",
    "SC-HF-NI-31" :  "0401", # "0405" ok
    "SC-HF-NI-33" : "0424", # "0428" ok
    "SC-HYP-6" : "0767", # "0768" "0770" "0771" ok
    "SC-HYP-7" : "0007", # "0011" ok
    "SC-HYP-8" : "0796", # "0797" ok
    "SC-HYP-37" :  "0702", # "0714" ok
    "SC-N-5" : "0963", # "0965" ok
    "SC-N-6" : "0984", # "0982" "0983" "0981" "0985" ok
    "SC-N-7" : "1009", # "1013" ok
}


def shrink_case(case):
    toks = case.split("-")
    def shrink_if_number(x):
        try:
            cvt = int(x)
            return str(cvt)
        except ValueError:
            return x
    return "-".join([shrink_if_number(t) for t in toks])

class Contour(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r"/([^/]*)/contours-manual/IRCCI-expert/IM-0001-(\d{4})-icontour-manual.txt", ctr_path)
        self.case = shrink_case(match.group(1))
        self.img_no = int(match.group(2))
    
    def __str__(self):
        return "<Contour for case %s, image %d>" % (self.case, self.img_no)
    
    __repr__ = __str__

def load_contour(contour, img_path):
    filename = "IM-%s-%04d.dcm" % (SAX_SERIES[contour.case], contour.img_no)
    full_path = os.path.join(img_path, contour.case, filename)
    f = dicom.read_file(full_path)
    img = f.pixel_array.astype(np.int)
    ctrs = np.loadtxt(contour.ctr_path, delimiter=" ").astype(np.int)
    label = np.zeros_like(img, dtype="uint8")
    cv2.fillPoly(label, [ctrs], 1)
    return img, label
    
def get_all_contours(contour_path):
    contours = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(contour_path)
        for f in fnmatch.filter(files, 'IM-0001-*-icontour-manual.txt')]
    #   print("Shuffle data")
    #   np.random.shuffle(contours)
    #print("Number of examples: {:d}".format(len(contours)))
    extracted = map(Contour, contours)
    return extracted

def get_all_cases(contour_path):
  cases = []
  contours = get_all_contours(contour_path)
  for ctr in contours:
    cases.append(ctr.case)
  return list(set(cases))

def number_of_cases(contour_path, case):
  contours = get_all_contours(contour_path)
  nb_cases = 0
  for ctr in contours:
    if ctr.case == case:
      nb_cases += 1
  return nb_cases

def get_data(contour_path, img_path, case, resize_param=80.):
  contours = get_all_contours(contour_path)
  nb_cases = number_of_cases(contour_path, case)
  data, label = np.zeros([nb_cases, resize_param, resize_param]), np.zeros([nb_cases, resize_param, resize_param])
  i = 0
  for ctr in contours:
    if ctr.case == case:
      im_data, im_label = load_contour(ctr, img_path)

      assert(im_data.shape == im_label.shape)

      original_height, original_width = im_data.shape
      multiplier        = max(resize_param / original_width, resize_param / original_height)

      width    = int(math.ceil(original_width * multiplier))
      height   = int(math.ceil(original_height * multiplier))

      im_data  = Image.fromarray(im_data.astype('int16'))
      im_data  = numpy.array(im_data.resize((width, height))).astype('int16')

      im_label = Image.fromarray(im_label.astype('int16'))
      im_label = numpy.array(im_label.resize((width, height))).astype('int16')

      data[i]  = np.array(im_data).astype('int16')
      label[i] = np.array(im_label).astype('int16')
      i       += 1
  return [data, label], multiplier

#n_train = 260
#n_examples_submit = 2128
train_contour_path      = "data_sunnybrook/SunnybrookCardiacMRDatabaseContoursPart3/TrainingDataContours"
train_img_path          = "data_sunnybrook/challenge_training"
online_contour_path     = "data_sunnybrook/SunnybrookCardiacMRDatabaseContoursPart1/OnlineDataContours"
online_img_path         = "data_sunnybrook/challenge_online"
validation_contour_path = "data_sunnybrook/SunnybrookCardiacMRDatabaseContoursPart2/ValidationDataContours"
validation_img_path     = "data_sunnybrook/challenge_validation"

cases_train      = get_all_cases(train_contour_path)
cases_online     = get_all_cases(online_contour_path)
cases_validation = get_all_cases(validation_contour_path)
n_train      = len(cases_train)
n_online     = len(cases_online)
n_validation = len(cases_validation)
n_total      = n_train + n_online + n_validation #n_examples_train + n_examples_submit


output_path = './data_sunnybrook/sunnybrook_heart.hdf5'
h5file = h5py.File(output_path, mode='w')
dtype = h5py.special_dtype(vlen=numpy.dtype('uint16'))

hdf_features = h5file.create_dataset('image_features', (n_total,), dtype=dtype)
hdf_labels   = h5file.create_dataset('image_targets', (n_total,), dtype=dtype)
hdf_shapes   = h5file.create_dataset('image_targets_shapes', (n_total, 3), dtype='int32')
hdf_mult     = h5file.create_dataset('multiplier', (n_total, 1), dtype='float32')
hdf_cases    = h5file.create_dataset('cases', (n_total, 1), dtype='int32')

# Attach shape annotations and scales
hdf_features.dims.create_scale(hdf_shapes, 'shapes')
hdf_features.dims[0].attach_scale(hdf_shapes)

hdf_labels.dims.create_scale(hdf_shapes, 'shapes')
hdf_labels.dims[0].attach_scale(hdf_shapes)

hdf_shapes_labels = h5file.create_dataset('image_features_labels', (3,), dtype='S7')
hdf_shapes_labels[...] = ['features'.encode('utf8'),
                          'height'.encode('utf8'),
                          'width'.encode('utf8')]

hdf_features.dims.create_scale(hdf_shapes_labels, 'shape_labels')
hdf_features.dims[0].attach_scale(hdf_shapes_labels)

hdf_labels.dims.create_scale(hdf_shapes_labels, 'shape_labels')
hdf_labels.dims[0].attach_scale(hdf_shapes_labels)

# Add axis annotations
hdf_features.dims[0].label = 'batch'
hdf_labels.dims[0].label   = 'batch'
hdf_cases.dims[0].label    = 'batch'
hdf_cases.dims[1].label    = 'index'
hdf_mult.dims[0].label     = 'batch'
hdf_mult.dims[1].label     = 'index'

### loading train
i = 0

with progress_bar('train_data ', n_train) as bar:
  for c in cases_train:
    [d,l], m = get_data(train_contour_path, train_img_path, c)
    train_images    = numpy.array(d)
    label_images    = numpy.array(l)
    assert(train_images.shape == label_images.shape)
    hdf_shapes[i]   = train_images.shape
    hdf_features[i] = train_images.flatten().astype(numpy.dtype('uint16'))
    hdf_mult[i]     = m
    hdf_labels[i]   = label_images.flatten().astype(numpy.dtype('uint16'))
    hdf_cases[i]    = i
    i += 1
    bar.update(i)

with progress_bar('online_data ', n_online) as bar:
  for c in cases_online:
    [d,l], m = get_data(online_contour_path, online_img_path, c)
    train_images    = numpy.array(d)
    label_images    = numpy.array(l)
    assert(train_images.shape == label_images.shape)
    hdf_shapes[i]   = train_images.shape
    hdf_features[i] = train_images.flatten().astype(numpy.dtype('uint16'))
    hdf_mult[i]     = m
    hdf_labels[i]   = label_images.flatten().astype(numpy.dtype('uint16'))
    hdf_cases[i]    = i
    i += 1
    bar.update(i - n_train)

with progress_bar('validation_data ', n_validation) as bar:
  for c in cases_validation:
    [d,l], m = get_data(validation_contour_path, validation_img_path, c)
    train_images    = numpy.array(d)
    label_images    = numpy.array(l)
    assert(train_images.shape == label_images.shape)
    hdf_shapes[i]   = train_images.shape
    hdf_features[i] = train_images.flatten().astype(numpy.dtype('uint16'))
    hdf_mult[i]     = m
    hdf_labels[i]   = label_images.flatten().astype(numpy.dtype('uint16'))
    hdf_cases[i]    = i
    i += 1
    bar.update(i - n_online - n_train)


# Add the labels
split_dict = {}
sources = ['image_features', 'image_targets', 'cases', 'multiplier']
for name, slice_ in zip(['train'],
                        [(0, n_total)]):
    split_dict[name] = dict(zip(sources, [slice_] * len(sources)))
h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

h5file.flush()
h5file.close()
