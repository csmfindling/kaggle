from models.simple_conv_seq import build_model
from blocks.serialization import load_parameters, load
import theano
from theano import tensor
from blocks.model import Model
import sys
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.streams import DataStream
sys.path.append('./functions/')
from custom_transformers import Normalize, Cast
from fuel.converters.base import progress_bar
import numpy

# load model
images = tensor.ftensor4('images')
images_test = tensor.ftensor4('images_test')
labels = tensor.imatrix('labels')

#parameters = load_parameters(open("./train", "rb"))
#model = Model(cost)
models = []
for i in range(4):
    main_loop = load(open("./trained_%d" % ((i+1)*2, ), "rb"))
    models.append(main_loop.model)

#model.set_parameter_values(parameters)
functions = [theano.function([images], model.get_top_bricks()[0].apply(images)) for model in models]

# load data
submit_set = H5PYDataset('../data/data_1.hdf5', which_sets=('submit',))
submit_stream = DataStream.default_stream(
    submit_set,
    iteration_scheme=SequentialScheme(submit_set.num_examples, 50)
)
submit_stream = Normalize(submit_stream)
submit_stream = Cast(submit_stream, 'floatX')
submit_it= submit_stream.get_epoch_iterator()

output = open('submission.csv', 'w')
output.write('img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n')

# i = 0
# for data in submit_it:
#     d = [f(data[1]) for f in functions]
#     means = (d[0] + d[1] + d[2] + d[3]) / 4
# 
#     for driver_id, pred in zip(data[0], means):
# 
#         output.write('img_' + str(driver_id[0]) + '.jpg,' + ','.join(['%.3f' % p for p in pred]))
#         output.write('\n')
#         i += 1
#         
#     print 'done %d/%d' % (i, submit_set.num_examples)

i = 0
for data in submit_it:
   d = [f(data[1]) for f in functions]
   d = numpy.array(d)
   best = numpy.argmax(numpy.max(d, axis=2), axis=0)
   means = d[best,numpy.arange(d.shape[1])]

   for j, driver_id in enumerate(data[0]):
       pred = means[j,:]

       output.write('img_' + str(driver_id[0]) + '.jpg,' + ','.join(['%.3f' % p for p in pred]))
       output.write('\n')
       i += 1
       
   print 'done %d/%d' % (i, submit_set.num_examples)

output.close()
