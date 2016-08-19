from blocks.serialization import load
from theano import tensor, function

# theano variables
features_car_cat = tensor.dmatrix('features_car_cat')
features_car_int = tensor.dmatrix('features_car_int')
features_nocar_cat = tensor.dmatrix('features_nocar_cat')
features_nocar_int = tensor.dmatrix('features_nocar_int')
features_cp = tensor.imatrix('codepostal')
features_hascar = tensor.imatrix('features_hascar')

main_loop = load(open("./model", "rb"))
model = main_loop.model

f = model.get_theano_function()

from fuel.datasets.hdf5 import H5PYDataset

submit_set = H5PYDataset(
    './data/data.hdf5',
    which_sets=('submit',),
    load_in_memory=True
)

print model.inputs
print submit_set.provides_sources
m = []
for i in model.inputs:
    m.append(submit_set.provides_sources.index(i.name))

from fuel.schemes import SequentialScheme
from fuel.streams import DataStream

submit_stream = DataStream.default_stream(
    submit_set,
    iteration_scheme=SequentialScheme(submit_set.num_examples, batch_size=5000)
)

i = 300001
output_file = open('submit.csv', 'w')
output_file.write('ID;COTIS\n')
for d in submit_stream.get_epoch_iterator():
    output = f(d[m[0]], d[m[1]], d[m[2]], d[m[3]], d[m[4]], d[m[5]], d[m[6]])
    
    for estim in output[1][:,0]:
        output_file.write('%d;%.3f\n' % (i, estim))
        i += 1
    
output_file.close()
