from fuel.streams import ServerDataStream
import numpy as np

train_stream = ServerDataStream(('features', 'labels'),
                                produces_examples=False)

labels_count = np.zeros((17,))
mb_count = 0

for data in train_stream.get_epoch_iterator():
    labels_count += data[1].sum(axis=0)
    mb_count += 1

print('I counted %d minibatches' % mb_count)
print('labels_count', labels_count)
print('img shape', data[0][0].shape)
print('img stats for last mini batch:')
print('min: %.2f, max: %.2f, mean: %.2f, std: %.2f' %
      (data[0].min(), data[0].max(), data[0].mean(),
       data[0].std()))
