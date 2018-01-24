from config import path_csv
import csv
import numpy as np

all_labels = []

with open(path_csv, 'rb') as csvfile:
    csvbody = csv.reader(csvfile, delimiter=',')
    for row in csvbody:
        all_labels.extend(row[1].split(' '))

all_labels.pop(0)
all_labels, counts = np.unique(all_labels, return_counts=True)

counts, all_labels = zip(*sorted(zip(counts, all_labels), reverse=True))
for label, count in zip(all_labels, counts):
    print label, count

