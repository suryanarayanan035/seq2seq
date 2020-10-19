import csv
import numpy as np
import tensorflow as tf

def read_file(filename):
    dataset=[]
    with open(filename) as csvfile:
        csvfile = csv.reader(csvfile,delimiter="|")
        for rows in csvfile:
            dataset.append([[rows[0]],[rows[1]]])
    return dataset
        
def get_dataset(filename):
    ds=read_file(filename)
    dataset=[]
    for data in ds:
        data=np.array(data)
        data = tf.ragged.constant(data)
        dataset.append(data)
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    print(ds[0])
    return dataset
