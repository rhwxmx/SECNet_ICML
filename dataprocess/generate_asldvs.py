###############author: Hongwei Ren################
# This script is used to generate the ASL-DVS dataset in the format of h5py.
# Date: 2024-09-23
# Thanks to the reference code from GET: Group Event Transformer for Event-Based Vision.
# Thanks to the reference code from Space-time Event Clouds for Gesture Recognition: from RGB Cameras to Event Cameras.
# Thanks to the Spikingjelly repository. 
from spikingjelly.datasets.asl_dvs import ASLDVS
import extractdata_uti as uti
import numpy as np
import h5py
import os

def read_file(dataset,NUM_POINTS):
    data= []
    labels = []
    for i in range(len(dataset)):
        events,label= dataset[i]
        # print(len(data['x']))
        print(label)
        class_events = np.zeros(shape=(int(len(events['x'])),4),dtype=np.int64)
        class_events[:,0] = events['t']
        class_events[:,1] = events['x']
        class_events[:,2] = events['y']
        class_events[:,3] = events['p']
        extracted_events = uti.shuffle_downsample(class_events,NUM_POINTS)
        if(len(extracted_events[:,0]) == NUM_POINTS):
            # extracted_events[:,0] = extracted_events[:,0]//1000
            extracted_events[:,0] = extracted_events[:,0]-extracted_events[:,0].min(axis=0)
            events_normed = extracted_events / extracted_events.max(axis=0)
            events_normed[:,1] = extracted_events[:,1] / 240
            events_normed[:,2] = extracted_events[:,2] / 180
            events_normed[:,3] = extracted_events[:,3]*2-1
            data.append(events_normed)
            labels.append(label)
    data = np.array(data)
    labels = np.array(labels)
    return data,labels

root_dir = 'D:\\dataset\\ASLDVS\\'
NUM_POINTS = 2048
EXPORT_PATH = 'D:\\dataset\\ASLDVS\\point\\'

dataset = ASLDVS(root=root_dir,data_type='event')
full_data,full_label = read_file(dataset,NUM_POINTS)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(full_data, full_label, test_size=0.2,shuffle=True)

data = X_train
label = y_train
print(data.shape)
print(label.shape)
with h5py.File(os.path.join(EXPORT_PATH,"train.h5"), 'a') as hf:

    dset = hf.create_dataset('data', shape=data.shape, maxshape = (None,NUM_POINTS,4), chunks=True, dtype='float32')
    lset = hf.create_dataset('label',shape=label.shape, maxshape = (None), chunks=True, dtype='int16')
    hf['data'][:] = data
    hf['label'][:] = label

data = X_test
label = y_test
print(data.shape)
print(label.shape)
with h5py.File(os.path.join(EXPORT_PATH,"test.h5"), 'a') as hf:

    dset = hf.create_dataset('data', shape=data.shape, maxshape = (None,NUM_POINTS,4), chunks=True, dtype='float32')
    lset = hf.create_dataset('label',shape=label.shape, maxshape = (None), chunks=True, dtype='int16')
    hf['data'][:] = data
    hf['label'][:] = label