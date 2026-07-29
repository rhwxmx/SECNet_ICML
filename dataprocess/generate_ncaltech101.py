###############author: Hongwei Ren################
# This script is used to generate the N-caltech101 dataset in the format of h5py.
# Date: 2024-09-23
# Thanks to the reference code from GET: Group Event Transformer for Event-Based Vision.
# Thanks to the reference code from Space-time Event Clouds for Gesture Recognition: from RGB Cameras to Event Cameras.
# Thanks to the Spikingjelly repository. 
from spikingjelly.datasets.n_caltech101 import NCaltech101
import extractdata_uti as uti
import numpy as np
import h5py
import os
import random
from torch.utils.data import DataLoader, Subset
import torch
from tqdm import tqdm
random.seed(1234)

def read_file(dataset,NUM_POINTS,train=True):
    data= []
    labels = []
    marks = []
    for i in tqdm(range(len(dataset))):
        events,label= dataset[i]
        class_events = np.zeros(shape=(int(len(events['x'])),4),dtype=np.int64)
        class_events[:,0] = events['t']
        class_events[:,1] = events['x']
        class_events[:,2] = events['y']
        class_events[:,3] = events['p']
        time_begin_index = [35,135,235]
        time_end_index = [65,165,265]
        for j in range(len(time_begin_index)):
            differences = np.abs(events['t'] - time_begin_index[j]*1000)
            win_start_index = np.argmin(differences)
            differences = np.abs(events['t'] - time_end_index[j]*1000)
            win_end_index = np.argmin(differences)
            sub_events = class_events[win_start_index:win_end_index,:]
            sub_events = sub_events.astype(float)
            sub_events[:,0]/=1000
            sub_events = sub_events.astype(int)
            _,unique_x_y_combinations = np.unique(sub_events, axis=0,return_index=True) 
            sub_events = sub_events[unique_x_y_combinations]
            sub_events[:,0]*=1000
            sub_events = sub_events.astype(float)
            
            if sub_events.shape[0] > 100 and sub_events[-1,0] >sub_events[0,0]:
                extracted_events = uti.shuffle_downsample(sub_events,NUM_POINTS)
                if(len(extracted_events[:,0]) == NUM_POINTS):
                    events_normed = uti.normaliztion(extracted_events,240,180,True)
                    data.append(events_normed)
                    labels.append(label)
                    marks.append(i)
                if train:
                    crop_events = uti.random_crop(sub_events,240,180)
                    extracted_events = uti.shuffle_downsample(crop_events,NUM_POINTS)
                    if(len(extracted_events[:,0]) == NUM_POINTS):
                        events_normed = uti.normaliztion(extracted_events,240,180,True)
                        data.append(events_normed)
                        labels.append(label)
                        marks.append(i)
                    reverse_events = uti.reverse_T(sub_events)
                    extracted_events = uti.shuffle_downsample(reverse_events,NUM_POINTS)
                    if(len(extracted_events[:,0]) == NUM_POINTS):
                        events_normed = uti.normaliztion(extracted_events,240,180,True)
                        data.append(events_normed)
                        labels.append(label)
                        marks.append(i)
    data = np.array(data)
    labels = np.array(labels)
    marks = np.array(marks)
    print(data.shape,labels.shape,marks.shape)
    return data,labels,marks

NUM_POINTS = 10240
root_dir = 'D:\\dataset\\N-Caltech101\\'
EXPORT_PATH = 'D:\\dataset\\N-Caltech101\\point\\'
dataset = NCaltech101(root=root_dir,data_type='event')
total_size = len(dataset)
indices = np.arange(total_size)
np.random.shuffle(indices)
split = int(np.floor(0.8 * total_size))
train_indices, test_indices = indices[:split], indices[split:]
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)
data_train,label_train,mark_train = read_file(train_dataset,NUM_POINTS,True)
data_test,label_test,mark_test = read_file(test_dataset,NUM_POINTS,False)

data = data_train
label = label_train
mark = mark_train
print(data.shape)
print(label.shape)
print(mark.shape)
with h5py.File(os.path.join(EXPORT_PATH,"train.h5"), 'a') as hf:
    dset = hf.create_dataset('data', shape=data.shape, maxshape = (None,NUM_POINTS,4), chunks=True, dtype='float32')
    lset = hf.create_dataset('label',shape=label.shape, maxshape = (None), chunks=True, dtype='int16')
    mset = hf.create_dataset('mark',shape=mark.shape, maxshape = (None), chunks=True, dtype='int16')
    hf['data'][:] = data
    hf['label'][:] = label
    hf['mark'][:] = mark

data = data_test
label = label_test
mark = mark_test
print(data.shape)
print(label.shape)
print(mark.shape)
with h5py.File(os.path.join(EXPORT_PATH,"test.h5"), 'a') as hf:
    dset = hf.create_dataset('data', shape=data.shape, maxshape = (None,NUM_POINTS,4), chunks=True, dtype='float32')
    lset = hf.create_dataset('label',shape=label.shape, maxshape = (None), chunks=True, dtype='int16')
    mset = hf.create_dataset('mark',shape=mark.shape, maxshape = (None), chunks=True, dtype='int16')
    hf['data'][:] = data
    hf['label'][:] = label
    hf['mark'][:] = mark