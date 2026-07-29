###############author: Hongwei Ren################
# This script is used to generate the DVS-CIFAR10 dataset in the format of h5py.
# Date: 2024-09-23
# Thanks to the reference code from GET: Group Event Transformer for Event-Based Vision.
# Thanks to the reference code from Space-time Event Clouds for Gesture Recognition: from RGB Cameras to Event Cameras.
# Thanks to the Spikingjelly repository. 
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
import extractdata_uti as uti
import numpy as np
import h5py
import os
import random
from torch.utils.data import DataLoader, Subset
import torch
from tqdm import tqdm

def read_file(dataset,NUM_POINTS,STEP_SIZE,WINDOW_SIZE,train=True):
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
        win_start_index,win_end_index = uti.get_window_index(events['t'],events['t'][0],stepsize=STEP_SIZE*1000000,windowsize = WINDOW_SIZE*1000000)
        NUM_WINDOWS = len(win_start_index)
        for n in range(NUM_WINDOWS):#NUM_WINDOWS
            window_events = class_events[win_start_index[n]:win_end_index[n],:].copy()
            window_events = window_events.astype(float)
            window_events[:,0]/=1000
            window_events[:,1]/=4
            window_events[:,2]/=4
            window_events = window_events.astype(int)
            _,unique_x_y_combinations = np.unique(window_events, axis=0,return_index=True) 
            window_events = window_events[unique_x_y_combinations]
            window_events[:,0]*=1000
            window_events[:,1]*=4
            window_events[:,2]*=4
            window_events = window_events.astype(float)
            if window_events.shape[0] > 100:
                extracted_events = uti.shuffle_downsample(window_events,NUM_POINTS)
                if(len(extracted_events[:,0]) == NUM_POINTS):
                    events_normed = uti.normaliztion(extracted_events,128,128,True)
                    # print(events_normed)
                    data.append(events_normed)
                    labels.append(label)
                    marks.append(i)
            if train:
                crop_events = uti.random_crop(window_events,128,128)
                extracted_events = uti.shuffle_downsample(crop_events,NUM_POINTS)
                if(len(extracted_events[:,0]) == NUM_POINTS):
                    events_normed = uti.normaliztion(extracted_events,128,128,True)
                    data.append(events_normed)
                    labels.append(label)
                    marks.append(i)
                reverse_events = uti.reverse_T(window_events)
                extracted_events = uti.shuffle_downsample(reverse_events,NUM_POINTS)
                if(len(extracted_events[:,0]) == NUM_POINTS):
                    events_normed = uti.normaliztion(extracted_events,128,128,True)
                    data.append(events_normed)
                    labels.append(label)
                    marks.append(i)

    data = np.array(data)
    labels = np.array(labels)
    marks = np.array(marks)
    return data,labels,marks

NUM_POINTS = 10240
root_dir = '/root/cvpr/FECNet/data/dvscifar10/'
EXPORT_PATH = '/root/cvpr/FECNet/data/dvscifar10/10240_aug_0.1/'
STEP_SIZE = 0.1
WINDOW_SIZE = 0.1
dataset = CIFAR10DVS(root = root_dir,data_type='event')
total_size = len(dataset)
indices = np.arange(total_size)
np.random.shuffle(indices)
split = int(np.floor(0.8 * total_size))
train_indices, test_indices = indices[:split], indices[split:]
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

data_train,label_train,mark_train = read_file(train_dataset,NUM_POINTS,STEP_SIZE,WINDOW_SIZE,True)
data_test,label_test,mark_test = read_file(test_dataset,NUM_POINTS,STEP_SIZE,WINDOW_SIZE,False)


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