###############author: Hongwei Ren################
# This script is used to generate the NCAR dataset in the format of h5py.
# Date: 2025-03-18
# Thanks to the reference code from GET: Group Event Transformer for Event-Based Vision.
# Thanks to the reference code from Space-time Event Clouds for Gesture Recognition: from RGB Cameras to Event Cameras
import scipy.io
from spikingjelly.datasets.n_caltech101 import NCaltech101
import extractdata_uti as uti
import numpy as np
import h5py
import os
import random
from torch.utils.data import DataLoader, Subset
import torch
from tqdm import tqdm
import numpy as np
import open3d as o3d
import random

def read_file(dataset,NUM_POINTS,train=True):
    data= []
    labels = []
    marks = []
    for i in tqdm(range(len(dataset[0]))):
        events = dataset[0][i][0]
        label = dataset[0][i][1][0][0]
        if events.shape[0] > 100:
            # print("after",sub_events.shape)
            if events.shape[0] > 15000:
                sub_events = events.copy()
                sub_events = sub_events.astype(float)
                sub_events[:,0]/=10000
                # sub_events[:,1]/=4
                # sub_events[:,2]/=4
                sub_events = sub_events.astype(int)
                _,unique_x_y_combinations = np.unique(sub_events, axis=0,return_index=True) 
                sub_events = sub_events[unique_x_y_combinations]
                sub_events[:,0]*=10000
                # sub_events[:,1]*=4
                # sub_events[:,2]*=4
                sub_events = sub_events.astype(float)
                print(events.shape,sub_events.shape)
                events = sub_events
                ###no polarity
                # events[:,3] = 0
            try:
                extracted_events = uti.shuffle_downsample(events,NUM_POINTS)
                if(len(extracted_events[:,0]) == NUM_POINTS):
                    events_normed = uti.normaliztion(extracted_events,128,128,False)
                    data.append(events_normed)
                    labels.append(label)
                    marks.append(i)
                if train:
                    crop_events = uti.random_crop(events,128,128)
                    extracted_events = uti.shuffle_downsample(crop_events,NUM_POINTS)
                    if(len(extracted_events[:,0]) == NUM_POINTS):
                        events_normed = uti.normaliztion(extracted_events,128,128,False)
                        data.append(events_normed)
                        labels.append(label)
                        marks.append(i)
                    reverse_events = uti.reverse_T(events)
                    extracted_events = uti.shuffle_downsample(reverse_events,NUM_POINTS)
                    if(len(extracted_events[:,0]) == NUM_POINTS):
                        events_normed = uti.normaliztion(extracted_events,128,128,False)
                        data.append(events_normed)
                        labels.append(label)
                        marks.append(i)
                    fliph_events = uti.flip_H(events,128)
                    extracted_events = uti.shuffle_downsample(fliph_events,NUM_POINTS)
                    if(len(extracted_events[:,0]) == NUM_POINTS):
                        events_normed = uti.normaliztion(extracted_events,128,128,False)
                        data.append(events_normed)
                        labels.append(label)
                        marks.append(i)
                    flipw_events = uti.flip_W(events,128)
                    extracted_events = uti.shuffle_downsample(flipw_events,NUM_POINTS)
                    if(len(extracted_events[:,0]) == NUM_POINTS):
                        events_normed = uti.normaliztion(extracted_events,128,128,False)
                        data.append(events_normed)
                        labels.append(label)
                        marks.append(i)
            except:
                pass

    data = np.array(data)
    labels = np.array(labels)
    marks = np.array(marks)
    return data,labels,marks

NUM_POINTS = 8192
root_dir = 'D:\\dataset\\NCARS\\'
EXPORT_PATH = 'D:\\dataset\\NCARS\\point\\'
train_mat_filename = root_dir+"train_data.mat"
test_mat_filename = root_dir+"test_data.mat"

data_train = scipy.io.loadmat(train_mat_filename)
data_test = scipy.io.loadmat(test_mat_filename)
data_train = data_train['data_train']
data_test = data_test['data_test']

data_train,label_train,mark_train = read_file(data_train,NUM_POINTS,True)
data_test,label_test,mark_test = read_file(data_test,NUM_POINTS,False)

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


