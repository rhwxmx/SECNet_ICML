###############author: Hongwei Ren################
# This script is used to generate the DVS-Gesture dataset in the format of h5py.
# Date: 2024-09-23
# Thanks to the reference code from GET: Group Event Transformer for Event-Based Vision.
# Thanks to the reference code from Space-time Event Clouds for Gesture Recognition: from RGB Cameras to Event Cameras.
# Thanks to the Spikingjelly repository. 
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import extractdata_uti as uti
import numpy as np
import h5py
import os
import random
from torch.utils.data import DataLoader, Subset
import open3d as o3d
import torch
from tqdm import tqdm

def read_file(dataset,NUM_POINTS,STEP_SIZE,WINDOW_SIZE,train=True):
    data= []
    labels = []
    marks = []
    for i in tqdm(range(len(dataset))):
        events,label= dataset[i]
        if label == 10:
            continue
        class_events = np.zeros(shape=(int(len(events['x'])),3),dtype=np.int64)
        class_events[:,0] = events['t']
        class_events[:,1] = events['x']
        class_events[:,2] = events['y']
        # class_events[:,3] = events['p']
        win_start_index,win_end_index = uti.get_window_index(events['t'],events['t'][0],stepsize=STEP_SIZE*1000000,windowsize = WINDOW_SIZE*1000000)
        NUM_WINDOWS = len(win_start_index)
        for n in range(NUM_WINDOWS):
            window_events = class_events[win_start_index[n]:win_end_index[n],:].copy()
            if window_events.shape[0] > 100:
                extracted_events = uti.shuffle_downsample(window_events,NUM_POINTS)
                if(len(extracted_events[:,0]) == NUM_POINTS):
                    # events_normed = uti.normaliztion(extracted_events,128,128,True)
                    events_normed = uti.normaliztion(extracted_events,128,128,False)
                    data.append(events_normed)
                    labels.append(label)
                    marks.append(i)
                if train:
                    crop_events = uti.random_crop(window_events,128,128)
                    extracted_events = uti.shuffle_downsample(crop_events,NUM_POINTS)
                    if(len(extracted_events[:,0]) == NUM_POINTS):
                        # events_normed = uti.normaliztion(extracted_events,128,128,True)
                        events_normed = uti.normaliztion(extracted_events,128,128,False)
                        data.append(events_normed)
                        labels.append(label)
                        marks.append(i)
                    

    data = np.array(data)
    labels = np.array(labels)
    marks = np.array(marks)
    return data,labels,marks

ROOT_DIR = 'D:\\dataset\\ibm-gesture\\dataset\\'
EXPORT_PATH = 'D:\\dataset\\ibm-gesture\\'
NUM_POINTS = 1024
STEP_SIZE = 0.25
WINDOW_SIZE = 0.5
train_set = DVS128Gesture(ROOT_DIR, train=True)
test_set = DVS128Gesture(ROOT_DIR, train=False)

data_train,label_train,mark_train = read_file(train_set,NUM_POINTS,STEP_SIZE,WINDOW_SIZE,train=True)
data_test,label_test,mark_test = read_file(test_set,NUM_POINTS,STEP_SIZE,WINDOW_SIZE,train=False)

data = data_train
label = label_train
mark = mark_train
print(data.shape)
print(label.shape)
print(mark.shape)
with h5py.File(os.path.join(EXPORT_PATH,"train.h5"), 'a') as hf:
    dset = hf.create_dataset('data', shape=data.shape, maxshape = (None,NUM_POINTS,3), chunks=True, dtype='float32')
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
    dset = hf.create_dataset('data', shape=data.shape, maxshape = (None,NUM_POINTS,3), chunks=True, dtype='float32')
    lset = hf.create_dataset('label',shape=label.shape, maxshape = (None), chunks=True, dtype='int16')
    mset = hf.create_dataset('mark',shape=mark.shape, maxshape = (None), chunks=True, dtype='int16')
    hf['data'][:] = data
    hf['label'][:] = label
    hf['mark'][:] = mark
