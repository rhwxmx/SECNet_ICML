###############author: Hongwei Ren################
# This script is used to generate the NMNIST dataset in the format of h5py.
# Date: 2024-09-23
# Thanks to the reference code from GET: Group Event Transformer for Event-Based Vision.
# Thanks to the reference code from Space-time Event Clouds for Gesture Recognition: from RGB Cameras to Event Cameras
from spikingjelly.datasets.n_mnist import NMNIST
import extractdata_uti as uti
import numpy as np
import h5py
import os
import random
import torch
def flip_W(events, W=32):
    """
    Flip events horizontally.
    """
    events[:, 1] = W - events[:, 1]
    return events

def flip_H(events, H=32):
    """
    Flip events vertically.
    """
    events[:, 2] = H - events[:, 2]
    return events

def reverse_T(events):
    """
    reverse events on timesteps.
    """
    T_max = events[:, 0].max()
    T_min = events[:, 0].min()
    events[:, 0] = T_max - events[:, 0] + T_min
    return events

def random_crop(events,w=32,h=32):
    """
    Randomly crop events in space and time.
    """
    spatial_crop_range = [0.7, (w-1) / w]
    time_crop_range=[0.6, 1.0]
    min_x, max_x = 0, w
    min_y, max_y = 0, h
    min_t, max_t = int(events[0, 0]),  int(events[-1, 0])
    events = torch.from_numpy(events)
    if random.random() > 0.5:
        # Spatial cropping
        scale = torch.rand(2) * (spatial_crop_range[1] - spatial_crop_range[0]) + spatial_crop_range[0]
        crop_size_x = int(scale[0] * (max_x - min_x))
        crop_size_y = int(scale[1] * (max_y - min_y))
        start_x = int(torch.randint(0, max_x - crop_size_x, (1,)))
        start_y = int(torch.randint(0, max_y - crop_size_y, (1,)))
        mask_x = torch.logical_and(events[:, 1] >= start_x, events[:, 1] <= start_x + crop_size_x)
        mask_y = torch.logical_and(events[:, 2] >= start_y, events[:, 2] <= start_y + crop_size_y)
        crop_mask = torch.logical_and(mask_x, mask_y)
        cropped_events = events[crop_mask]
        # Adaptive shift based on crop size
        x_shift = torch.randint(-start_x, w - start_x - crop_size_x + 1, size=(1,))
        y_shift = torch.randint(-start_y, h - start_y - crop_size_y + 1, size=(1,))
        cropped_events[:, 1] += x_shift
        cropped_events[:, 2] += y_shift    

    else:
        # Time cropping
        time_crop_range[1] = (max_t - min_t - 1) / (max_t - min_t)
        scale = torch.rand(1) * (time_crop_range[1] - time_crop_range[0]) + time_crop_range[0]
        crop_size_t = int(scale * (max_t - min_t))
        start_t = int(torch.randint(min_t, max_t - crop_size_t, (1,)))
        crop_mask = torch.logical_and(events[:, 0] >= start_t, events[:, 0] <= start_t + crop_size_t)
        cropped_events = events[crop_mask]
    
    return cropped_events.numpy()

def normaliztion(orinal_events,w,h):
    """
    Normalize events.
    """
    events = orinal_events.copy()
    events = events.astype('float32')
    events[:, 0] = (events[:, 0] - events[:, 0].min(axis=0)) / (events[:, 0].max(axis=0) - events[:, 0].min(axis=0)+1e-6)
    events[:, 1] = events[:, 1] / w
    events[:, 2] = events[:, 2] / h
    return events

def read_file(dataset,NUM_POINTS,train=True):
    data= []
    labels = []
    marks = []
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
            events_normed = normaliztion(extracted_events,32,32)
            data.append(events_normed)
            labels.append(label)
            marks.append(i)
        if train:
            crop_events = random_crop(class_events,32,32)
            extracted_events = uti.shuffle_downsample(crop_events,NUM_POINTS)
            if(len(extracted_events[:,0]) == NUM_POINTS):
                events_normed = normaliztion(extracted_events,32,32)
                print(events_normed)
                data.append(events_normed)
                labels.append(label)
                marks.append(i)
            reverse_events = reverse_T(class_events)
            extracted_events = uti.shuffle_downsample(reverse_events,NUM_POINTS)
            if(len(extracted_events[:,0]) == NUM_POINTS):
                events_normed = normaliztion(extracted_events,32,32)
                print(events_normed)
                data.append(events_normed)
                labels.append(label)
                marks.append(i)
    data = np.array(data)
    labels = np.array(labels)
    marks = np.array(marks)
    return data,labels,marks

root_dir = 'D:\\dataset\\NMNIST\\'
EXPORT_PATH = 'D:\\dataset\\NMNIST\\point\\4096_mix\\'
train_dataset = NMNIST(root=root_dir, train=True,data_type='event')
test_dataset = NMNIST(root=root_dir, train=False,data_type='event')
NUM_POINTS = 4096
data_train,label_train,mark_train = read_file(train_dataset,NUM_POINTS,train=True)
data_test,label_test,mark_test = read_file(test_dataset,NUM_POINTS,train=False)




print(data_train.shape)
print(label_train.shape)
with h5py.File(os.path.join(EXPORT_PATH,"train.h5"), 'a') as hf:

    dset = hf.create_dataset('data', shape=data_train.shape, maxshape = (None,NUM_POINTS,4), chunks=True, dtype='float32')
    lset = hf.create_dataset('label',shape=label_train.shape, maxshape = (None), chunks=True, dtype='int16')
    mset = hf.create_dataset('mark',shape=mark_train.shape, maxshape = (None), chunks=True, dtype='int16')
    hf['data'][:] = data_train
    hf['label'][:] = label_train
    hf['mark'][:] = mark_train


print(data_test.shape)
print(label_test.shape)
with h5py.File(os.path.join(EXPORT_PATH,"test.h5"), 'a') as hf:

    dset = hf.create_dataset('data', shape=data_test.shape, maxshape = (None,NUM_POINTS,4), chunks=True,dtype='float32')
    lset = hf.create_dataset('label',shape=label_test.shape, maxshape = (None), chunks=True,dtype='int16')
    mset = hf.create_dataset('mark',shape=mark_test.shape, maxshape = (None), chunks=True, dtype='int16')
    hf['data'][:] = data_test
    hf['label'][:] = label_test
    hf['mark'][:] = mark_test