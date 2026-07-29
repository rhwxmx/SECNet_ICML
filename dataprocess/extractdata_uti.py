"""
Supporting functions to generate train/test data
Author: Wang Qinyi
Date: Jan 2018
"""
import sys
import os
import h5py
import numpy as np
import fnmatch
import csv
from sklearn.neighbors import KDTree
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
# from PyAedatTools.ImportAedat import ImportAedat
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch
import random
def train_file_name(i,lstm=False):
    if lstm:
        return 'train_lstm_' + str(i) + '.h5'
    else:
        return 'train_' + str(i) + '.h5'

def test_file_name(i):
    return 'test_' + str(i) + '2'+'.h5'


def get_file_list(PATH):
    data = []
    timelabel= []
    with open(PATH) as csvfile:
        csvreader = csv.reader(csvfile, delimiter = ',')
        for row in csvreader:
            data.append(row[0])
            timelabel.append(row[0][0:-6]+'_labels.csv')
    return data, timelabel

def get_export_path(PATH, NUM_CLASSES, WINDOW_SIZE, STEP_SIZE, TIMESTEP, NUM_POINTS):
    '''
    generate export path
    '''
    os.chdir(PATH)
    foldername1 = 'C'+str(NUM_CLASSES)+'_TS'+str(TIMESTEP)+'_'+str(NUM_POINTS)
    foldername2 = 'W'+str(WINDOW_SIZE).replace('.','')+'S'+str(STEP_SIZE).replace('.','')
    if os.path.exists(foldername1):
        os.chdir(foldername1)
        if os.path.exists(foldername2):
            os.chdir(foldername2)
            return os.getcwd() 
        else:
            os.mkdir(foldername2)
            os.chdir(foldername2)
            return os.getcwd() 
    else:
        os.mkdir(foldername1)
        os.chdir(foldername1)
        if os.path.exists(foldername2):
            os.chdir(foldername2)
            return os.getcwd() 
        else:
            os.mkdir(foldername2)
            os.chdir(foldername2)
            return os.getcwd() 
              
    
def get_class_index(events,start_timelabel,end_timelabel):
    """
    Extract each class from original video
    """
    start_timelabel.sort()
    end_timelabel.sort()
    class_start_index = []
    class_end_index = []
    idx = 0
    for i in range(len(end_timelabel)):
        while idx < len(events):
            if events[idx] >= start_timelabel[i]:
                start_idx = idx
                class_start_index.append(start_idx)
                #idx = len(events)
                break
            else:
                idx = idx + 1
        idx = start_idx
        while idx < len(events):
            if events[idx] >= end_timelabel[i]:
                end_idx = idx-1
                class_end_index.append(end_idx)
                idx = len(events)
                break
            else:
                idx = idx + 1 
        idx = end_idx
    return class_start_index,class_end_index

def get_window_index_cal(events,start,end):
    """
    Extract each class from original video
    """
    win_start_index = []
    win_end_index = []
    win_end_index_ = []


   
    return win_start_index,win_end_index

def get_window_index(events,start,stepsize,windowsize):
    """
    Extract each class from original video
    """
    win_start_index = []
    win_end_index = []
    win_end_index_ = []
    idx = 0
   
    while idx < len(events):
        if (events[idx] >= start)&(start+windowsize<events[-1]):
            win_start_index.append(idx)
            start = start + stepsize
        else:
            idx = idx + 1
   
    idx = len(events)-1
    end = start - stepsize + windowsize
    while idx >= 0:
        if events[idx] <= end:
            win_end_index_.append(idx)
            end = end - stepsize
        else:
            idx = idx - 1
    
    
    for j in range(len(win_start_index)):
        win_end_index.append(win_end_index_[j])
    
    win_end_index=win_end_index[::-1]
   
    return win_start_index,win_end_index

def show_events(events):
    """
    plot events
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = events[:,0]
    y = events[:,1]
    z = events[:,2]
    col = np.arange(len(x))
    ax.view_init(elev=10, azim=78)
    p = ax.scatter(x, y, z, c=col, s=1, marker='.')
    ax.set_ylim([0,128])
    ax.set_zlim([0,128])
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    start, end = ax.get_xlim()
    plt.gca().invert_xaxis()
    plt.gca().invert_zaxis()
    ax.grid(False)
  
    ax.set_xticks([])                               
    ax.set_yticks([])                               
    ax.set_zticks([])
    fig.colorbar(p)
    plt.show()


def show_events_2D(events):
    index = np.zeros(shape=(events.shape[0], 1),dtype=np.int32)
    for i in range(events.shape[0]):
        index[i] = (events[i,2]*128 - 1) * 128 + (events[i,1]*128-1)
            
    frequency = np.zeros(shape=(128 * 128, 1), dtype=np.int32)
    for i in range(index.shape[0]):
        frequency[index[i]] += 1
    
    new_image = np.zeros(shape=(128, 128), dtype=np.int32)
    for i in range(128):
        for j in range(128):
            new_image[i,j] = frequency[i*128+j]
    plt.imshow(new_image)
    plt.show()


def shuffle_downsample(data,num=None):
    ''' data is a numpy array '''
    if num == None:
        idx = np.arange(data.shape[0])
        np.random.shuffle(idx)
    elif num > data.shape[0]:
        # idx = bootstrap_resample(data.shape[0],num)
        idx = np.random.choice(np.arange(data.shape[0]), size=num, replace=True)
        idx.sort()
        # print(idx)
    else:
        # print(data.shape[0],num,"kkkk")
        idx = np.arange(data.shape[0])
        np.random.shuffle(idx)
        idx = idx[0:num]
        idx.sort()
    
    return data[idx,...]

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
    # print(min_t,max_t)
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

def normaliztion(orinal_events,w,h,process_p = False):
    """
    Normalize events.
    """
    events = orinal_events.copy()
    events = events.astype('float32')
    events[:, 0] = (events[:, 0] - events[:, 0].min(axis=0)) / (events[:, 0].max(axis=0) - events[:, 0].min(axis=0)+1e-6)
    events[:, 1] = events[:, 1] / w
    events[:, 2] = events[:, 2] / h
    if process_p:
        events[:, 3] = events[:, 3]*2-1
    return events
    