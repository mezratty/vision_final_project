""" Load mat files for easy processing with tensorflow"""
import glob
import scipy.io as sio
import numpy as np

# Number of features is points detected * 2 because of x and y coordinates
NUM_FEATURES = 18 * 2
# Number of frames is the number of frames per movement returned to us
NUM_FRAMES = 80

def get_one_keypoint(num_trials, joint_index):
    path = 'mat/*.mat'
    # Sorted list of files so that we know what action is being loaded to what index
    list_files = sorted(glob.glob(path))
    assert num_trials <= 59
    assert joint_index < NUM_FEATURES
 
    matrix = np.zeros([num_trials, NUM_FRAMES])
    for i in range(num_trials):
        fname = list_files[i]
        print(fname)
        mat_contents = sio.loadmat(fname)
        matrix[i, :] = mat_contents['keypoints'][joint_index, :]
 
    return matrix

def get_labels(num_trials):
    path = 'mat/*.mat'
    # Sorted list of files so that we know what action is being loaded to what index
    list_files = sorted(glob.glob(path))
    assert num_trials <= 59
    assert joint_index < NUM_FEATURES
    
    matrix = np.zeros([num_trials, NUM_FRAMES])
    for i in range(num_trials):
        fname = list_files[i]
        print(fname)
        mat_contents = sio.loadmat(fname)
        matrix[i, :] = mat_contents['label']
 
    return matrix
