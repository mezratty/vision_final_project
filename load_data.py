""" Load mat files for easy processing with tensorflow"""
import glob
import scipy.io as sio
import numpy as np

# Number of features is points detected * 2 because of x and y coordinates
NUM_FEATURES = 18 * 2
# Number of frames is the number of frames per movement returned to us
NUM_FRAMES = 80
NUM_CLASSES = 6

def get_one_keypoint(num_trials, joint_index):
    path = 'mat/*.mat'
    # Note: change from before, no longer need sorted list of files b/c of labels
    list_files = glob.glob(path)
    assert num_trials <= 59
    assert joint_index < NUM_FEATURES

    move_data = np.zeros([num_trials, NUM_FRAMES])
    labels = np.zeros([num_trials, NUM_CLASSES])
    for i in range(num_trials):
        fname = list_files[i]
        print(fname)
        mat_contents = sio.loadmat(fname)
        move_data[i, :] = mat_contents['keypoints'][joint_index, :]
        labels[i, :] = mat_contents['label']
 
    return move_data, labels