""" Load mat files for easy processing with tensorflow"""
import glob
import scipy.io as sio
import numpy as np

# Number of features is points detected * 2 because of x and y coordinates
NUM_FEATURES = 36
# Number of frames is the number of frames per movement returned to us
NUM_FRAMES = 80
NUM_CLASSES = 3

def train_get_one_keypoint(num_trials, joint_index):
    path = 'mat/*.mat'
    list_files = glob.glob(path)
    assert num_trials <= 59
    assert joint_index < NUM_FEATURES

    train_data = np.zeros([num_trials, NUM_FRAMES])
    labels = np.zeros([num_trials, NUM_CLASSES])
    for i in range(num_trials):
        fname = list_files[i]
        mat_contents = sio.loadmat(fname)
        train_data[i, :] = mat_contents['keypoints'][joint_index, :]
        labels[i, :] = mat_contents['label']

    return train_data, labels

def train_get_one_frame(num_trials, frame_index):
    path = 'mat/*.mat'
    list_files = glob.glob(path)
    assert num_trials <= 59
    assert frame_index < NUM_FRAMES

    train_data = np.zeros([num_trials, NUM_FEATURES])
    labels = np.zeros([num_trials, NUM_CLASSES])
    for i in range(num_trials):
        fname = list_files[i]
        mat_contents = sio.loadmat(fname)
        train_data[i, :] = mat_contents['keypoints'][:, frame_index]
        labels[i, :] = mat_contents['label']

    return train_data, labels

def train_get_three_dimension(num_trials):
    path = 'mat/*.mat'
    list_files = glob.glob(path)
    assert num_trials <= 59

    train_data = np.zeros([num_trials, NUM_FRAMES, NUM_FEATURES])
    labels = np.zeros([num_trials, NUM_CLASSES])
    for i in range(num_trials):
        for j in range(NUM_FEATURES):
            fname = list_files[i]
            mat_contents = sio.loadmat(fname)
            train_data[i, :, j] = mat_contents['keypoints'] [j, :]

        labels[i, :] = mat_contents['label']
    return train_data, labels


def test_get_one_keypoint(num_trials, joint_index):
    path = 'mat_test/*.mat'
    list_files = glob.glob(path)
    assert num_trials <= 21
    assert joint_index < NUM_FEATURES

    test_data = np.zeros([num_trials, NUM_FRAMES])
    labels = np.zeros([num_trials, NUM_CLASSES])
    for i in range(num_trials):
        fname = list_files[i]
        mat_contents = sio.loadmat(fname)
        test_data[i, :] = mat_contents['keypoints'][joint_index, :]
        labels[i, :] = mat_contents['label']

    return test_data, labels

def test_get_one_frame(num_trials, frame_index):
    path = 'mat_test/*.mat'
    list_files = glob.glob(path)
    assert num_trials <= 21
    assert frame_index < NUM_FRAMES

    test_data = np.zeros([num_trials, NUM_FEATURES])
    labels = np.zeros([num_trials, NUM_CLASSES])
    for i in range(num_trials):
        fname = list_files[i]
        mat_contents = sio.loadmat(fname)
        test_data[i, :] = mat_contents['keypoints'][:, frame_index]
        labels[i, :] = mat_contents['label']

    return test_data, labels

def test_get_three_dimension(num_trials):
    path = 'mat_test/*.mat'
    list_files = glob.glob(path)
    assert num_trials <= 21
    train_data = np.zeros([num_trials, NUM_FRAMES, NUM_FEATURES])
    labels = np.zeros([num_trials, NUM_CLASSES])
    for i in range(num_trials):
        for j in range(NUM_FEATURES):
            fname = list_files[i]
            mat_contents = sio.loadmat(fname)
            train_data[i, :, j] = mat_contents['keypoints'] [j, :]

        labels[i, :] = mat_contents['label']
    return train_data, labels


###### Getting train and test data for motion vectors #########
def train_get_motion_vector(num_trials):
    path = 'mat_motion_vector_train/*.mat'
    assert num_trials <= 59
    list_files = glob.glob(path)
    train_data = np.zeros([num_trials, NUM_FRAMES, NUM_FEATURES])
    labels = np.zeros([num_trials, NUM_CLASSES])
    for i in range(num_trials):
        for j in range(NUM_FEATURES):
            fname = list_files[i]
            mat_contents = sio.loadmat(fname)
            train_data[i, :, j] = mat_contents['keypoints'] [j, :]

        labels[i, :] = mat_contents['label']
    return train_data, labels

def test_get_motion_vector(num_trials):
    path = 'mat_motion_vector_test/*.mat'
    assert num_trials <= 21
    list_files = glob.glob(path)
    train_data = np.zeros([num_trials, NUM_FRAMES, NUM_FEATURES])
    labels = np.zeros([num_trials, NUM_CLASSES])
    for i in range(num_trials):
        for j in range(NUM_FEATURES):
            fname = list_files[i]
            mat_contents = sio.loadmat(fname)
            train_data[i, :, j] = mat_contents['keypoints'] [j, :]

        labels[i, :] = mat_contents['label']
    return train_data, labels