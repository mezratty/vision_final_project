""" Data for training and testing """
from __future__ import division
import random
import glob
import scipy.io as sio
import numpy as np

# Number of features is points detected * 2 because of x and y coordinates
NUM_FEATURES = 15 * 2
# Number of frames is the number of frames per movement returned to us
NUM_FRAMES = 80

def get_movement_data(folder_name):
    path = folder_name + '/*.json'
    list_files = glob.glob(path)
    matrix = np.zeros(30, NUM_FRAMES)
    for i in range(NUM_FRAMES):
        # Later will want to edit to choose center 80 frames
        fname = list_files[i]
        

    length = len(list_files)
    train_files = list_files[:length-100]
    test_files = list_files[length-100:length]

def get_maia_turn():
    get_movement_data('maia_turn')
    

    return train_files, test_files

# def next_train_batch(batch_size):
#     """ Get next training batch """
#     data = np.zeros((batch_size, INTERVAL_SIZE))
#     labels = np.zeros(batch_size)
#     m_dict = init_dict()
#     train, _ = split_data()
#     for i in range(batch_size):
#         # Randomly chooses a file from the training data
#         fname = random.choice(train)
#         # Loads the contents of the .mat file
#         mat_contents = sio.loadmat(fname)
#         # Choose random audio interval from the given .mat file
#         aud_length = len(mat_contents['aud'])
#         start_index = random.randint(0, aud_length - INTERVAL_SIZE)
#         wave_form = mat_contents['aud'][start_index:start_index+INTERVAL_SIZE]
#         data[i, :] = np.ravel(wave_form)
#         time = (start_index + INTERVAL_SIZE) / (2* FREQUENCY)
#         index = binary_search(mat_contents['intervals'], time)
#         key = mat_contents['phonemes'][0, index][0]
#         labels[i] = m_dict.get(key)

#     return data, labels
